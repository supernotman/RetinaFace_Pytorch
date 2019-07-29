import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, RegressionTransform
from anchors import Anchors
import losses

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1           = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # Retinaface does not need P7
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # solve C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P6_x = self.P6(C5)

        return [P2_x, P3_x, P4_x, P5_x, P6_x]

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*2,kernel_size=(1,1),stride=1)
        #self.conv1x1 = nn.Conv2d(inchannels,num_anchors*1,kernel_size=(1,1),stride=1)

        # if use focal loss instead of OHEM
        #self.output_act = nn.Sigmoid()

        # if use OHEM
        self.output_act = nn.LogSoftmax(dim=-1)


    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous().view(out.shape[0], -1, 2)
        out = self.output_act(out)

        return out
        #return out.contiguous().view(out.shape[0], -1, 1)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 10)


class CBR(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(CBR,self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, std=0.01)      

    def forward(self,x):
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class CB(nn.Module):
    def __init__(self,inchannels):
        super(CB,self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels,inchannels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(inchannels)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, std=0.01)

    def forward(self,x):
        x = self.conv3x3(x)
        x = self.bn(x)

        return x

class Concat(nn.Module):
    def forward(self,*feature):
        out = torch.cat(feature,dim=1)
        return out

class Context(nn.Module):
    def __init__(self,inchannels=256):
        super(Context,self).__init__()
        self.context_plain = inchannels//2
        self.conv1 = CB(inchannels)
        self.conv2 = CBR(inchannels,self.context_plain)
        self.conv2_1 = CB(self.context_plain)
        self.conv2_2_1 = CBR(self.context_plain,self.context_plain)
        self.conv2_2_2 = CB(self.context_plain)
        self.concat = Concat()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        f1 = self.conv1(x)
        f2_ = self.conv2(x)
        f2 = self.conv2_1(f2_)
        f3 = self.conv2_2_1(f2_)
        f3 = self.conv2_2_2(f3)

        #out = torch.cat([f1,f2,f3],dim=1)
        out = self.concat(f1,f2,f3)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, num_anchors=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels, 
                        self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels, 
                        self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],fpn_sizes[3])

        self.context = self._make_contextlayer()       
        
        # self.clsHead = ClassHead()
        # self.bboxHead = BboxHead()
        # self.ldmHead = LandmarkHead()

        self.clsHead = self._make_class_head()
        self.bboxHead = self._make_bbox_head()
        self.ldmHead = self._make_landmark_head()

        self.anchors = Anchors()

        self.regressBoxes = RegressionTransform()
        
        self.losslayer = losses.LossLayer()

        self.freeze_bn()

        #initialization
        for layer in self.context:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        for layer in self.clsHead:
            nn.init.kaiming_normal_(layer.conv1x1.weight,mode='fan_out', nonlinearity='relu')


    def _make_contextlayer(self,fpn_num=5,inchannels=256):
        context = nn.ModuleList()
        for i in range(fpn_num):
            context.append(Context())

        return context

    def _make_class_head(self,fpn_num=5,inchannels=512,anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=5,inchannels=512,anchor_num=3):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=5,inchannels=512,anchor_num=3):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead
    
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def freeze_first_layer(self):
        '''Freeze First layer'''
        for param in self.conv1.parameters():
            param.requires_grad = False


    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x1, x2, x3, x4])
        context_features = [self.context[i](feature) for i,feature in enumerate(features)]

        bbox_regressions = torch.cat([self.bboxHead[i](feature) for i,feature in enumerate(context_features)], dim=1)
        ldm_regressions = torch.cat([self.ldmHead[i](feature) for i,feature in enumerate(context_features)], dim=1)
        classifications = torch.cat([self.clsHead[i](feature) for i,feature in enumerate(context_features)],dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.losslayer(classifications, bbox_regressions,ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)

            return classifications, bboxes, landmarks

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model