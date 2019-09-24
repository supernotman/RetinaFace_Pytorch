#-*- coding: UTF-8 -*-
import sys
sys.path.append('pose')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
from PIL import Image
import cv2
import torchvision
import torchvision_model, eval_widerface
from torchvision import transforms
from pose import hopenet, utils
import model
import os


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for head_pose.")
    parser.add_argument('--video_path', type=str, default='video_record.avi', help='Path for video to detect')
    parser.add_argument('--image_path', type=str, default='test.jpg', help='Path for image to detect')
    parser.add_argument('--out', type=str, default='out.avi', help='Path for image to detect')
    parser.add_argument('--f_model', type=str, default='model/model_epoch_190.pt', help='Path for model')
    parser.add_argument('--p_model', type=str, default='model/hopenet_robust_alpha1.pkl', help='Path for model')
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    parser.add_argument('--type', type=str, default='image', help='image or video detect', )
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    # Create retinaface
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)

    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(args.f_model)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()

    print('Retinaface create success.')

    # Create hopenet
    Hopenet = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    saved_state_dict = torch.load(args.p_model)
    Hopenet.load_state_dict(saved_state_dict)
    Hopenet = Hopenet.cuda()
    Hopenet.eval()

    print('Hopenet create success.')

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.type == 'image':
        cv2_img = cv2.imread(args.image_path)
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img)
        img = img.permute(2,0,1)

        if not args.scale == 1.0:
            size1 = int(img.shape[1]/args.scale)
            size2 = int(img.shape[2]/args.scale)
            img = resize(img.float(),(size1,size2))
        
        input_img = img.unsqueeze(0).float().cuda()
        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(input_img, RetinaFace, 
                                                        score_threshold=0.5, iou_threshold=0.3)

        np_img = img.cpu().permute(1,2,0).numpy()
        np_img.astype(int)
        img = np_img.astype(np.uint8)
            
        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box,landmark in zip(boxes,picked_landmarks[j]):
                    # Crop face
                    x_min = int(box[0])
                    x_max = int(box[2])
                    y_min = int(box[1])
                    y_max = int(box[3])
                    # Clip
                    x_min = x_min if x_min > 0 else 0
                    x_max = x_max if x_max < img.shape[1] else img.shape[1]
                    y_min = y_min if y_min > 0 else 0
                    y_max = y_max if y_max < img.shape[0] else img.shape[0]

                    if not x_min < x_max or not y_min < y_max:
                        continue

                    bbox_height = abs(y_max - y_min)
                    face_img = img[y_min:y_max, x_min:x_max]
                    # cv2.imshow('face_img', face_img)
                    # cv2.imwrite('face_img.jpg', face_img)
                    # cv2.waitKey(0)

                    face_img = Image.fromarray(face_img)

                    # Transform
                    face_img = transformations(face_img)
                    img_shape = face_img.size()
                    face_img = face_img.view(1, img_shape[0], img_shape[1], img_shape[2])
                    face_img = face_img.cuda()

                    yaw, pitch, roll = Hopenet(face_img) 

                    yaw_predicted = F.softmax(yaw)
                    pitch_predicted = F.softmax(pitch)
                    roll_predicted = F.softmax(roll)

                    # print("yaw_predicted", yaw_predicted)
                    # print("pitch_predicted", pitch_predicted)
                    # print("roll_predicted", roll_predicted)

                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                    utils.draw_axis(cv2_img, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                    cv2.rectangle(cv2_img,(box[0],box[1]),(box[2],box[3]),(255,0,255),thickness=2)

            cv2.imshow('RetinaFace-Hopenet',cv2_img)
            key = cv2.waitKey()

    else:
        # Read video
        cap = cv2.VideoCapture(args.video_path)

        codec = cv2.VideoWriter_fourcc(*'MJPG')

        width = int(cap.get(3))
        height = int(cap.get(4))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = 25.0

        out = cv2.VideoWriter(args.out, codec, fps, (width, height))

        while(True):
            ret, cv2_img = cap.read()
            img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)

            if not ret:
                print('Video open error.')
                break

            img = torch.from_numpy(img)
            img = img.permute(2,0,1)

            if not args.scale == 1.0:
                size1 = int(img.shape[1]/args.scale)
                size2 = int(img.shape[2]/args.scale)
                img = resize(img.float(),(size1,size2))

            input_img = img.unsqueeze(0).float().cuda()
            picked_boxes, picked_landmarks, _ = eval_widerface.get_detections(input_img, RetinaFace, 
                                                        score_threshold=0.5, iou_threshold=0.3)

            # np_img = resized_img.cpu().permute(1,2,0).numpy()
            np_img = img.cpu().permute(1,2,0).numpy()
            np_img.astype(int)
            img = np_img.astype(np.uint8)

            for j, boxes in enumerate(picked_boxes):
                if boxes is not None:
                    for box,landmark in zip(boxes,picked_landmarks[j]):
                        # Crop face
                        x_min = int(box[0])
                        x_max = int(box[2])
                        y_min = int(box[1])
                        y_max = int(box[3])
                        # Clip
                        x_min = x_min if x_min > 0 else 0
                        x_max = x_max if x_max < img.shape[1] else img.shape[1]
                        y_min = y_min if y_min > 0 else 0
                        y_max = y_max if y_max < img.shape[0] else img.shape[0]

                        if not x_min < x_max or not y_min < y_max:
                            continue

                        bbox_height = abs(y_max - y_min)
                        face_img = img[y_min:y_max, x_min:x_max]
                        face_img = Image.fromarray(face_img)

                        # Transform
                        face_img = transformations(face_img)
                        img_shape = face_img.size()
                        face_img = face_img.view(1, img_shape[0], img_shape[1], img_shape[2])
                        face_img = face_img.cuda()

                        yaw, pitch, roll = Hopenet(face_img) 

                        yaw_predicted = F.softmax(yaw)
                        pitch_predicted = F.softmax(pitch)
                        roll_predicted = F.softmax(roll)
                        # Get continuous predictions in degrees.
                        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        utils.draw_axis(cv2_img, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                        cv2.rectangle(cv2_img,(box[0],box[1]),(box[2],box[3]),(255,0,255),thickness=2)
                        # cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255,0,255),thickness=2)
                        cv2.circle(cv2_img,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
                        cv2.circle(cv2_img,(landmark[2],landmark[3]),radius=1,color=(0,255,0),thickness=2)
                        cv2.circle(cv2_img,(landmark[4],landmark[5]),radius=1,color=(255,0,0),thickness=2)
                        cv2.circle(cv2_img,(landmark[6],landmark[7]),radius=1,color=(0,255,255),thickness=2)
                        cv2.circle(cv2_img,(landmark[8],landmark[9]),radius=1,color=(255,255,0),thickness=2)

            out.write(cv2_img)
            cv2.imshow('RetinaFace-Pytorch',cv2_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                print('Now quit.')
                break

        cap.release()
        out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
