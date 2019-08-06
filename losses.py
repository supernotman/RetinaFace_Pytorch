import numpy as numpy
import torch.nn as nn
import torch

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def filt_IoU(a, b, l):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    ldm_sum = l.sum(dim=1)
    mask = ldm_sum<0
    ldm_mask = torch.ones_like(mask)
    ldm_mask[mask] = -1
    filted_IoU = IoU * ldm_mask.float()

    return IoU, filted_IoU

class LossLayer(nn.Module):
    def __init__(self):
        super(LossLayer, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss()

    def forward(self,classifications,bbox_regressions,ldm_regressions,anchors,annotations):
        batch_size = classifications.shape[0]
        classification_losses = []
        bbox_regression_losses = []
        ldm_regression_losses = []  

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights  

        #temp
        positive_indices_list = []    

        for j in range(batch_size):
            classification = classifications[j,:,:]
            bbox_regression = bbox_regressions[j,:,:]
            ldm_regression = ldm_regressions[j,:,:]

            annotation = annotations[j,:,:]
            annotation = annotation[annotation[:,0] != -1]
            bbox_annotation = annotation[:,:4]
            ldm_annotation = annotation[:,4:]

            if bbox_annotation.shape[0] == 0:
                bbox_regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                ldm_regression_losses.append(torch.tensor(0).float().cuda())

                # temp
                positive_indices_list.append([])

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            #IoU, filt_iou = filt_IoU(anchors[0, :, :], bbox_annotation, ldm_annotation)

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            targets = torch.ones(classification.shape)*-1
            targets = targets.cuda()       
            
            # those whose iou<0.3 have no object
            negative_indices = torch.lt(IoU_max, 0.3)
            targets[negative_indices, :] = 0
            targets[negative_indices, 1] = 1

            # those whose iou>0.5 have object
            positive_indices = torch.ge(IoU_max, 0.5)

            #temp
            positive_indices_list.append(positive_indices)

            num_positive_anchors = positive_indices.sum()

            #keep positive and negative ratios with 1:3
            keep_negative_anchors = num_positive_anchors * 3

            bbox_assigned_annotations = bbox_annotation[IoU_argmax, :]
            ldm_assigned_annotations = ldm_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, 0] = 1

            # ignore targets with no landmarks
            # f_IoU_max ,f_IoU_argmax = torch.max(filt_iou, dim=1)
            # ldm_positive_indices = torch.ge(f_IoU_max, 0.5)
            
            ldm_sum = ldm_assigned_annotations.sum(dim=1)
            ge0_mask = ldm_sum > 0
            ldm_positive_indices = ge0_mask & positive_indices

            # OHEM
            negative_losses = classification[negative_indices,1] * -1
            sorted_losses, _ = torch.sort(negative_losses, descending=True)
            if sorted_losses.numel() > keep_negative_anchors:
                sorted_losses = sorted_losses[:keep_negative_anchors]
            positive_losses = classification[positive_indices,0] * -1
                
            focal_loss = False
            # focal loss
            if focal_loss:
                alpha = 0.25
                gamma = 2.0            
                alpha_factor = torch.ones(targets.shape).cuda() * alpha

                alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
                focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

                classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            else:
                if positive_indices.sum() > 0:
                    classification_losses.append(positive_losses.mean() + sorted_losses.mean())
                else:
                    classification_losses.append(torch.tensor(0).float().cuda())


            # compute bboxes loss
            if positive_indices.sum() > 0:
                # bbox
                bbox_assigned_annotations = bbox_assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = bbox_assigned_annotations[:, 2] - bbox_assigned_annotations[:, 0]
                gt_heights = bbox_assigned_annotations[:, 3] - bbox_assigned_annotations[:, 1]
                gt_ctr_x   = bbox_assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = bbox_assigned_annotations[:, 1] + 0.5 * gt_heights

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / (anchor_widths_pi + 1e-14)
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / (anchor_heights_pi + 1e-14)
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                bbox_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                bbox_targets = bbox_targets.t()

                # Rescale
                #bbox_targets = bbox_targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                # smooth L1
                # box losses
                bbox_regression_loss = self.smoothl1(bbox_targets,bbox_regression[positive_indices, :])
                bbox_regression_losses.append(bbox_regression_loss)
            else:
                bbox_regression_losses.append(torch.tensor(0).float().cuda())  

            # compute landmarks loss
            if ldm_positive_indices.sum() > 0 :
                ldm_assigned_annotations = ldm_assigned_annotations[ldm_positive_indices, :]

                anchor_widths_l = anchor_widths[ldm_positive_indices]
                anchor_heights_l = anchor_heights[ldm_positive_indices]
                anchor_ctr_x_l = anchor_ctr_x[ldm_positive_indices]
                anchor_ctr_y_l = anchor_ctr_y[ldm_positive_indices]             

                l0_x = (ldm_assigned_annotations[:,0] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                l0_y = (ldm_assigned_annotations[:,1] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)
                l1_x = (ldm_assigned_annotations[:,2] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                l1_y = (ldm_assigned_annotations[:,3] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)
                l2_x = (ldm_assigned_annotations[:,4] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                l2_y = (ldm_assigned_annotations[:,5] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)
                l3_x = (ldm_assigned_annotations[:,6] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                l3_y = (ldm_assigned_annotations[:,7] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)
                l4_x = (ldm_assigned_annotations[:,8] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                l4_y = (ldm_assigned_annotations[:,9] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)

                ldm_targets = torch.stack((l0_x,l0_y,l1_x,l1_y,l2_x,l2_y,l3_x,l3_y,l4_x,l4_y))
                ldm_targets = ldm_targets.t()

                # Rescale
                #scale = torch.ones(1,10)*0.1
                #ldm_targets = ldm_targets/scale.cuda()

                ldm_regression_loss = self.smoothl1(ldm_targets, ldm_regression[ldm_positive_indices, :])
                ldm_regression_losses.append(ldm_regression_loss)
            else:
                ldm_regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses), torch.stack(bbox_regression_losses),torch.stack(ldm_regression_losses)
        #return positive_indices_list, torch.stack(classification_losses), torch.stack(bbox_regression_losses),torch.stack(ldm_regression_losses)





