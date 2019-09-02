import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io
import dlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='mmod_human_face_detector.dat', type=str)
    parser.add_argument('--image', dest='image_path', help='Path of image')
    # parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Pretrained model
    saved_state_dict = torch.load(args.snapshot)
    model.load_state_dict(saved_state_dict)
    model = model.cuda()

    print('hopenet create success')

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)
    
    print('dlib face detector create success')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.eval()

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    image = cv2.imread(args.image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    dets = cnn_face_detector(image, 1)

    for idx, det in enumerate(dets):
        # Get x_min, y_min, x_max, y_max, conf
        x_min = det.rect.left()
        y_min = det.rect.top()
        x_max = det.rect.right()
        y_max = det.rect.bottom()
        conf = det.confidence      

        if conf > 1.0:
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            # To int
            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

            # Crop image
            img = image[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)          

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = img.cuda()

            yaw, pitch, roll = model(img) 

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            print('roll:', roll_predicted.item())
            print('yaw:', yaw_predicted.item())
            print('pitch:', pitch_predicted.item())

            utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
            # Plot expanded bounding box
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    cv2.imshow('res',image)
    cv2.waitKey()


