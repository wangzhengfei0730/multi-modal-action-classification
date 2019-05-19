import os
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.models.resnet import resnet18, resnet101
from skeleton_model import HCN
from data_load_utils import parse_label, preprocess_skeleton_frames


NUM_CLASSES = 51
modals = ['skeleton', 'rgb', 'optical_flow', 'depth', 'infrared', 'infrared_depth']
models_checkpoints = {
    'skeleton': (HCN(), 'checkpoints/skeleton.pt'),
    'rgb': (resnet101(num_classes=NUM_CLASSES), 'checkpoints/rgb.pt'),
    'optical_flow': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/optical_flow.pt'),
    'depth': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/depth.pt'),
    'infrared': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/infrared.pt'),
    'infrared_depth': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/infrared_depth.pt')
}

data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
rgb_image_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_data(data_dir, start_time, end_time):
    skeleton_data_path = os.path.join(data_dir, 'skeleton.txt')
    rgb_video_path = os.path.join(data_dir, 'rgb.avi')
    depth_video_path = os.path.join(data_dir, 'depth.avi')
    infrared_video_path = os.path.join(data_dir, 'infrared.avi')

    multi_modal = dict()
    multi_modal['timestamp'] = []

    # parse skeleton
    with open(skeleton_data_path, 'r') as fp:
        lines = fp.readlines()
    skeletons = [line.strip().split(' ') for line in lines]
    interested_frame = skeletons[start_time : end_time + 1]
    multi_modal['skeleton'] = preprocess_skeleton_frames(interested_frame)
    
    # parse rgb and optical flow
    multi_modal['rgb'], multi_modal['optical_flow'] = [], []
    rgb_video_capture = cv2.VideoCapture(rgb_video_path)
    rgb_index = 0
    _, rgb_frame = rgb_video_capture.read()
    while rgb_index < start_time:
        rgb_index += 1
        _, rgb_frame = rgb_video_capture.read()
    # optical flow needed
    prvs = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(rgb_frame)
    hsv[..., 1] = 255
    while rgb_index <= end_time:
        # optical flow related
        next_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        multi_modal['timestamp'].append(rgb_index)
        multi_modal['rgb'].append(rgb_frame)
        multi_modal['optical_flow'].append(bgr)
        prvs = next_frame
        rgb_index += 1
        _, rgb_frame = rgb_video_capture.read()
    
    # parse depth, infrared and depth
    multi_modal['depth'], multi_modal['infrared'], multi_modal['infrared_depth'] = [], [], []
    depth_video_capture = cv2.VideoCapture(depth_video_path)
    infrared_video_capture = cv2.VideoCapture(infrared_video_path)
    video_index = 0
    _, depth_frame = depth_video_capture.read()
    _, infrared_frame = infrared_video_capture.read()
    while video_index < start_time:
        video_index += 1
        _, depth_frame = depth_video_capture.read()
        _, infrared_frame = infrared_video_capture.read()
    while video_index <= end_time:
        video_index += 1
        _, depth_frame = depth_video_capture.read()
        _, infrared_frame = infrared_video_capture.read()
        multi_modal['depth'].append(depth_frame)
        multi_modal['infrared'].append(infrared_frame)
        # infrared + depth process
        infrared_depth_frame = np.zeros_like(np.array(infrared_frame))
        infrared_frame = np.uint8(np.clip((10 * infrared_frame + 50), 0, 255))
        depth_grayscale = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        depth_grayscale = np.uint8(np.clip((1.5 * depth_grayscale + 20), 0, 255))
        for x in range(len(depth_grayscale)):
            for y in range(len(depth_frame[0])):
                if 33 <= depth_grayscale[x][y] <= 37:
                    infrared_depth_frame[x][y] = infrared_frame[x][y]
                else:
                    infrared_depth_frame[x][y] = [0] * 3
        infrared_depth_frame = cv2.cvtColor(infrared_depth_frame, cv2.COLOR_BGR2GRAY)
        multi_modal['infrared_depth'].append(infrared_depth_frame)

    return multi_modal


def model_load_checkpoint():
    loaded_models = {}
    for modal in modals:
        model, checkpoint_path = models_checkpoints[modal]
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        loaded_models[modal] = model
    return loaded_models


def predict(models, inputs):
    choices = []
    for modal in modals:
        model, x = models[modal], inputs[modal]
        if modal == 'skeleton':
            x = torch.tensor(x).float()
            x = x.unsqueeze(0)
        else:
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
            if modal == 'rgb':
                x = rgb_image_transforms(x).float()
            else:
                x = data_transforms(x).float()
            x = x.unsqueeze(0)
        output = model(x)
        _, prediction = torch.max(output, 1)
        choices.append(prediction.item())
    return choices
