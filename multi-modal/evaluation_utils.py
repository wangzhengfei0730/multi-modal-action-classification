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


def load_data(data_dir, data_id, start_time, end_time):
    skeleton_data_path = os.path.join(os.path.join(data_dir, 'SKELETON'), '{0}.txt'.format(data_id))
    rgb_video_path = os.path.join(os.path.join(data_dir, 'RGB_VIDEO'), '{0}.avi'.format(data_id))
    depth_video_path = os.path.join(os.path.join(data_dir, 'DEPTH_VIDEO'), '{0}-depth.avi'.format(data_id))
    infrared_video_path = os.path.join(os.path.join(data_dir, 'IR_VIDEO'), '{0}-infrared.avi'.format(data_id))

    multi_modal = dict()

    # parse skeleton
    with open(skeleton_data_path, 'r') as fp:
        lines = fp.readlines()
    skeletons = [line.strip().split(' ') for line in lines]
    interested_frame = skeletons[start_time : end_time + 1]
    multi_modal['skeleton'] = preprocess_skeleton_frames(interested_frame)
    
    # parse rgb and optical flow
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
    # sample frames
    sample_rgb_frames, sample_optical_flow_frames = [], []
    current_sample_count = 0
    while rgb_index < end_time:
        # optical flow related
        next_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        prvs = next_frame
        # sampled frame
        if (current_sample_count + 1) % 16 == 0:
            sample_rgb_frames.append(rgb_frame)
            sample_optical_flow_frames.append(bgr)
        current_sample_count, rgb_index = current_sample_count + 1, rgb_index + 1
        _, rgb_frame = rgb_video_capture.read()
    multi_modal['rgb'] = random.choice(sample_rgb_frames)
    multi_modal['optical_flow'] = random.choice(sample_optical_flow_frames)
    
    # parse depth, infrared and depth
    depth_video_capture = cv2.VideoCapture(depth_video_path)
    infrared_video_capture = cv2.VideoCapture(infrared_video_path)
    video_index = 0
    _, depth_frame = depth_video_capture.read()
    _, infrared_frame = infrared_video_capture.read()
    while video_index < start_time:
        video_index += 1
        _, depth_frame = depth_video_capture.read()
        _, infrared_frame = infrared_video_capture.read()
    sample_depth_frames, sample_infrared_frames, sample_infrared_depth_frames = [], [], []
    current_sample_count = 0
    while video_index < end_time:
        if (current_sample_count + 1) % 16 == 0:
            sample_depth_frames.append(depth_frame)
            sample_infrared_frames.append(infrared_frame)
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
            sample_infrared_depth_frames.append(infrared_depth_frame)
        current_sample_count, video_index = current_sample_count + 1, video_index + 1
        _, depth_frame = depth_video_capture.read()
        _, infrared_frame = infrared_video_capture.read()
    multi_modal['depth'] = random.choice(sample_depth_frames)
    multi_modal['infrared'] = random.choice(sample_infrared_frames)
    multi_modal['infrared_depth'] = random.choice(sample_infrared_depth_frames)

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
            x = data_transforms(x).float()
            x = x.unsqueeze(0)
        output = model(x)
        _, prediction = torch.max(output, 1)
        choices.append(prediction.item())
    return choices
