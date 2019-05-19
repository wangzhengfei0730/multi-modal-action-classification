import os
import random
import numpy as np
import cv2


def parse_label(label_path):
    """ parse the lable file, return number of samples and each class, start and end frame

    Arg:
        label_path: str, file path of label file
    Returns:
        len(action_classes): int, number of samples
        action_classes: list(int), class of each sample
        start_frames: list(int), start_frame of each sample
        end_frames: list(int), end_frame of each sample
    """
    with open(label_path, 'r') as fp:
        content = fp.readlines()
    action_classes, start_frames, end_frames = [], [], []
    for label in content:
        label = label.split(',')
        action_classes.append(int(label[0]))
        start_frames.append(int(label[1]))
        end_frames.append(int(label[2]))
    return len(action_classes), action_classes, start_frames, end_frames


def preprocess_skeleton_frames(frames):
    """ preprocess skeleton frames into unified dimensions

    Arg:
        frames: list, array of skeleton frames
    Return:
        processed: list, skeleton frames with unified dimensions
    """
    processed = []
    num_frames = len(frames)
    # dimension
    for dim in range(3):
        sequence = []
        # sequence length
        for i in range(num_frames):
            joints = []
            # number of joints
            for j in range(25):
                persons = []
                # number of persons
                for p in range(2):
                    person = float(frames[i][p * 3 * 25 + j * 3 + dim])
                    persons.append(person)
                joints.append(persons)
            sequence.append(joints)
        processed.append(sequence)
    processed = np.array(processed)
    return np.resize(processed, (3, 128, 25, 2))


def extract_data(sample_dir, start_timestamp, end_timestamp):
    """ extract skeleton, rgb, optical_flow, depth, infrared and infrared_depth modal data from original data

    Args:
        sample_dir: str, directory path of the data
        start_timestamp: int, start frame of interested sample
        end_timestamp: int, end frame of interested sample
    Return:
        multi_modal: dict, different modal data with corresponding modal
    """
    skeleton_data_path = os.path.join(sample_dir, 'skeleton.txt')
    rgb_video_path = os.path.join(sample_dir, 'rgb.avi')
    depth_video_path = os.path.join(sample_dir, 'depth.avi')
    infrared_video_path = os.path.join(sample_dir, 'infrared.avi')

    multi_modal = dict()

    # parse skeleton
    with open(skeleton_data_path, 'r') as fp:
        lines = fp.readlines()
    skeletons = [line.strip().split(' ') for line in lines]
    interested_frame = skeletons[start_timestamp : end_timestamp + 1]
    multi_modal['skeleton'] = preprocess_skeleton_frames(interested_frame)
    
    # parse rgb and optical flow
    rgb_video_capture = cv2.VideoCapture(rgb_video_path)
    rgb_index = 0
    _, rgb_frame = rgb_video_capture.read()
    while rgb_index < start_timestamp:
        rgb_index += 1
        _, rgb_frame = rgb_video_capture.read()
    # optical flow needed
    prvs = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(rgb_frame)
    hsv[..., 1] = 255
    # sample frames
    sample_rgb_frames, sample_optical_flow_frames = [], []
    current_sample_count = 0
    while rgb_index < end_timestamp:
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
    while video_index < start_timestamp:
        video_index += 1
        _, depth_frame = depth_video_capture.read()
        _, infrared_frame = infrared_video_capture.read()
    sample_depth_frames, sample_infrared_frames, sample_infrared_depth_frames = [], [], []
    current_sample_count = 0
    while video_index < end_timestamp:
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
