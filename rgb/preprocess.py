import os
import numpy as np
import cv2


NUM_FRAME_SAMPLE = 16

video_dir = '../PKUMMDv1/Data/RGB_VIDEO'
label_dir = '../PKUMMDv1/Label'
rgb_frame_dir = '../PKUMMDv1/Data/RGB/rgb'
optical_flow_dir = '../PKUMMDv1/Data/RGB/optical_flow'
if not os.path.exists(rgb_frame_dir):
    os.mkdir(rgb_frame_dir)
if not os.path.exists(optical_flow_dir):
    os.mkdir(optical_flow_dir)


def retrieve_content(file_path):
    with open(file_path, 'r') as fp:
        content = fp.readlines()
    return content


def parse_label(label_path):
    labels = retrieve_content(label_path)
    action_classes, start_times, end_times = [], [], []
    for label in labels:
        label = label.split(',')
        action_classes.append(int(label[0]))
        start_times.append(int(label[1]))
        end_times.append(int(label[2]))
    return len(action_classes), action_classes, start_times, end_times


def preprocess_video(data_id):
    print('preprocessing data: {0} ...'.format(data_id))
    label_path = os.path.join(label_dir, data_id + '.txt')
    video_path = os.path.join(video_dir, data_id + '.avi')

    video_capture = cv2.VideoCapture(video_path)
    _, frame = video_capture.read()
    frame_id = 1
    
    num_actions, action_classes, start_frames, end_frames = parse_label(label_path)
        
    for i in range(num_actions):
        cur_action_class = action_classes[i]
        print('  No.{:02} action, belongs to class {:02}'.format(i + 1, cur_action_class))

        cur_rgb_dir = os.path.join(rgb_frame_dir, '{:02}'.format(cur_action_class))
        cur_optical_flow_dir = os.path.join(optical_flow_dir, '{:02}'.format(cur_action_class))
        if not os.path.exists(cur_rgb_dir):
            os.mkdir(cur_rgb_dir)
        if not os.path.exists(cur_optical_flow_dir):
            os.mkdir(cur_optical_flow_dir)

        start_frame, end_frame = start_frames[i], end_frames[i]
        frame_interval = round((end_frame - start_frame) / NUM_FRAME_SAMPLE)
        frame_count = 0

        # optical flow frame
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255

        while frame_id < start_frame:
            _, frame = video_capture.read()
            frame_id += 1
            
        while frame_id < end_frame:
            if frame_id % frame_interval == 0:
                frame_count += 1
                # rgb frame
                frame_filename = data_id + '-{:02}.jpg'.format(frame_count)
                frame_file_path = os.path.join(cur_rgb_dir, frame_filename)
                cv2.imwrite(frame_file_path, frame)
                # optical flow frame
                # reference: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                optical_flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                optical_flow_filename = data_id + '-flow-{:02}.jpg'.format(frame_count)
                optical_flow_file_path = os.path.join(cur_optical_flow_dir, optical_flow_filename)
                cv2.imwrite(optical_flow_file_path, optical_flow_frame)
            prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame_id += 1
            _, frame = video_capture.read()


for label_filename in os.listdir(label_dir):
    data_id = label_filename.split('.')[0]
    preprocess_video(data_id)
