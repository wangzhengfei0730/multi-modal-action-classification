import numpy as np
import cv2
from data_load_utils import parse_label

FPS = 25
ACTION_ID = 4


def split(video_path, start_timestamp, end_timestamp):
    frames = []
    cap = cv2.VideoCapture(video_path)
    current_timestamp = 0
    _, frame = cap.read()

    while current_timestamp < start_timestamp:
        current_timestamp += 1
        _, frame = cap.read()
    
    while current_timestamp <= end_timestamp:
        frames.append(frame)
        current_timestamp += 1
        _, frame = cap.read()
    
    width, height = len(frames[0][0]), len(frames[0])
    print('width = {0}, height = {1}'.format(width, height))
    
    writer = cv2.VideoWriter(
        video_path.replace('.avi', '-sliced.avi'),
        cv2.VideoWriter_fourcc('M','J','P','G'),
        FPS, 
        (width, height)
    )

    for frame in frames:
        writer.write(frame)


def split_opticalflow(rgb_video_path, start_timestamp, end_timestamp):
    frames = []
    cap = cv2.VideoCapture(rgb_video_path)
    current_timestamp = 0
    _, frame = cap.read()

    while current_timestamp < start_timestamp:
        current_timestamp += 1
        _, frame = cap.read()
    
    prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    
    while current_timestamp <= end_timestamp:
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        prvs = next

        frames.append(bgr)
        current_timestamp += 1
        _, frame = cap.read()
    
    width, height = len(frames[0][0]), len(frames[0])
    print('width = {0}, height = {1}'.format(width, height))
    
    writer = cv2.VideoWriter(
        './data_sample/opticalflow-sliced.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        FPS, 
        (width, height)
    )

    for frame in frames:
        writer.write(frame)


def split_ird(depth_video_path, infrared_video_path, start_timestamp, end_timestamp):
    frames = []
    depth_cap = cv2.VideoCapture(depth_video_path)
    infrared_cap = cv2.VideoCapture(infrared_video_path)
    current_timestamp = 0
    _, depth_frame = depth_cap.read()
    _, infrared_frame = infrared_cap.read()

    while current_timestamp < start_timestamp:
        current_timestamp += 1
        _, depth_frame = depth_cap.read()
        _, infrared_frame = infrared_cap.read()
    
    while current_timestamp <= end_timestamp:
        filter_frame = np.zeros_like(np.array(infrared_frame))
        infrared_frame = np.uint8(np.clip((10 * infrared_frame + 50), 0, 255))
        depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        depth_gray = np.uint8(np.clip((1.5 * depth_gray + 20), 0, 255))
        for x in range(len(depth_gray)):
            for y in range(len(depth_gray[0])):
                if 33 <= depth_gray[x][y] <= 37:
                    filter_frame[x][y] = infrared_frame[x][y]
                else:
                    filter_frame[x][y] = [0] * 3
        frames.append(filter_frame)
        current_timestamp += 1
        _, depth_frame = depth_cap.read()
        _, infrared_frame = infrared_cap.read()
    
    width, height = len(frames[0][0]), len(frames[0])
    print('width = {0}, height = {1}'.format(width, height))
    
    writer = cv2.VideoWriter(
        './data_sample/ird-sliced.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        FPS, 
        (width, height)
    )

    for frame in frames:
        writer.write(frame)


label_path = './data_sample/label.txt'
num_actions, action_classes, start_timestamps, end_timestamps = parse_label(label_path)

interest_action_class = action_classes[ACTION_ID]
interest_start_timestamp = start_timestamps[ACTION_ID]
interest_end_timestamp = end_timestamps[ACTION_ID]

rgb_video = './data_sample/rgb.avi'
depth_video = './data_sample/depth.avi'
infrared_video = './data_sample/infrared.avi'
split(rgb_video, interest_start_timestamp, interest_end_timestamp)
split(depth_video, interest_start_timestamp, interest_end_timestamp)
split(infrared_video, interest_start_timestamp, interest_end_timestamp)
split_opticalflow(rgb_video, interest_start_timestamp, interest_end_timestamp)
split_ird(depth_video, infrared_video, interest_start_timestamp, interest_end_timestamp)
