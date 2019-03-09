import os
import time
import argparse
import numpy as np
import cv2
from utils import parse_label


parser = argparse.ArgumentParser(description='filter depth videos with infrared')
parser.add_argument('--threshold', default=0, type=int, help='threshold of filter')
args = parser.parse_args()

depth_dir = '../PKUMMDv1/Data/DEPTH_VIDEO'
ir_dir = '../PKUMMDv1/Data/IR_VIDEO'
label_dir = '../PKUMMDv1/Label'
filtered_dir = '../PKUMMDv1/Data/filtered'
if not os.path.exists(filtered_dir):
    os.mkdir(filtered_dir)

for label_filename in os.listdir(label_dir):
    data_id = label_filename.split('.')[0]
    print('processing data id: {0}'.format(data_id))
    label_path = os.path.join(label_dir, label_filename)
    depth_video_path = os.path.join(depth_dir, data_id + '-depth.avi')
    ir_video_path = os.path.join(ir_dir, data_id + '-infrared.avi')

    frame_id = 1
    depth_video = cv2.VideoCapture(depth_video_path)
    ir_video = cv2.VideoCapture(ir_video_path)

    num_actions, action_classes, start_frames, end_frames = parse_label(label_path)

    for i in range(num_actions):
        cur_action_class = action_classes[i]
        start_frame, end_frame = start_frames[i], end_frames[i]
        print('  No.{0} action, class is {1}...'.format(i + 1, cur_action_class))
        print('    start frame: {0}, end frame: {1}'.format(start_frame, end_frame))

        class_dir = os.path.join(filtered_dir, '{:02d}'.format(cur_action_class))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        output_dir = os.path.join(class_dir, data_id + '-{:02}'.format(i + 1))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # pass interval frames
        while frame_id < start_frame:
            _, depth_frame = depth_video.read()
            _, ir_frame = ir_video.read()
            frame_id += 1

        while frame_id < end_frame:
            ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
            filter_frame = np.zeros_like(np.array(depth_frame))

            for x in range(len(ir_gray)):
                for y in range(len(ir_gray[0])):
                    if ir_gray[x][y] >= args.threshold:
                        filter_frame[x][y] = depth_frame[x][y]
                    else:
                        filter_frame[x][y] = [0] * 3
            
            cv2.imwrite(os.path.join(output_dir, data_id + '-{0}.jpg'.format(frame_id)), filter_frame)

            frame_id += 1
            _, depth_frame = depth_video.read()
            _, ir_frame = ir_video.read()
