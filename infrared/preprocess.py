import os
import multiprocessing
import numpy as np
import cv2


NUM_FRAME_SAMPLE = 16

video_dir = '../PKUMMDv1/Data/IR_VIDEO'
label_dir = '../PKUMMDv1/Label'
infrared_dir = '../PKUMMDv1/Data/Infrared'
if not os.path.exists(infrared_dir):
    os.mkdir(infrared_dir)


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
    try:
        print('preprocessing data: {0} ...'.format(data_id))
        label_path = os.path.join(label_dir, data_id + '.txt')
        video_path = os.path.join(video_dir, data_id + '-infrared.avi')

        video_capture = cv2.VideoCapture(video_path)
        _, frame = video_capture.read()
        frame_id = 1
        
        num_actions, action_classes, start_frames, end_frames = parse_label(label_path)
            
        for i in range(num_actions):
            cur_action_class = action_classes[i]
            # print('  No.{:02} action, belongs to class {:02}'.format(i + 1, cur_action_class))

            cur_infrared_dir = os.path.join(infrared_dir, '{:02}'.format(cur_action_class))
            if not os.path.exists(cur_infrared_dir):
                os.mkdir(cur_infrared_dir)

            start_frame, end_frame = start_frames[i], end_frames[i]
            frame_interval = max(1, (end_frame - start_frame) // NUM_FRAME_SAMPLE)
            frame_count = 0

            while frame_id < start_frame:
                _, frame = video_capture.read()
                frame_id += 1
            
            cur_frame_id = 0
            while frame_id < end_frame:
                if (cur_frame_id + 1) % frame_interval == 0:
                    infrared_filename = data_id + '-infrared-{:02}.jpg'.format(frame_count)
                    infrared_filename_path = os.path.join(cur_infrared_dir, infrared_filename)
                    cv2.imwrite(infrared_filename_path, frame)
                    frame_count += 1
                
                frame_id += 1
                cur_frame_id += 1
                _, frame = video_capture.read()
    except Exception:
        print('{0} occurs exception...'.format(data_id))


label_filenames = sorted(os.listdir(label_dir))
i = 0 
while i < len(label_filenames):
    processes = []
    for j in range(8):
        if i >= len(label_filenames):
            continue
        data_id = label_filenames[i].split('.')[0]
        i += 1
        p = multiprocessing.Process(target=preprocess_video, args=(data_id, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
