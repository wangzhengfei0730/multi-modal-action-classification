import os
import threading, multiprocessing
import numpy as np
import cv2


NUM_FRAME_SAMPLE = 16

depth_video_dir = '../PKUMMDv1/Data/DEPTH_VIDEO'
ir_video_dir = '../PKUMMDv1/Data/IR_VIDEO'
label_dir = '../PKUMMDv1/Label'
ird_dir = '../PKUMMDv1/Data/IRD'
if not os.path.exists(ird_dir):
    os.mkdir(ird_dir)


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
        depth_video_path = os.path.join(depth_video_dir, data_id + '-depth.avi')
        ir_video_path = os.path.join(ir_video_dir, data_id + '-infrared.avi')

        depth_cap = cv2.VideoCapture(depth_video_path)
        ir_cap = cv2.VideoCapture(ir_video_path)
        _, depth_frame = depth_cap.read()
        _, ir_frame = ir_cap.read()
        frame_id = 1
        
        num_actions, action_classes, start_frames, end_frames = parse_label(label_path)
            
        for i in range(num_actions):
            cur_action_class = action_classes[i]
            # print('  No.{:02} action, belongs to class {:02}'.format(i + 1, cur_action_class))

            cur_ird_dir = os.path.join(ird_dir, '{:02}'.format(cur_action_class))
            if not os.path.exists(cur_ird_dir):
                os.mkdir(cur_ird_dir)

            start_frame, end_frame = start_frames[i], end_frames[i]
            frame_interval = max(1, (end_frame - start_frame) // NUM_FRAME_SAMPLE)
            frame_count = 0

            while frame_id < start_frame:
                _, depth_frame = depth_cap.read()
                _, ir_frame = ir_cap.read()
                frame_id += 1
                
            while frame_id < end_frame:
                if frame_id % frame_interval == 0:
                    filter_frame = np.zeros_like(np.array(ir_frame))
                    ir_frame = np.uint8(np.clip((10 * ir_frame + 50), 0, 255))
                    depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                    depth_gray = np.uint8(np.clip((1.5 * depth_gray + 20), 0, 255))
                    for x in range(len(depth_gray)):
                        for y in range(len(depth_gray[0])):
                            if 33 <= depth_gray[x][y] <= 37:
                                filter_frame[x][y] = ir_frame[x][y]
                            else:
                                filter_frame[x][y] = [0] * 3
                    filter_frame = cv2.cvtColor(filter_frame, cv2.COLOR_BGR2GRAY)
                    ird_filename = data_id + '-ird-{:02}.jpg'.format(frame_count)
                    ird_filename_path = os.path.join(cur_ird_dir, ird_filename)
                    cv2.imwrite(ird_filename_path, filter_frame)
                    frame_count += 1
                
                frame_id += 1
                _, depth_frame = depth_cap.read()
                _, ir_frame = ir_cap.read()
    except Exception:
        print('{0} occurs exception...'.format(data_id))


label_filenames = sorted(os.listdir(label_dir))
i = 0
while i < len(label_filenames):
    threads = []
    for j in range(multiprocessing.cpu_count()):
        if i >= len(label_filenames):
            continue
        data_id = label_filenames[i].split('.')[0]
        i += 1
        t = threading.Thread(target=preprocess_video, args=(data_id, ))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
