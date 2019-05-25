import os
import random
import traceback
import cv2
import numpy as np
from collections import Counter
from evaluation_utils import model_load_checkpoint, predict, preprocess_skeleton_frames


def get_data_ids(label_dir):
    data_ids = []
    for label_file_name in os.listdir(label_dir):
        if label_file_name[-4:] != '.txt':
            continue
        data_ids.append(label_file_name[:-4])
    return data_ids


def parse_label_file(label_dir, data_id):
    label_file_path = os.path.join(label_dir, data_id + '.txt')
    with open(label_file_path, 'r') as label_fp:
        label_contents = label_fp.readlines()
    labels, start_times, end_times = [], [], []
    for line in label_contents:
        line = line.split(',')
        labels.append(int(line[0]))
        start_times.append(int(line[1]))
        end_times.append(int(line[2]))
    return labels, start_times, end_times


def load_data(data_dir, data_id, start_time, end_time):
    skeleton_data_path = os.path.join(data_dir, '{0}.txt'.format(data_id))
    rgb_video_path = os.path.join(data_dir, '{0}.avi'.format(data_id))
    depth_video_path = os.path.join(data_dir, '{0}-depth.avi'.format(data_id))
    infrared_video_path = os.path.join(data_dir, '{0}-infrared.avi'.format(data_id))

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


def execute_vote(votes):
    counter = Counter(votes)
    winner_vote = max(counter.values())
    candidates = []
    for candidate, vote in counter.items():
        if vote == winner_vote:
            candidates.append(candidate)
    return random.choice(candidates)


if __name__ == '__main__':
    data_id = '0361-R'
    data_dir = './data_sample'

    models = model_load_checkpoint()
    print('model initialization finished...')

    num_total_cases, num_correct_cases = 0, 0
    local_record_path = 'record.txt'

    labels, start_times, end_times = parse_label_file(data_dir, 'label')
    sample_cases_indexes = random.sample(range(0, len(labels)), max(2, len(labels) // 10))
    for i, case_index in enumerate(sample_cases_indexes):
        print(' - {0}/{1} cases predicting: {2}'.format(i + 1, len(sample_cases_indexes), case_index))
        current_label = labels[case_index] - 1
        current_start_time = start_times[case_index]
        current_end_time = end_times[case_index]
        multi_modal_inputs = load_data(data_dir, data_id, current_start_time, current_end_time)
        for modal in multi_modal_inputs:
            if modal == 'skeleton':
                continue
            else:
                cv2.imwrite('{0}_{1}.jpg'.format(i + 1, modal), multi_modal_inputs[modal])
        prediction = predict(models, multi_modal_inputs)
        winner = execute_vote(prediction)
        if winner == current_label:
            num_correct_cases += 1
            num_total_cases += 1
        print('prediction:', prediction, 'winner:', winner, 'label:', current_label)
