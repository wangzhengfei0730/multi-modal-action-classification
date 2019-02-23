import os
import time
import pickle
import numpy as np


NUM_DIMENSION = 3
NUM_JOINTS = 25
NUM_PERSON = 2
# according to statistics, reference skeleton_statistics.html
MAX_SEQUENCE_LENGTH = 800


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


def frames_preprocess(frames):
    processed = []
    num_frames = len(frames)
    for dim in range(NUM_DIMENSION):
        sequence = []
        for i in range(MAX_SEQUENCE_LENGTH):
            joints = []
            if i < num_frames:
                for j in range(NUM_JOINTS):
                    persons = []
                    for p in range(NUM_PERSON):
                        person = float(frames[i][p * NUM_DIMENSION * NUM_JOINTS + j * NUM_DIMENSION + dim])
                        persons.append(person)
                    joints.append(persons)
            else:
                for j in range(NUM_JOINTS):
                    persons = [0.0, 0.0]
                    joints.append(persons)
            sequence.append(joints)
        processed.append(sequence)
    return np.array(processed)

# assign the directory of labels and skeleton data
dataset_dir = './PKUMMDv2'
lable_dir = os.path.join(dataset_dir, 'Label')
skeleton_dir = os.path.join(dataset_dir, 'Data/skeleton')
assert os.path.exists(lable_dir), 'label directory does not exist'
assert os.path.exists(skeleton_dir), 'skeleton directory does not exist'
# output directory of processed data
processed_dir = os.path.join(dataset_dir, 'Data/skeleton_processed')
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)

for label_file in os.listdir(lable_dir):
    # avoid file like .DS_Store
    if '.txt' not in label_file:
        continue

    data_id = label_file.split('.')[0]
    label_path = os.path.join(lable_dir, label_file)
    skeleton_path = os.path.join(skeleton_dir, data_id + '.txt')

    num_actions, action_classes, start_times, end_times = parse_label(label_path)

    skeletons = retrieve_content(skeleton_path)
    skeletons = [s.strip().split(' ') for s in skeletons]

    for i in range(num_actions):
        cur_action_class = action_classes[i]
        output_dir = os.path.join(processed_dir, '{:02d}'.format(cur_action_class))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        timestamp = '{:.8f}'.format(time.time()).replace('.', '')[-12:]
        output_path = os.path.join(output_dir, data_id + '-' + timestamp + '.pkl')
        
        s_time, e_time = start_times[i], end_times[i]
        frames = skeletons[s_time:e_time + 1]
        preprocessed = frames_preprocess(frames)
        # print(preprocessed.shape)
        print('processing skeleton data: {} - NO.{} action'.format(data_id, i + 1))
        with open(output_path, 'wb') as fp:
            pickle.dump(preprocessed, fp)
