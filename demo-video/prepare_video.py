import os
import random
import traceback
from collections import Counter
from evaluation_utils import model_load_checkpoint, load_data, predict


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


def execute_vote(votes):
    counter = Counter(votes)
    winner_vote = max(counter.values())
    candidates = []
    for candidate, vote in counter.items():
        if vote == winner_vote:
            candidates.append(candidate)
    return random.choice(candidates)


if __name__ == '__main__':
    models = model_load_checkpoint()
    print('model initialization finished...')

    labels, start_times, end_times = parse_label_file('./data_sample', 'label')
    for i in range(len(labels)):
        print('action #{0}'.format(i + 1))
        current_label = labels[i] - 1
        start_time, end_time = start_times[i], end_times[i]
        multi_modal_inputs = load_data('./data_sample', start_time, end_time)

        for i, timestamp in enumerate(multi_modal_inputs['timestamp']):
            print(' timestamp #{0}:'.format(timestamp))
            multi_modal_input = {
                'skeleton': multi_modal_inputs['skeleton'],
                'rgb': multi_modal_inputs['rgb'][i],
                'optical_flow': multi_modal_inputs['optical_flow'][i],
                'depth': multi_modal_inputs['depth'][i],
                'infrared': multi_modal_inputs['infrared'][i],
                'infrared_depth': multi_modal_inputs['infrared_depth'][i]
            }
            prediction = predict(models, multi_modal_input)
            winner = execute_vote(prediction)
            print(' > label = {0}, vote winner = {1}'.format(current_label, winner))
            print(' >> prediction = {0}'.format(prediction))
    
    print('finish')
