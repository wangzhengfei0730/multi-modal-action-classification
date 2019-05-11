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
    data_dir = '../PKUMMDv1/Data'
    label_dir = '../PKUMMDv1/Label'

    models = model_load_checkpoint()
    print('model initialization finished...')

    data_ids = get_data_ids(label_dir)
    print('data ids:', data_ids)

    num_total_cases, num_correct_cases = 0, 0
    local_record_path = 'record.txt'

    for data_id in sorted(data_ids):
        try:
            print('current evaluate data id = {0}...'.format(data_id))
            labels, start_times, end_times = parse_label_file(label_dir, data_id)
            sample_cases_indexes = random.sample(range(0, len(labels)), max(2, len(labels) // 10))

            for i, case_index in enumerate(sample_cases_indexes):
                try:
                    print(' - {0}/{1} cases predicting:'.format(i + 1, len(sample_cases_indexes)))
                    current_label = labels[case_index] - 1
                    current_start_time = start_times[case_index]
                    current_end_time = end_times[case_index]
                    multi_modal_inputs = load_data(data_dir, data_id, current_start_time, current_end_time)
                    prediction = predict(models, multi_modal_inputs)
                    winner = execute_vote(prediction)
                    if winner == current_label:
                        num_correct_cases += 1
                    num_total_cases += 1
                    print('prediction:', prediction, 'winner:', winner, 'label:', current_label)
                    print('#total cases:', num_total_cases, '#correct cases:', num_correct_cases, 'accuracy:', 1.0 * num_correct_cases / num_total_cases)
                except:
                    print('data_id = {0}, order = {1}:'.format(data_id, case_index), traceback.format_exc())
        except:
            print('data_id = {0}:'.format(data_id), traceback.format_exc())
