import os
import matplotlib
# Ubuntu
# matplotlib.use('Agg')
# Mac
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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


NUM_ACTIONS =  51

dataset_dir = './PKUMMDv1/'
skeleton_dir = os.path.join(dataset_dir, 'Data/SKELETON')
label_dir = os.path.join(dataset_dir, 'Label')
print('skeleton directory:', skeleton_dir)
print('label directory:', label_dir)

num_data = [0] * NUM_ACTIONS
sum_data = [0] * NUM_ACTIONS
max_sequence_length = [0] * NUM_ACTIONS
min_sequence_length = [0] * NUM_ACTIONS

for label_file in os.listdir(label_dir):
    # avoid file like .DS_Store
    if '.txt' not in label_file:
        continue
    data_id = label_file.split('.')[0]
    label_path = os.path.join(label_dir, label_file)
    num_actions, action_classes, start_times, end_times = parse_label(label_path)
    
    for i in range(num_actions):
        action_class = action_classes[i] - 1
        sequence_length = end_times[i] - start_times[i]
        num_data[action_class] += 1
        sum_data[action_class] += sequence_length
        max_sequence_length[action_class] = max(max_sequence_length[action_class], sequence_length)

no_data_classes = []
for i in range(NUM_ACTIONS):
    if num_data[i] is 0:
        no_data_classes.append(i + 1)
    else:
        sum_data[i] /= num_data[i]
average_data = sum_data
print('PKU-MMD v1 does not contain data in class:', no_data_classes)

actions = [i for i in range(1, NUM_ACTIONS + 1)]
# number of each action class
plt.figure()
plt.title('Number of each action class')
plt.bar(x=actions, height=num_data)
plt.savefig('number_action_classes.png', dpi=300)
# average number of each action class
plt.figure()
plt.title('Average number of each action class')
plt.bar(x=actions, height=sum_data)
plt.savefig('average_number.png', dpi=300)
# max sequence length of each action class
plt.figure()
plt.title('Max sequence length of each action class')
plt.bar(x=actions, height=max_sequence_length)
plt.savefig('max_sequence.png', dpi=300)

print('dataset statistic:')
for i in range(NUM_ACTIONS):
    print('action class: {0:2d}, total number = {1:3d}, average = {2:3.0f}, max = {3:3d}'.format(
        i + 1, num_data[i], average_data[i], max_sequence_length[i]
    ))
