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
