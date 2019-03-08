import os
import random
import shutil


NUM_ACTIONS = 51
VALIDATION_RATE = 0.1
TEST_RATE = 0.1
dataset_dir = '../PKUMMDv2/Data/skeleton_processed/'

for tag in ['train', 'val', 'test']:
    # train/val/test directory
    tag_dir = os.path.join(dataset_dir, tag)
    if not os.path.exists(tag_dir):
        os.mkdir(tag_dir)
    # class directory in train/val/test
    for i in range(1, NUM_ACTIONS + 1):
        class_dir = os.path.join(tag_dir, '{:02}'.format(i))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

for i in range(1, NUM_ACTIONS + 1):
    # empty class
    if not os.path.exists(os.path.join(dataset_dir, '{:02}/'.format(i))):
        continue
    class_dir = os.path.join(dataset_dir, '{:02}/'.format(i))
    data_pkls = []
    for data_pkl in os.listdir(class_dir):
        if '.pkl' not in data_pkl:
            continue
        else:
            data_pkls.append(data_pkl)
    print('number of data in class {:02}: {}'.format(i, len(data_pkls)))

    val_size = int(len(data_pkls) * VALIDATION_RATE)
    test_size = int(len(data_pkls) * TEST_RATE)
    train_size = len(data_pkls) - val_size - test_size
    print('train/val/test dataset size:', train_size, val_size, test_size)

    random.shuffle(data_pkls)
    
    for j, data_pkl in enumerate(data_pkls):
        src_path = os.path.join(class_dir, data_pkl)
        if j < train_size:
            dst_path = os.path.join(os.path.join(os.path.join(dataset_dir, 'train'), '{:02}'.format(i)), data_pkl)
        elif j < train_size + val_size:
            dst_path = os.path.join(os.path.join(os.path.join(dataset_dir, 'val'), '{:02}'.format(i)), data_pkl)
        else:
            dst_path = os.path.join(os.path.join(os.path.join(dataset_dir, 'test'), '{:02}'.format(i)), data_pkl)

        shutil.move(src_path, dst_path)
