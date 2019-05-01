import os
import random
import shutil


NUM_ACTION_CLASSES = 51
VALIDATION_RATE = 0.1
TEST_RATE = 0.1

data_dir = '../PKUMMDv1/Data'
depth_dir = os.path.join(data_dir, 'Depth')

for tag in ['train', 'val', 'test']:
    depth_tag_dir = os.path.join(depth_dir, tag)
    if not os.path.exists(depth_tag_dir):
        os.mkdir(depth_tag_dir)
    for i in range(1, NUM_ACTION_CLASSES + 1):
        depth_class_dir = os.path.join(depth_tag_dir, '{:02}'.format(i))
        if not os.path.exists(depth_class_dir):
            os.mkdir(depth_class_dir)

print('processing depth ...')
for i in range(1, NUM_ACTION_CLASSES + 1):
    class_dir = os.path.join(depth_dir, '{:02}'.format(i))
    if not os.path.exists(class_dir):
        continue
    images = []
    for image in os.listdir(class_dir):
        if '.jpg' not in image:
            continue
        else:
            images.append(image)
    print('number of images in class {:02}: {}'.format(i, len(images)))

    val_size = int(len(images) * VALIDATION_RATE)
    test_size = int(len(images) * TEST_RATE)
    train_size = len(images) - val_size - test_size
    print('train/val/test dataset size:', train_size, val_size, test_size)

    random.shuffle(images)

    for j, image in enumerate(images):
        src_path = os.path.join(class_dir, image)
        if j < train_size:
            dst_path = os.path.join(os.path.join(os.path.join(depth_dir, 'train'), '{:02}'.format(i)), image)
        elif j < train_size + val_size:
            dst_path = os.path.join(os.path.join(os.path.join(depth_dir, 'val'), '{:02}'.format(i)), image)
        else:
            dst_path = os.path.join(os.path.join(os.path.join(depth_dir, 'test'), '{:02}'.format(i)), image)
        shutil.move(src_path, dst_path)
    
    os.rmdir(class_dir)
