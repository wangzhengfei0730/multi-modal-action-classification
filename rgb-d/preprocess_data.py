import os
import argparse

# version: 1, 2
# modality: RGB, Depth
# purpose: slice (videos), organize (directory)


parser = argparse.ArgumentParser(description='preprocess PKU-MMD data')
parser.add_argument('--version', default=1, type=int, help='version of PKU-MMD')
parser.add_argument('--modal', default='RGB', type=str, help='modality of PKU-MMD')
parser.add_argument('--purpose', default='slice', type=str, help='preprocess purpose')
args = parser.parse_args()

data_dir = '../PKUMMDv' + str(args.version) + '/Data/' + args.modal
label_dir = '../PKUMMDv' + str(args.version) + '/Label'
print('working directory:', data_dir, '...')

for label_filename in os.listdir(label_dir):
    print('label:', label_filename)
    label_file = os.path.join(label_dir, label_filename)
    print('label file path:', label_file)


def slice_video():
    pass


if __name__ == "__main__":
    if args.purpose == 'slice':
        slice_video()
    else:
        raise NotImplementedError
