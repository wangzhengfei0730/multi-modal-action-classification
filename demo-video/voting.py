import os
import random
import torch
from collections import Counter
from PIL import Image
from skeleton_model import HCN
from torchvision.transforms import transforms
from torchvision.models.resnet import resnet18, resnet101
from data_load_utils import parse_label, extract_data


NUM_CLASSES = 51

# modal, models architecture and checkpoints
modals = ['skeleton', 'rgb', 'optical_flow', 'depth', 'infrared', 'infrared_depth']
models_checkpoints = {
    'skeleton': (HCN(), 'checkpoints/skeleton.pt'),
    'rgb': (resnet101(num_classes=NUM_CLASSES), 'checkpoints/rgb.pt'),
    'optical_flow': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/optical_flow.pt'),
    'depth': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/depth.pt'),
    'infrared': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/infrared.pt'),
    'infrared_depth': (resnet18(num_classes=NUM_CLASSES), 'checkpoints/infrared_depth.pt')
}

data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])


def model_load_checkpoint():
    loaded_models = {}
    for modal in modals:
        model, checkpoint_path = models_checkpoints[modal]
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        loaded_models[modal] = model
    return loaded_models


def predict(models, inputs):
    choices = []
    for modal in modals:
        print('predicting modal - {0}...'.format(modal))
        model, x = models[modal], inputs[modal]
        if modal == 'skeleton':
            x = torch.tensor(x).float()
            x = x.unsqueeze(0)
        else:
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
            x = data_transforms(x).float()
            x = x.unsqueeze(0)
        output = model(x)
        _, prediction = torch.max(output, 1)
        print(' > modal - {0}: {1}'.format(modal, prediction))
        choices.append(prediction.item())
    # count votes and random select if more than one candidate
    counter = Counter(choices)
    print('vote result:', counter)
    most_vote = max(counter.values())
    options = []
    for k, v in counter.items():
        if v == most_vote:
            options.append(k)
    return random.choice(options)


if __name__ == '__main__':
    models = model_load_checkpoint()
    print('model initialization finished and parameters loaded...')

    data_dir = './data_sample'
    label_file_path = os.path.join(data_dir, 'label.txt')
    num_actions, action_classes, start_frames, end_frames = parse_label(label_file_path)

    random_index = random.randint(1, num_actions) - 1
    cur_action_class = action_classes[random_index]
    s_timestamp, e_timestamp = start_frames[random_index], end_frames[random_index]
    multi_modal_inputs = extract_data(data_dir, s_timestamp, e_timestamp)
    print('ground truth label:', cur_action_class - 1)
    prediction = predict(models, multi_modal_inputs)
    print('voting result:', prediction)
