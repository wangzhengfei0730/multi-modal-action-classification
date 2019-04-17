import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.models.resnet import resnet101

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='PKUMMDv1', help='dataset directory')
parser.add_argument('--gpu', default=False, action='store_true', help='whether to use gpus for training')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers for multiprocessing')
parser.add_argument('--model-path', type=str, default='top-rgb-checkpoint.pt', help='model parameter file path')
parser.add_argument('--selected-layer', type=str, default='conv1', help='layers name want to be visualized')


def load_data(dataset_dir, batch_size, num_workers):
    vis_dataset = datasets.ImageFolder(root=dataset_dir + 'rgb/vis', transform=transform)
    vis_dataset_size = len(vis_dataset)
    vis_dataset_loader = DataLoader(vis_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return vis_dataset_loader, vis_dataset_size


def visualize(model, dataloader, dataset_size, device, selected_layer):
    model.eval()
    vis_results = []
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        x = inputs
        for name, module in model._modules.items():
            x = module(x)
            if name == selected_layer:
                vis_results.append(x)
                break
    return vis_results


def main():
    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    dataset_dir = '../' + args.dataset_dir + '/Data/RGB/'
    dataloader, dataset_size = load_data(dataset_dir, args.batch_size, args.num_workers)
    
    model = resnet101(num_classes=51)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    vis_results = visualize(model, dataloader, dataset_size, device, args.selected_layer)
    
    layer_output = vis_results[0][0]
    print(layer_output.shape)
    num_features = layer_output.shape[0]
    num_row = int(num_features ** 0.5)
    for i in range(num_features):
        feature = layer_output[i].data.numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.round(feature * 255)
        ax = plt.subplot(num_row, num_row, i + 1)
        ax.axis('off')
        plt.imshow(feature, cmap=plt.cm.gray)
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig('{0}.jpg'.format(args.selected_layer))

if __name__ == '__main__':
    args = parser.parse_args()
    main()
