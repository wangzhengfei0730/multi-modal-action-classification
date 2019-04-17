import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.models.resnet import resnet101

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='PKUMMDv1', help='dataset directory')
parser.add_argument('--gpu', default=False, action='store_true', help='whether to use gpus for training')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers for multiprocessing')
parser.add_argument('--model-path', type=str, default='top-rgb-checkpoint.pt', help='model parameter file path')


def load_data(dataset_dir, batch_size, num_workers):
    vis_dataset = datasets.ImageFolder(root=dataset_dir + 'rgb/vis', transform=transform)
    vis_dataset_size = len(vis_dataset)
    vis_dataset_loader = DataLoader(vis_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return vis_dataset_loader, vis_dataset_size


def visualize(model, dataloader, dataset_size, device):
    model.eval()
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        x = inputs
        print('input:', x.shape)
        for name, module in model._modules.items():
            if name == 'fc':
                x = x.view(x.size(0), -1)
            x = module(x)
            print(name, x.shape[1:])


def main():
    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    dataset_dir = '../' + args.dataset_dir + '/Data/RGB/'
    dataloader, dataset_size = load_data(dataset_dir, args.batch_size, args.num_workers)
    
    model = resnet101(num_classes=51)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    visualize(model, dataloader, dataset_size, device)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
