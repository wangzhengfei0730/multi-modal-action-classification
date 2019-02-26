import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tensorboardX import SummaryWriter

from model import HCN


writer = SummaryWriter()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='PKUMMDv2/Data/skeleton_processed', help='dataset directory')
parser.add_argument('--gpu', default=False, action='store_true', help='whether to use gpus for training')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers for multiprocessing')
parser.add_argument('--checkpoint-path', type=str, default='checkpoint.pt', help='checkpoint file path')
parser.add_argument('--seed', type=int, default=429, help='random seed')


def pickle_loader(path):
    with open(path, 'rb') as f:
        return np.array(pickle.load(f), dtype=np.float32)


def load_data(dataset_dir, batch_size, num_workers):
    pkummd_dataset = datasets.DatasetFolder(root=args.dataset_dir, loader=pickle_loader, extensions=['pkl'])
    dataset_size = len(pkummd_dataset)
    dataloader = torch.utils.data.DataLoader(pkummd_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader, dataset_size


def train(model, dataloader, num_epochs, dataset_size, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
        
        epoch_loss = running_loss / dataset_size
        epoch_accuracy = running_corrects.double() / dataset_size

        print('  Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_accuracy))
        writer.add_scalar('loss', epoch_loss)
        writer.add_scalar('accuracy', epoch_accuracy)
    
    return model


def evaluate():
    pass


def main():
    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    dataloader, dataset_size = load_data(args.dataset_dir, args.batch_size, args.num_workers)
    model = HCN()
    if args.gpu and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)
    model = train(model, dataloader, args.num_epochs, dataset_size, device)
    torch.save(model.module.state_dict(), args.checkpoint_path)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main()

