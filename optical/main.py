import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.resnet import resnet50
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter


writer = SummaryWriter()
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='PKUMMDv1', help='dataset directory')
parser.add_argument('--gpu', default=False, action='store_true', help='whether to use gpus for training')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--num-workers', type=int, default=4, help='number of workers for multiprocessing')
parser.add_argument('--checkpoint-path', type=str, default='checkpoint.pt', help='checkpoint file path')
parser.add_argument('--seed', type=int, default=429, help='random seed')
parser.add_argument('--evaluation', default=False, action='store_true', help='whether to evaluate the model')


def pickle_loader(path):
    with open(path, 'rb') as f:
        return np.array(pickle.load(f), dtype=np.float32)


def load_data(dataset_dir, batch_size, num_workers):
    if args.evaluation:
        test_dataset = {
            'optical': datasets.ImageFolder(root=dataset_dir + 'optical/test', transform=transform)
        }
        test_dataset_loader = {
            'optical': DataLoader(test_dataset['optical'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        }
        return test_dataset_loader, len(test_dataset)
    else:
        train_val_dataset = {
            '{0}/optical'.format(tag): datasets.ImageFolder(root=dataset_dir + 'optical/{0}'.format(tag), transform=transform)
            for tag in ['train', 'val']
        }
        train_val_dataset_size = {
            '{0}/optical'.format(tag): len(train_val_dataset['{0}/optical'.format(tag)])
            for tag in ['train', 'val']
        }
        train_val_dataset_loader = {
            '{0}/optical'.format(tag): DataLoader(
                train_val_dataset['{0}/optical'.format(tag)],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            for tag in ['train', 'val']
        }
        return train_val_dataset_loader, train_val_dataset_size


def save_model(model, tag, stream='optical'):
    if tag is 'top':
        torch.save(model.module.state_dict(), 'top-{0}-checkpoint.pt'.format(stream))
    elif tag is 'final':
        torch.save(model.module.state_dict(), 'final-{0}-checkpoint.pt'.format(stream))
    else:
        raise NotImplementedError


def train(model, dataloader, num_epochs, dataset_size, device):
    top_accuracy = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        epoch_loss = {'train': 0.0, 'val': 0.0}
        epoch_accuracy = {'train': 0.0, 'val': 0.0}

        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            if phase is 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloader['{0}/optical'.format(phase)]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase is 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    if phase is 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss['{0}/optical'.format(phase)] = running_loss / dataset_size['{0}/optical'.format(phase)]
            epoch_accuracy['{0}/optical'.format(phase)] = running_corrects.double() / dataset_size['{0}/optical'.format(phase)]
            print('  {0}/optical Loss: {1:.4f} Acc: {2:.4f}'.format(
                phase,
                epoch_loss['{0}/optical'.format(phase)],
                epoch_accuracy['{0}/optical'.format(phase)]
            ))

            if epoch_accuracy['{0}/optical'.format(phase)] > top_accuracy:
                print('best optical model ever! save at global step {0}'.format(epoch))
                save_model(model, tag='top')
                top_accuracy = epoch_accuracy['{0}/optical'.format(phase)]

        writer.add_scalars('loss', {
            '{0}/optical'.format(phase): epoch_loss['{0}/optical'.format(phase)]
            for phase in ['train', 'val']
        }, epoch)
        writer.add_scalars('accuracy', {
            '{0}/optical'.format(phase): epoch_accuracy['{0}/optical'.format(phase)]
            for phase in ['train', 'val']
        }, epoch)
    
    return model


def evaluate(model, dataloader, dataset_size, device):
    model.eval()
    corrects = 0
    for inputs, labels in dataloader['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
    acc = corrects.double() / dataset_size['test']
    return acc.item()


def main():
    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    dataset_dir = '../' + args.dataset_dir + '/Data/RGB/'
    dataloader, dataset_size = load_data(dataset_dir, args.batch_size, args.num_workers)
    model = resnet50(num_classes=51)
    if args.gpu and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)
    model = train(model, dataloader, args.num_epochs, dataset_size, device)
    save_model(model, tag='final')
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main()
