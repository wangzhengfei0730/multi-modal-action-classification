import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter

from model import SpatialStreamConvNet, TemporalStreamConvNet

TYPE_STREAMS = ['rgb', 'optical_flow']
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
            stream: datasets.ImageFolder(root=dataset_dir + 'test/{0}'.format(stream), transform=transform)
            for stream in TYPE_STREAMS
        }
        test_dataset_loader = {
            stream: DataLoader(test_dataset[stream], batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for stream in TYPE_STREAMS
        }
        return test_dataset_loader, len(test_dataset)
    else:
        train_val_dataset = {
            '{0}/{1}'.format(tag, stream): datasets.ImageFolder(root=dataset_dir + '{0}/{1}'.format(tag, stream), transform=transform)
            for tag in ['train', 'val'] for stream in TYPE_STREAMS
        }
        train_val_dataset_size = {
            '{0}/{1}'.format(tag, stream): len(train_val_dataset['{0}/{1}'.format(tag, stream)])
            for tag in ['train', 'val'] for stream in TYPE_STREAMS
        }
        train_val_dataset_loader = {
            '{0}/{1}'.format(tag, stream): DataLoader(
                train_val_dataset['{0}/{1}'.format(tag, stream)],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
            for tag in ['train', 'val'] for stream in TYPE_STREAMS
        }
        return train_val_dataset_loader, train_val_dataset_size


def save_model(model, tag, stream):
    if tag is 'top':
        torch.save(model.module.state_dict(), 'top-{0}-checkpoint.pt'.format(stream))
    elif tag is 'final':
        torch.save(model.module.state_dict(), 'final-{0}-checkpoint.pt'.format(stream))
    else:
        raise NotImplementedError


def train(spatial_network, temporal_network, dataloader, num_epochs, dataset_size, device):
    top_accuracy = { stream: 0.0 for stream in TYPE_STREAMS }
    model = { 'rgb': spatial_network, 'optical_flow': temporal_network }
    optimizer = { phase: optim.Adam(model[phase].parameters(), lr=args.learning_rate) for phase in TYPE_STREAMS }
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        epoch_loss = {'{0}/{1}'.format(phase, stream): 0.0 for phase in ['train', 'val'] for stream in TYPE_STREAMS}
        epoch_accuracy = {'{0}/{1}'.format(phase, stream): 0.0 for phase in ['train', 'val'] for stream in TYPE_STREAMS}

        for phase in ['train', 'val']:
            if phase is 'train':
                for stream in TYPE_STREAMS:
                    model[stream].train()
            else:
                for stream in TYPE_STREAMS:
                    model[stream].eval()
            
            for stream in TYPE_STREAMS:
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloader['{0}/{1}'.format(phase, stream)]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer[stream].zero_grad()
                    with torch.set_grad_enabled(phase is 'train'):
                        outputs = model[stream](inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                        if phase is 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss['{0}/{1}'.format(phase, stream)] = running_loss / dataset_size['{0}/{1}'.format(phase, stream)]
                epoch_accuracy['{0}/{1}'.format(phase, stream)] = running_corrects.double() / dataset_size['{0}/{1}'.format(phase, stream)]
                print('  {0}/{1} Loss: {2:.4f} Acc: {3:.4f}'.format(
                    phase, stream,
                    epoch_loss['{0}/{1}'.format(phase, stream)],
                    epoch_accuracy['{0}/{1}'.format(phase, stream)]
                ))

                if epoch_accuracy['{0}/{1}'.format(phase, stream)] > top_accuracy[stream]:
                    print('best {0} model ever! save at global step {1}'.format(stream, epoch))
                    save_model(model[stream], tag='top', stream=stream)
                    top_accuracy = epoch_accuracy['{0}/{1}'.format(phase, stream)]

        writer.add_scalars('loss', {
            '{0}/{1}'.format(phase, stream): epoch_loss['{0}/{1}'.format(phase, stream)]
            for phase in ['train', 'val'] for stream in TYPE_STREAMS 
        }, epoch)
        writer.add_scalars('accuracy', {
            '{0}/{1}'.format(phase, stream): epoch_accuracy['{0}/{1}'.format(phase, stream)]
            for phase in ['train', 'val'] for stream in TYPE_STREAMS 
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
    dataset_dir = os.path.join(args.dataset_dir, 'Data/skeleton_processed')
    dataloader, dataset_size = load_data(dataset_dir, args.batch_size, args.num_workers)
    spatial_network, temporal_network = SpatialStreamConvNet(), TemporalStreamConvNet()
    if args.gpu and torch.cuda.is_available():
        spatial_network = torch.nn.DataParallel(spatial_network)
        temporal_network = torch.nn.DataParallel(temporal_network)
    spatial_network.to(device)
    temporal_network.to(device)
    model = train(spatial_network, temporal_network, dataloader, args.num_epochs, dataset_size, device)
    for stream in TYPE_STREAMS:
        save_model(model[stream], tag='final', stream=stream)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main()
