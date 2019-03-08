import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils as U


NUM_ACTION_CLASSES = 51

# hierarchical co-occurrence network
# reference: Hikvision. IJCAI 2018. arXiv:1804.06055
class HCN(nn.Module):
    '''
    Input shape:
    (N, D, T, J, P)
        N: batch size
        D: input channels
        T: sequence length
        J: number of joints
        P: number of persons
    '''
    def __init__(
        self, 
        in_channel=3, num_joints=25,
        sequence_length=256,
        num_persons=2, num_classes=NUM_ACTION_CLASSES
    ):
        super(HCN, self).__init__()
        self.num_persons = num_persons
        self.num_classes = num_classes

        # sequence
        self.sequence_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.sequence_conv2 = nn.Conv2d(in_channels=64, out_channels=sequence_length, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.sequence_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joints, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        self.sequence_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        # motion
        self.motion_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.motion_conv2 = nn.Conv2d(in_channels=64, out_channels=sequence_length, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.motion_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joints, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        self.motion_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        # concatenate sequence and motion
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(256 * (sequence_length // 16) * (sequence_length // 16), 256 * self.num_persons),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.fc8 = nn.Linear(256 * self.num_persons, num_classes)

        # initialize model weight
        U.initialize_model_weight(list(self.children()))
        print('model weight initialization finished!')
    
    def forward(self, x):
        N, D, T, J, P = x.size()
        motion = x[:,:,1::,:,:] - x[:,:,0:-1,:,:]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, D * P, T - 1, J)
        motion = F.upsample(motion, size=(T, J), mode='bilinear', align_corners=False).contiguous().view(N, D, P, T, J).permute(0, 1, 3, 4, 2)

        logits = []
        for i in range(self.num_persons):
            # sequence
            out = self.sequence_conv1(x[:,:,:,:,i])
            out = self.sequence_conv2(out)
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.sequence_conv3(out)
            out_sequence = self.sequence_conv4(out)
            # motion
            out = self.motion_conv1(motion[:,:,:,:,i])
            out = self.motion_conv2(out)
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.motion_conv3(out)
            out_motion = self.motion_conv4(out)
            # concatenate sequence and motion
            out = torch.cat((out_sequence, out_motion), dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)
        
        out = torch.max(logits[0], logits[1])
        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        out = self.fc8(out)

        return out


if __name__ == '__main__':
    model = HCN()
    children = list(model.children())
    print(children)
