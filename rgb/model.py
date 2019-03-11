import torch.nn as nn


class SpatialStreamConvNet(nn.Module):
    def __init__(self):
        super(SpatialStreamConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),        
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.Linear(2048, 51)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TemporalStreamConvNet(nn.Module):
    def __init__(self):
        super(TemporalStreamConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),         
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.Linear(2048, 51)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
