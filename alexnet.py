import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes, channels):

        super().__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(channels, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.classification = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.ConvBlock1(x)
        out = self.ConvBlock2(out)
        out = self.ConvBlock3(out)
        out = self.ConvBlock4(out)
        out = self.ConvBlock5(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.classification(out)

        return out





        
