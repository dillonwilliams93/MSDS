# define the CNN architecture
import torch.nn as nn
import torch


class CNNInception(nn.Module):
    def __init__(self):
        super(CNNInception, self).__init__()

        # Convolutional Layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=2),
            nn.Relu(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, padding=2),
            nn.Relu()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Relu(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        # Inception Layers
        self.inception1 = nn.Conv2d(64, 128, kernel_size=1)
        self.inception2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.Conv2d(128, 192, kernel_size=3, padding=1)
        )
        self.inception3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=5, padding=2)
        )
        self.inception4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=1)
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 10)

        # Softmax for output
        self.softmax = nn.Softmax(dim=1)

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.epochs = 10

    def forward(self, x):
        # Compute convolutions
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # Compute Inception layers
        out1 = self.inception1(out)
        out2 = self.inception2(out)
        out3 = self.inception3(out)
        out4 = self.inception4(out)

        # Concat the output of the inception layers
        out = torch.cat((out1, out2, out3, out4), 1)

        # Flatten the output for the fully connected layers
        out = out.view(out.size(0), -1)

        # Compute fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)

        # Apply softmax
        out = self.softmax(out)
        return out
