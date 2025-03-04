import torch.nn as nn
import torch


# Define the CNN architecture
# Basic CNN with 3 convolutional layers and 2 fully connected layers
class CNN(nn.Module):
    def __init__(self, num_classes, num_epochs):
        super(CNN, self).__init__()
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

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 10)

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.epochs = 10

    def forward(self, x):
        # Compute convolutions
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # Flatten the output for the fully connected layers
        out = out.view(out.size(0), -1)

        # Compute fully connected layers
        out = self.fc1(out)
        out = self.fc2(out)
        return out

