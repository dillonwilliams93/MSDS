# define the CNN architecture
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.LazyLinear(1)

        # Loss, optimizer and epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0015)
        self.epochs = 50

    def forward(self, x):
        # Convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Flatten the tensor before the fully connected layers
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
