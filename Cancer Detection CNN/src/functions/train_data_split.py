# function to separate train and validation data
from functions import transform
import os
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder


def data_split(path, batch_size):
    transformer = transform.transformer()
    # use ImageFolder to load dataset from folders labeled 0 and 1
    dataset = ImageFolder(root=path, transform=transformer)

    # split the data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    # create the dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
