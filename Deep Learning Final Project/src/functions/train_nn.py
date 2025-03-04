# functions the neural network
# local functions
from functions import transform
# external functions
import torchvision
from torch.utils.data import DataLoader


# functions the neural network
def train(path, model):
    # initiate transformer
    transformer = transform.transformer()

    # initiate functions data set and loader
    train_data = torchvision.datasets.ImageFolder(root=path, transform=transformer)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

    # initiate loss function and optimizer
    criterion = model.criterion
    optimizer = model.optimizer

    # initiate functions loss history
    train_loss_history = []

    # functions the models
    for epoch in range(model.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            train_loss_history.append(loss.item())
            if i % 100 == 99:  # Print every 100 mini-batches
                print(
                    f'Epoch [{epoch + 1}/{model.epochs}], Step [{i + 1}/{len(train_loader)}], '
                    f'Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    return train_loss_history
