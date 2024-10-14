# function to train the neural network
# local functions
from functions import transform
import torch


# functions the neural network
def train(train_loader, val_loader, model):
    # initiate transformer
    transformer = transform.transformer()

    # initiate loss function and optimizer
    criterion = model.criterion
    optimizer = model.optimizer

    # initiate functions loss history
    epoch_loss_history_train = []
    epoch_loss_history_val = []

    # functions the models
    for epoch in range(model.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # Print every 1000 mini-batches
                print(
                    f'Epoch [{epoch + 1}/{model.epochs}], Step [{i + 1}/{len(train_loader)}]'
                )

        # Calculate and store the average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_loss_history_train.append(epoch_loss)

        # Validate the model
        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            for i, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_running_loss += loss.item()
            print(f'Epoch #{epoch + 1} Validation Loss: {val_running_loss / len(val_loader)}')

        # Calculate and store the average loss for this epoch
        val_epoch_loss = val_running_loss / len(val_loader)
        epoch_loss_history_val.append(val_epoch_loss)

    return epoch_loss_history_train, epoch_loss_history_val
