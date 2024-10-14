# main function to functions and test the models
# local functions
from classes import cnn
from functions import train_nn, train_data_split, plot_loss
# external functions
import torch
import json

# models name
model_name = 'model-2'

# main function
if __name__ == '__main__':
    # initiate the models
    model = cnn.CNN()

    # route to GPU if available
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print('GPU available')
        device = torch.device("cuda")
    else:
        print('GPU not available')
        device = torch.device("cpu")
    model.to(device)

    # split the train data to test and validation
    path = '/Users/dillonwilliams/pycharmprojects/Cancer Detection/data/train'
    train_loader, val_loader = train_data_split.data_split(path, batch_size=100)

    # train the model
    epoch_loss_history_train, epoch_loss_history_val = train_nn.train(train_loader, val_loader, model)

    # save the models
    torch.save(model.state_dict(),
               f'/Users/dillonwilliams/pycharmprojects/Cancer Detection/results/models/{model_name}.pth')

    print('Model trained and saved')

    # save the loss history as json
    with open(f'/Users/dillonwilliams/pycharmprojects/Cancer Detection/results/statistics/'
              f'loss_history_{model_name}_train.json',
              'w') as file:
        file.write(json.dumps(epoch_loss_history_train))

    # save the loss history as json
    with open(f'/Users/dillonwilliams/pycharmprojects/Cancer Detection/results/statistics/'
              f'loss_history_{model_name}_val.json',
              'w') as file:
        file.write(json.dumps(epoch_loss_history_val))

    # plot the loss history
    plot_loss.plot_loss(epoch_loss_history_train, epoch_loss_history_val)
