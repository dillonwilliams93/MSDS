# main function to functions and test the models
# local functions
from classes import cnn
from functions import train_nn
import torch
# external functions
import matplotlib.pyplot as plt
import json

# models name
model_name = 'classes-1'

# main function
if __name__ == '__main__':
    # initiate the models
    model = cnn.CNN()

    # route to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # train the models
    path = '/Users/dillonwilliams/pycharmprojects/MSDS/Deep Learning/Cancer Detection/data/train'
    loss_history = train_nn.train(path, model)

    # save the models
    torch.save(model.state_dict(),
               f'/Users/dillonwilliams/pycharmprojects/MSDS/Deep Learning'
               f'/Cancer Detection/results/models/{model_name}.pth')

    print('Model trained and saved')

    # visualize the loss history
    plt.plot(loss_history)
    plt.show()

    # save the loss history as json
    with open(f'/Users/dillonwilliams/pycharmprojects/MSDS/Deep Learning/Cancer Detection/results/statistics/'
              f'loss_history_{model_name}.json',
              'w') as file:
        file.write(json.dumps(loss_history))
