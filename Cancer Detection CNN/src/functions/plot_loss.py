# Defines the loss plotting function
import matplotlib.pyplot as plt


# Plot the Cross Validation loss per epoch
def plot_loss(epoch_loss_history_train, epoch_loss_history_val):
    plt.plot(epoch_loss_history_train, label='Train Loss', color='black')
    plt.plot(epoch_loss_history_val, label='Validation Loss', color='yellow')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Classification Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.show()
