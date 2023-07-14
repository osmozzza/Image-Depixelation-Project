"""
Author: Angelika Vižintin

################################################################################

Utils file of Image Depixelation Project.
"""
from matplotlib import pyplot as plt

def plot_loss(train_losses: list, eval_losses: list, eval_epochs: list):
    """ Function for plotting training and evaluation loss.

    :param train_losses: list of training losses
    :param eval_losses: list of evaluation losses
    :param eval_epochs: list of epochs at which evaluation was performed
    """
    x = range(len(train_losses))

    plt.plot(x, train_losses, label='train loss')
    plt.plot(eval_epochs, eval_losses, label='eval loss')

    plt.xlabel('Epoch')
    plt.xticks(eval_epochs)
    plt.ylabel('MSE')
    plt.legend()

    plt.show()





