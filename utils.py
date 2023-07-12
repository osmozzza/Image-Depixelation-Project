"""
Author: Angelika Vižintin

################################################################################

Utils file of Image Depixelation Project.
"""
from matplotlib import pyplot as plt


def plot_loss(train_losses, eval_losses, eval_epochs):
    x = range(len(train_losses))

    plt.plot(x, train_losses, label='train loss')
    plt.plot(eval_epochs, eval_losses, label='eval loss')

    plt.xlabel('Epoch')
    plt.xticks(range(1, len(train_losses) + 1))
    plt.ylabel('MSE')
    plt.legend()

    plt.show()

