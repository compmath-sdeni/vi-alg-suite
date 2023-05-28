import numpy as np
from matplotlib import pyplot as plt

# import pytorch
import torch


# test matplotlib
def matplotlib_checks():
    # plot x squared from -3 to 3
    x = np.linspace(-3, 3, 100)
    y = x ** 2
    plt.plot(x, y)

    # show the plot
    plt.show()


# test pytorch
def pytorch_checks():
    # create a tensor
    x = torch.tensor([1, 2, 3, 4])
    print(x)


if __name__ == "__main__":
    matplotlib_checks()
    pytorch_checks()