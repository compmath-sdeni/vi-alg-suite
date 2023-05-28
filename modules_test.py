import numpy as np
from matplotlib import pyplot as plt

# import pytorch
import torch

# import jax
import jax.numpy as jnp
from jax import grad, jit, vmap


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
    x = torch.tensor([3], dtype=torch.float, requires_grad=True)
    # how to use retain_graph=True? Give code example
    # https://stackoverflow.com/questions/48001598/why-do-we-need-to-set-the-retain-graph-to-true-in-pytorch



    # create expression using the tensor with non-linear function and torch sine, and tell torch to track the gradient
    y = x ** 2 + torch.sin(x) + 2*x + 3 # derivative is 2x + cos(x) + 2, and at x=3 it is 2*3 + cos(3) + 2 =
    y.backward(retain_graph=True) # compute the gradient

    # print the result
    print(f"Derivative at {x.item()} is {x.grad.item()}")

    # print gradient of y with respect to x
    print(torch.autograd.grad(y, 5))

# test jax
def jax_checks():
    # create function - logistic curve (sigmoid)
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))

    # plot the function and its derivative
    x = jnp.linspace(-10, 10, 100)
    plt.plot(x, sigmoid(x), label="sigmoid")
    df = vmap(grad(sigmoid))
    plt.plot(x, df(x), label="derivative")

    # the same but using a loop
    # y = []
    # for xi in x:
    #     y.append(grad(sigmoid)(xi))
    # plt.plot(x, y, label="derivative")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # matplotlib_checks()
    # pytorch_checks()

    jax_checks()