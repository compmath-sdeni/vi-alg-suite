import numpy as np

from matplotlib import pyplot as plt

# plot x squared from -3 to 3
x = np.linspace(-3, 3, 100)
y = x ** 2
plt.plot(x, y)

# show the plot
plt.show()