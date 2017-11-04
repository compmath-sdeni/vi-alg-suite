import numpy as np


def bregmanProj(x: np.array, a: np.array):
    t = x * np.exp(a)
    return t / t.sum(0, keepdims=True)
