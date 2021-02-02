import numpy as np


def MSELoss(y, t):
    return 0.5 * np.sum((y - t)**2)


def CrossEntropyLoss(y, t=1):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
