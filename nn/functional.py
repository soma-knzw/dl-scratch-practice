import numpy as np

from .simple_func import softmax
from .loss import CrossEntropyLoss


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dout


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.out = y
        return y

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx


class SoftmaxWithCELoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss = CrossEntropyLoss(self.y, t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
