import numpy as np


class Linear:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None  # for tensor
        # grad params
        self.dW = None
        self.db = None

    def forward(self, x):
        # for tensor input
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x

        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)

        dx = dx.reshape(*self.original_x_shape)  # for tensor

        return dx


class Conv2d:
    pass
