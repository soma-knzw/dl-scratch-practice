import numpy as np


class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grad):
        for k in params.keys():
            params[k] -= self.lr * grad[k]


class Momentum:
    """
    Momentum SGD
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, params, grad):
        if self.v is None:  # first time called
            self.v = {}
            for k, v in params.items():
                self.v[k] = np.zeros_like(v)

        for k in params.keys():
            self.v[k] = self.momentum * self.v[k] - self.lr * grad[k]
            params[k] += self.v[k]


class AdaGrad:
    """
    AdaGrad optimizer
    with learining rate deacy
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def step(self, params, grad):
        if self.h is None:  # first time called
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)

        for k in params.keys():
            self.h[k] += grad[k] * grad[k]
            params[k] -= self.lr * grad[k] / (np.sqrt(self.h[k] + 1e-7))


class Adam:
    """
    Adam
    combination of Momemtum and AdaGrad?
    copied without understanding
    """
    def __init__(self, lr=0.001, beta=0.9, beta2=0.999):
        self.lr = lr
        self.beta = beta
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def step(self, params, grad):
        if self.m is None:
            self.m, self.v = {}, {}
            for k, v in params.items():
                self.m[k] = np.zeros_like(v)
                self.v[k] = np.zeros_like(v)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) /\
            (1.0 - self.beta**self.iter)

        for k in params.keys():
            self.m[k] += (1 - self.beta) * (grad[k] - self.m[k])
            self.v[k] += (1 - self.beta2) * (grad[k]**2 - self.v[k])

            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + 1e-7)
