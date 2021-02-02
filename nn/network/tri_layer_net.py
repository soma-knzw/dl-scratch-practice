import sys
import os

import numpy as np

from collections import OrderedDict

from ..layers import Linear
from ..functional import *


class TriLayerNet:
    """
    Lin -> Lin -> Lin -> softmax
    """
    def __init__(self, input_size, hidden1_size, hidden2_size,
                 output_size, weight_init_std=0.01):
        # initialize weights with random nums
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden1_size)
        self.params['b1'] = np.zeros(hidden1_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden1_size, hidden2_size)
        self.params['b2'] = np.zeros(hidden2_size)
        self.params['W3'] = weight_init_std * \
            np.random.randn(hidden2_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # creating layers
        self.layers = OrderedDict()
        self.layers['Lin1'] = Linear(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Lin2'] = Linear(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Lin3'] = Linear(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithCELoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        """
        calc backprop
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        # go backwards through each layer
        for layer in layers:
            dout = layer.backward(dout)

        # store each gradients to return
        grad = {}
        grad['W1'] = self.layers['Lin1'].dW
        grad['b1'] = self.layers['Lin1'].db
        grad['W2'] = self.layers['Lin2'].dW
        grad['b2'] = self.layers['Lin2'].db
        grad['W3'] = self.layers['Lin3'].dW
        grad['b3'] = self.layers['Lin3'].db

        return grad
