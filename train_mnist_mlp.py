import sys
import os

import numpy as np

from dataset.mnist import load_mnist
from nn.network.tri_layer_net import TriLayerNet
from nn.optimizer import Adam


def train_tri_layer_net():
    hyp_params = {'iter_n': 10000, 'batch_size': 100, 'lr': 0.01}

    # load mnist
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    net = TriLayerNet(input_size=28*28, hidden1_size=128,
                      hidden2_size=64, output_size=10)
    optimizer = Adam()

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_size = x_train.shape[0]

    iter_per_epoch = max(train_size / hyp_params['batch_size'], 1)

    # epoch loop
    for i in range(hyp_params['iter_n']):
        # mini batch
        batch_mask = np.random.choice(train_size, hyp_params['batch_size'])
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # clac grad
        grad = net.gradient(x_batch, t_batch)
        params = net.params

        # update weights
        optimizer.step(params, grad)

        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_y_pred = net.predict(x_train)
            test_y_pred = net.predict(x_test)
            train_acc = accuracy(train_y_pred, t_train)
            test_acc = accuracy(test_y_pred, t_test)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print('epoch: {} - loss: {:.5f} - acc: {:.5f} - test_acc: {:.5f}'
                  .format(int(i/iter_per_epoch),
                          np.mean(train_loss_list),
                          np.mean(train_acc_list),
                          np.mean(test_acc_list),
                          )
                  )


def accuracy(y_pred, t):
    y_pred = np.argmax(y_pred, axis=1)
    if t.ndim != 1:
        t = np.argmax(t, axis=1)

    return np.sum(y_pred == t) / float(len(t))


if __name__ == '__main__':
    train_tri_layer_net()
