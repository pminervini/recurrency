# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import recurrency.utils as utils
import recurrency.layers.layer as layer

import logging

class SoftMax(layer.Layer):
    def __init__(self):
        super(SoftMax, self).__init__()
    def __call__(self, x):
        return T.nnet.softmax(x)

class Sigmoid(layer.Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def __call__(self, x):
        return T.nnet.sigmoid(x)

class Tanh(layer.Layer):
    def __init__(self):
        super(Tanh, self).__init__()
    def __call__(self, x):
        return T.tanh(x)

class Linear(layer.Layer):
    def __init__(self):
        super(Linear, self).__init__()
    def __call__(self, x):
        return x

class ReLU(layer.Layer):
    '''
        Rectified Linear Units (ReLU) [1]

        [1] Glorot, X. et al. - Deep sparse rectifier neural networks - AISTATS 2011
    '''
    def __init__(self):
        super(ReLU, self).__init__()
    def __call__(self, x):
        return x * (x > 0)

class PReLU(layer.Layer):
    '''
        Parametric Rectified Linear Unit (PReLU) [1]

        [1] He, K. et al. - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification - http://arxiv.org/abs/1502.01852
    '''
    def __init__(self, input_shape):
        super(PReLU, self).__init__()
        self.alphas = utils.shared_zeros(input_shape, name='alphas')
        self.params += [self.alphas]

    def __call__(self, x):
        pos = ((x + abs(x)) / 2.0)
        neg = self.alphas * ((x - abs(x)) / 2.0)
        return pos + neg


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    x, act = T.vector('x'), SoftMax() #PReLU(11)
    f = theano.function([x], act(x))

    print(f([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
