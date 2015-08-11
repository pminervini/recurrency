# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import recurrency.layers.layer as layer
import logging


class Softmax(layer.Layer):

    def __init__(self):
        self.params = []

    def __call__(self, x):
        return T.nnet.softmax(x)


class PReLU(layer.Layer):
    '''
        Parametric Rectified Linear Unit (PReLU) [1]
        
        [1] He, K. et al. - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification - http://arxiv.org/abs/1502.01852
    '''
    def __init__(self, input_shape):
        self.alphas = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX), name='alphas')
        self.params = [self.alphas]

    def __call__(self, x):
        return T.nnet.softmax(x)
