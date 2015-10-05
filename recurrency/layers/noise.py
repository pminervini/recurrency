# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as R

import recurrency.layers.layer as layer
import recurrency.utils as utils

import logging

class NoiseLayer(layer.Layer):
    '''
        Abstract class for Noise layers.
    '''
    def __init__(self):
        super(NoiseLayer, self).__init__()
        self.params += []

    def __call__(self, X):
        return X

class Null(NoiseLayer):

    def __init__(self):
        super(Null, self).__init__()
        self.params += []

    def __call__(self, X):
        return X

class BinomialDropout(NoiseLayer):

    def __init__(self, p=.0):
        super(BinomialDropout, self).__init__()
        self.srng = R.MRG_RandomStreams()
        self.p = p
        self.params += []

    def __call__(self, X, inference=False):
        if self.p is not None and self.p > 0.:
            retain_prob = 1 - self.p
            if inference is True:
                X *= retain_prob
            else:
                M = self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
                X *= M
        return X

if __name__ == '__main__':
    x = T.matrix('x')
    d = BinomialDropout(p=.2)
    f = theano.function([x], [d(x)])
    print(f([[1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9]]))
