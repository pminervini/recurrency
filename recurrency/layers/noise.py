# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

import recurrency.utils.utils as utils

import logging


class BinomialDropout(layer.Layer):

    __metaclass__ = ABCMeta


    def __init__(self, p=.0):
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams()
        self.p = p
        self.params = []

    def __call__(self, x):
        if self.p is not None and self.p > 0:
            retain_prob = 1 - self.p
            nx *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            nx /= retain_prob
        return nx
