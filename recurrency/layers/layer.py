#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

import logging


class Layer(object):

    __metaclass__ = ABCMeta
    srng = theano.sandbox.rng_mrg.MRG_RandomStreams()

    def initialize(self, rng, size, tag):
        bound = 6. / np.sqrt(sum(size))
        values = np.asarray(rng.uniform(low=-bound, high=bound, size=size), dtype=theano.config.floatX)
        V = theano.shared(value=values, name=tag)
        return V

    def dropout(self, srng, X, p=0., deterministic=False):
        if not deterministic and p is not None and p > 0:
            retain_prob = 1 - p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X

    # B x D
    @abstractmethod
    def __call__(self, x):
        pass
