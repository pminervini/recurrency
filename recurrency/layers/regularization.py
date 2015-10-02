#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as T


class Regularizer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

class L1Regularizer(Regularizer):

    def __init__(self):
        super(L1Regularizer, self).__init__()

    # Note: r(W) = ||W||_1
    def __call__(self, x):
        return T.sum(T.abs_(x))

class L2Regularizer(Regularizer):

    def __init__(self):
        super(L2Regularizer, self).__init__()

    # Note: r(W) = ||W||_2^2
    def __call__(self, x):
        return T.sum(x ** 2)
