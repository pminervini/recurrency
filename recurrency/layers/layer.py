#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T

import recurrency.utils.utils as utils

import logging


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.params = []

    def initialize(self, rng, size, tag='', type='glorot'):
        if type in ['glorot', 'glorotuniform']:
            '''
                Glorot, X. et al. - Understanding the difficulty of training deep feedforward neural networks - AISTATS 2010
            '''
            bound = np.sqrt(6. / sum(size))
            V = utils.sharedX(rng.uniform(low=-bound, high=bound, size=size), name=tag)
        elif type in ['zero', 'zeros']:
            V = utils.shared_zeros(size, name=tag)
        else:
            raise ValueError('Unknown initialization: %s' % (type))
        return V

    @abstractmethod
    def __call__(self, x):
        pass
