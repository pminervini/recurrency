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

    def initialize(self, rng, size, tag='', type='glorot2010'):
        if type in ['glorot2010', 'glorotuniform']:
            '''
                Glorot, X. et al. - Understanding the difficulty of training deep feedforward neural networks - AISTATS 2010
            '''
            bound = np.sqrt(6. / sum(size))
            value = rng.uniform(low=-bound, high=bound, size=size)
            V = utils.sharedX(value, name=tag)
        elif type in ['identity', 'eye', 'le2015', 'leidentity']:
            '''
                Le, Q. V. et al. - A Simple Way to Initialize Recurrent Networks of Rectified Linear Units - arXiv:1504.00941
            '''
            diag = np.ones(min(size[0], size[1]))
            value = np.zeros(size)
            np.fill_diagonal(value, diag)
            V = utils.sharedX(value, name=tag)
        elif type in ['zero', 'zeros']:
            V = utils.shared_zeros(size, name=tag)
        elif type in ['one', 'ones']:
            V = utils.shared_ones(size, name=tag)
        else:
            raise ValueError('Unknown initialization: %s' % (type))
        return V

    @abstractmethod
    def __call__(self, x):
        pass
