# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T

import recurrency.layers.initialization as initialization
import recurrency.utils as utils

import logging

class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.params = []

    def initialize(self, rng, size, tag='', type='glorot_uniform'):
        V = initialization.initialize(rng, size, name=tag, type=type)
        return V

    @abstractmethod
    def __call__(self, x):
        pass
