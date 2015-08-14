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

    def initialize(self, rng, size, tag):
        bound = 6. / np.sqrt(sum(size))
        V = utils.sharedX(rng.uniform(low=-bound, high=bound, size=size), name=tag)
        return V

    # B x D
    @abstractmethod
    def __call__(self, x):
        pass
