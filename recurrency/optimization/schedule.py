# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy

import theano
import theano.tensor as T

import collections
import logging


# Note: RMS[g]t = sqrt(E[g^2]t + epsilon)
def RMS(value, epsilon=1e-6):
    return T.sqrt(value + epsilon)


class Schedule(object):
    '''
        Abstract class for adaptive per-parameter learning rate schedules.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, param, gradient, updates):
        pass


class SGD(Schedule):
    '''
        Plain Stochastic Gradient Descent. [1]

        [1] https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    '''
    def __init__(self, params, rate=1.0, decay=0.95):
        pass

    def update(self, param, gradient, updates):
        # Update: x_t - eta * g_t
        delta_x_t = - rate * gradient
        updates[param] = param + delta_x_t


class Momentum(Schedule):
    '''
        The Momentum method [1]

        [1] Rumelhart, D. E. et al. - Learning representations by back-propagating errors. - Nature 323
    '''
    def __init__(self, params, rate=1.0, decay=0.95):
        self.param_previous_update_map = collections.OrderedDict()
        self.rate, self.decay = rate, decay

        for param in params:
            # Allocate the previous updates
            previous_update_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_previous_update = theano.shared(value=previous_update_value, name='su_' + param.name)
            self.param_previous_update_map[param] = param_previous_update


    def update(self, param, gradient, updates):
        param_previous_update = self.param_previous_update_map[param]

        # decay represents the momentum
        delta_x_t = (self.decay * param_previous_update) - (self.rate * gradient)

        param_previous_update_updated = delta_x_t
        updates[param_previous_update] = param_previous_update_updated

        updates[param] = param + delta_x_t


class AdaGrad(Schedule):
    '''
        AdaGrad [1]

        [1] Duchi, J. et al. - Adaptive subgradient methods for online learning and stochastic optimization. - JMLR 12
    '''
    def __init__(self, params, rate=1.0, epsilon=1e-6):
        self.param_squared_gradients_map = collections.OrderedDict()
        self.rate, self.epsilon = rate, epsilon

        for param in params:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            self.param_squared_gradients_map[param] = param_squared_gradients

    def update(self, param, gradient, updates):
        param_squared_gradients = self.param_squared_gradients_map[param]

        # ssg = \sum_t=1^T  [grad_t]^2
        param_squared_gradients_updated = param_squared_gradients + (gradient ** 2)
        updates[param_squared_gradients] = param_squared_gradients_updated

        # Update: x_t - (eta / sqrt(ssg)) * g_t
        delta_x_t = - (self.rate / RMS(param_squared_gradients_updated, epsilon=self.epsilon)) * gradient
        updates[param] = param + delta_x_t


class AdaDelta(Schedule):
    '''
        Zeiler, M. D. - ADADELTA: An adaptive learning rate method. - arXiv:1212.5701
    '''
    def __init__(self, params, rate=1.0, decay=0.95, epsilon=1e-6):
        self.param_squared_gradients_map = collections.OrderedDict()
        self.param_squared_updates_map = collections.OrderedDict()
        self.rate, self.decay, self.epsilon = rate, decay, epsilon

        for param in params:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            self.param_squared_gradients_map[param] = param_squared_gradients

            # Allocate the sums of squared updates
            squared_updates_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_updates = theano.shared(value=squared_updates_value, name='su_' + param.name)

            self.param_squared_updates_map[param] = param_squared_updates

    def update(self, param, gradient, updates):
        param_squared_gradients = self.param_squared_gradients_map[param]
        param_squared_updates = self.param_squared_updates_map[param]

        # Accumulate Gradient:
        # E[g^2]t = rho * E[g^2]t-1 + (1 - rho) * g^2_t
        param_squared_gradients_updated = (self.decay * param_squared_gradients) + ((1.0 - self.decay) * (gradient ** 2)) # Eg2_t = rho Eg2_t-1 + (1-rho) g2_t
        updates[param_squared_gradients] = param_squared_gradients_updated # E[g^2]t

        # Compute Update (Hessian approximation):
        #   [delta_x]t = - (RMS[delta_x]t-1 / RMS[g]t) g_t
        # Learning rate specified as in:
        #   http://climin.readthedocs.org/en/latest/adadelta.html
        delta_x_t = - self.rate * (RMS(param_squared_updates, epsilon=self.epsilon) / RMS(param_squared_gradients_updated, epsilon=self.epsilon)) * gradient

        # Accumulate updates:
        # E[delta_x^2]t = rho * E[delta_x^2]t-1 + (1 - rho) * [delta_x^2]t
        param_squared_updates_updated = (self.decay * param_squared_updates) + ((1.0 - self.decay) * (delta_x_t ** 2))

        updates[param_squared_updates] = param_squared_updates_updated
        # Apply update:
        # x_t+1 = x_t + [delta_x]t,
        #   as in x_t+1 = x_t + [delta_x]t, with [delta_x]t = - eta g_t
        updates[param] = param + delta_x_t


class RMSProp(Schedule):
    '''
        RMSProp [1]

        [1] Tieleman, T. et al. - Lecture 6.5: rmsprop - COURSERA: Neural Networks for Machine Learning
    '''
    def __init__(self, params, rate=1.0, decay=0.95, max_learning_rate=1e4, epsilon=1e-6):
        self.param_squared_gradients_map = collections.OrderedDict()
        self.rate, self.decay, self.max_learning_rate, self.epsilon = rate, decay, max_learning_rate, epsilon

        for param in params:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            self.param_squared_gradients_map[param] = param_squared_gradients

    def update(self, param, gradient, updates):
        param_squared_gradients = self.param_squared_gradients_map[param]

        # Accumulate Gradient:
        # E[g^2]t = rho * E[g^2]t-1 + (1 - rho) * g^2_t
        param_squared_gradients_updated = (self.decay * param_squared_gradients) + ((1.0 - self.decay) * (gradient ** 2)) # Eg2_t = rho Eg2_t-1 + (1-rho) g2_t
        updates[param_squared_gradients] = param_squared_gradients_updated # E[g^2]t

        # Compute Update:
        # [delta_x]t = - (eta / E[g^2]t) g_
        delta_x_t = - (self.rate / RMS(param_squared_gradients_updated, epsilon=self.epsilon)) * gradient

        # maxLearningRate approx. as in https://github.com/w-cheng/optimx/blob/master/rmsprop.lua
        if (self.max_learning_rate is not None):
            max_rates = numpy.full(param.get_value().shape, self.max_learning_rate, dtype=theano.config.floatX)

            delta_x_t = T.minimum(delta_x_t, max_rates)
            # min_learning_rate mirrors max_learning_rate
            delta_x_t = T.maximum(delta_x_t, - max_rates)

        updates[param] = param + delta_x_t
