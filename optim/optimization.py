#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

import collections
import update as U

def minimize(inputs, y, params, optimizer='sgd', is_softmax=False, rate=1.0, decay=0.99, epsilon=1e-6):

    t, lr = T.vector(), T.scalar()
    loss = T.abs_(t - y).mean(axis = 0).sum()

    if is_softmax:
        t = T.ivector()
        loss = T.abs_(1 - y[t]).mean(axis = 0).sum()

    updates = collections.OrderedDict()

    if optimizer == 'sgd':
        pass # do nothing
    elif optimizer == 'momentum':
        param_previous_update_map = U.momentum_params(params)
    elif optimizer == 'adagrad':
        param_squared_gradients_map = U.adagrad_params(params)
    elif optimizer == 'adadelta':
        param_squared_gradients_map, param_squared_updates_map = U.adadelta_params(params)
    elif optimizer == 'rmsprop':
        param_squared_gradients_map = U.rmsprop_params(params)
    else:
        raise ValueError('Unknown optimizer: %s' % (optimizer))

    loss_gradients = [T.grad(loss, param) for param in params]

    for param, gradient in zip(params, loss_gradients):
        if optimizer == 'sgd': # SGD
            U.sgd(param, rate, gradient, updates)
        elif optimizer == 'momentum': # SGD+MOMENTUM
            param_previous_update = param_previous_update_map[param]
            U.momentum(param, rate, decay, gradient, updates, param_previous_update)
        elif optimizer == 'adagrad': # ADAGRAD
            param_squared_gradients = param_squared_gradients_map[param]
            U.adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients)
        elif optimizer == 'adadelta': # ADADELTA
            param_squared_gradients = param_squared_gradients_map[param]
            param_squared_updates = param_squared_updates_map[param]
            U.adadelta(param, rate, decay, epsilon, gradient, updates, param_squared_gradients, param_squared_updates)
        elif optimizer == 'rmsprop': # RMSPROP
            param_squared_gradients = param_squared_gradients_map[param]
            U.rmsprop(param, rate, decay, max_learning_rate, epsilon, gradient, updates, param_squared_gradients)
        else:
            raise ValueError('Unknown optimizer: %s' % (optimizer))

    # Note: indeed T.mean(cost) and cost are equivalent.
    return theano.function([inputs, t, lr], [loss], updates=updates, on_unused_input='ignore')
