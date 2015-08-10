# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

import collections
import recurrency.optimization.schedule as schedule


def minimize(inputs, y, params, optimizer='sgd', is_softmax=False, rate=1.0, decay=0.99, max_learning_rate=1e4, epsilon=1e-6):

    t, lr = T.vector(), T.scalar()
    loss = T.abs_(t - y).mean(axis = 0).sum()

    if is_softmax:
        t = T.ivector()
        loss = T.abs_(1 - y[t]).mean(axis = 0).sum()

    updates = collections.OrderedDict()

    scheduler = None

    if optimizer == 'sgd':
        scheduler = schedule.SGD(params, rate=rate)
    elif optimizer == 'momentum':
        scheduler = schedule.Momentum(params, rate=rate, decay=decay)
    elif optimizer == 'adagrad':
        scheduler = schedule.AdaGrad(params, rate=rate, epsilon=epsilon)
    elif optimizer == 'adadelta':
        scheduler = schedule.AdaDelta(params, rate=rate, decay=decay, epsilon=epsilon)
    elif optimizer == 'rmsprop':
        scheduler = schedule.RMSProp(params, rate=rate, decay=decay, max_learning_rate=max_learning_rate, epsilon=epsilon)
    else:
        raise ValueError('Unknown optimizer: %s' % (optimizer))

    loss_gradients = [T.grad(loss, param) for param in params]

    for param, gradient in zip(params, loss_gradients):
        scheduler.update(param, gradient, updates)

    # Note: indeed T.mean(cost) and cost are equivalent.
    return theano.function([inputs, t, lr], [loss], updates=updates, on_unused_input='ignore')
