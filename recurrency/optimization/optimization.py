# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

import collections
import recurrency.optimization.schedule as schedule


def get_scheduler(params, optimizer, rate=1.0, momentum=0.9, decay=0.99, max_learning_rate=1e4, epsilon=1e-6):
    scheduler = None
    if optimizer == 'sgd':
        scheduler = schedule.SGD(params, rate=rate)
    elif optimizer == 'momentum':
        scheduler = schedule.Momentum(params, rate=rate, momentum=momentum)
    elif optimizer == 'adagrad':
        scheduler = schedule.AdaGrad(params, rate=rate, epsilon=epsilon)
    elif optimizer == 'adadelta':
        scheduler = schedule.AdaDelta(params, rate=rate, decay=decay, epsilon=epsilon)
    elif optimizer == 'rmsprop':
        scheduler = schedule.RMSProp(params, rate=rate, decay=decay, max_learning_rate=max_learning_rate, epsilon=epsilon)
    else:
        raise ValueError('Unknown optimizer: %s' % (optimizer))
    return scheduler


def minimize(inputs, loss, params, optimizer='sgd', is_softmax=False, rate=1.0, momentum=0.9, decay=0.99, max_learning_rate=1e4, epsilon=1e-6):
    updates = collections.OrderedDict()
    scheduler = get_scheduler(params, optimizer, rate=rate, decay=decay, max_learning_rate=max_learning_rate, epsilon=epsilon);
    loss_gradients = [T.grad(loss, param) for param in params]
    for param, gradient in zip(params, loss_gradients):
        scheduler.update(param, gradient, updates)
    return theano.function(inputs, [loss], updates=updates, on_unused_input='ignore')
