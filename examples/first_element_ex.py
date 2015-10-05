#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import theano
import theano.tensor as T

import recurrency.layers.regularization as regularization
import recurrency.layers.activation as activation

import recurrency.layers.recurrent as recurrent
import recurrency.optimization.optimization as optimization

import examples.utils as utils

import collections

import random
import sys
import getopt
import logging

def experiment(model=None, epochs=20, hidden=100, method='momentum', rate=0.001, momentum=0.99, L1=None, L2=None, batch_size=100,
                sequence_length=10):
    x, idx = T.tensor3(), T.ivector()
    y, _ = model(x)

    # How the model provides the predicted class
    predicted_class = y[idx]
    predicted_class = T.clip(predicted_class, 1e-6, 1.0 - 1e-6)

    # Loss function (negative log-likelihood)
    real_class = T.matrix()
    loss = T.sum(T.nnet.binary_crossentropy(predicted_class, real_class))

    l1, l2 = regularization.L1Regularizer(), regularization.L2Regularizer()

    reg = 0.0
    for param in model.params:
        reg += (L1 * l1(param)) if L1 is not None else 0. + (L2 * l2(param)) if L2 is not None else 0.

    loss += reg

    # Minimization function (SGD with adaptive learning rates)
    f = optimization.minimize([x, idx, real_class], loss, model.params, optimizer=method, rate=rate, momentum=momentum)

    for epoch in range(epochs):
        loss_train = .0

        for i in range(100):
            batch = np.random.randint(2, size=(batch_size, sequence_length, 1))
            batch = batch.astype(theano.config.floatX)

            loss_train += f(batch, [sequence_length - 1] * batch_size, batch[:, 0])[0]

        logging.info('[%s %i]\t%s' % (model.name, epoch, loss_train))



def main(argv):
    model_name, method = 'rnn', 'adagrad'
    epochs, hidden, rate, momentum = 1000, 10, 0.001, 0.95
    L1, L2 = None, None
    batch_size = 10

    sequence_length = 20

    act_name = 'Sigmoid'
    sigma_name, g_name, h_name = 'Sigmoid', 'Linear', 'Linear'

    save_path = None

    # Non-linearities in the model (let's keep it simple)
    act = activation.Sigmoid()
    sigma, g, h = activation.Sigmoid(), activation.Linear(), activation.Linear()

    usage_str = ('Usage: %s [-h] [--model=<value>] [--epochs=<value>] [--hidden=<value>] [--rate=<value>] [--momentum=<value>] [--L1=<value>] [--L2=<value>] [--batch_size=<value>] [--sequence_length=<value>]' % (sys.argv[0]))

    try:
        opts, args = getopt.getopt(argv, 'h', ['model=', 'epochs=', 'hidden=', 'act=', 'sigma=', 'g=', 'h=', 'method=', 'rate=', 'momentum=', 'L1=', 'L2=', 'batch_size=', 'sequence_length='])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage_str)

            logging.info('\t--model=<value> Model to use. Options: rnn, lstm, lstmp, gru, mut1 (default: %s)' % model_name)
            logging.info('\t--epochs=<value> Number of epochs. (default: %s)' % epochs)
            logging.info('\t--hidden=<value> Number of hidden units. (default: %s)' % hidden)

            logging.info('\t--method=<value> Optimization method to use. Options: sgd, momentum, adagrad, adadelta, rmsprop (default: %s)' % method)
            logging.info('\t--rate=<value> Learning rate. (default: %s)' % rate)
            logging.info('\t--momentum=<value> Momentum. (default: %s)' % momentum)

            logging.info('\t--L1=<value> L1 regularization weight. (default: %s)' % L1)
            logging.info('\t--L2=<value> L2 regularization weight. (default: %s)' % L2)

            logging.info('\t--batch_size=<value> Batch size. (default: %s)' % batch_size)

            logging.info('\t--sequence_length=<value> Sequence length. (default: %s)' % sequence_length)
            return
        elif opt == '--model':
            model_name = arg
        elif opt == '--epochs':
            epochs = int(arg)
        elif opt == '--hidden':
            hidden = int(arg)

        elif opt == '--act':
            act_name = arg
        elif opt == '--sigma':
            sigma_name = arg
        elif opt == '--g':
            g_name = arg
        elif opt == '--h':
            h_name = arg

        elif opt == '--method':
            method = arg
        elif opt == '--rate':
            rate = float(arg)
        elif opt == '--momentum':
            momentum = float(arg)

        elif opt == '--L1':
            L1 = float(arg)
        elif opt == '--L2':
            L2 = float(arg)

        elif opt == '--batch_size':
            batch_size = int(arg)
        elif opt == '--sequence_length':
            sequence_length = int(arg)

    act = utils.get_activation(act_name)
    sigma = utils.get_activation(sigma_name)
    g = utils.get_activation(g_name, shape=(hidden,))
    h = utils.get_activation(h_name)

    np.random.seed(123)

    # input size: ND, hidden layer size: 10, output size: 1
    n_in, n_hidden, n_out = 1, hidden, 1

    # Model
    model = None
    if model_name == 'rnn':
        model = recurrent.RNN(np.random, n_in, n_hidden, n_out, g=g, act=act)
    elif model_name == 'mut1':
        model = recurrent.Mutation1(np.random, n_in, n_hidden, n_out, g=g, act=act)
    elif model_name == 'mut2':
        model = recurrent.Mutation2(np.random, n_in, n_hidden, n_out, g=g, act=act)
    elif model_name == 'lstm':
        model = recurrent.LSTM(np.random, n_in, n_hidden, n_out, sigma=sigma, g=g, h=h, act=act)
    elif model_name == 'lstmp':
        model = recurrent.LSTMP(np.random, n_in, n_hidden, n_out, sigma=sigma, g=g, h=h, act=act)
    elif model_name == 'gru':
        model = recurrent.GRU(np.random, n_in, n_hidden, n_out, sigma=sigma, g=g, act=act)
    else:
        raise ValueError('Unknown model: %s' % (model_name))

    experiment(model=model, epochs=epochs, hidden=hidden, method=method, rate=rate, momentum=momentum, L1=L1, L2=L2, batch_size=batch_size,
                sequence_length=sequence_length)

import warnings

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
