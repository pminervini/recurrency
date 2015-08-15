#!/usr/bin/python3 -uB
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import recurrency.layers.recurrent as recurrent
import recurrency.optimization.optimization as optimization

import collections

import logging
import getopt
import sys


def experiment(model, length, optimizer='sgd', rate=0.1, decay=0.95, epsilon=1e-6):
    np.random.seed(123)

    x = T.matrix(dtype=theano.config.floatX)
    y, _ = model(x)

    t = T.vector()
    loss = T.abs_(t - y[-1]).mean(axis = 0).sum()

    f = optimization.minimize([x, t], loss, model.params, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)

    for epoch in range(100):
        err = .0
        for i in range(1000):
            instance = np.random.randint(2, size=(length, 1)).astype(theano.config.floatX)
            vt = instance[0]
            err += f(instance, vt)[0]

        logging.info('[%s %i]\t%s' % (model.name, epoch, err))


def main(argv):
    is_rnn, is_lstm, is_lstmp, is_gru = False, False, False, False
    optimizer, rate, decay, epsilon = 'momentum', 0.1, 0.95, 1e-6
    length, n_hidden = 10, 10

    usage_str = ('Usage: %s [-h] [--rnn] [--lstm] [--lstmp] [--gru] [--hidden=<hidden>] [--length=<n>] [--optimizer=<optimizer>] [--rate=<rate>] [--decay=<decay>] [--epsilon=<epsilon>]' % (sys.argv[0]))

    try:
        opts, args = getopt.getopt(argv, 'h', ['rnn', 'lstm', 'lstmp', 'gru', 'hidden=', 'length=', 'optimizer=', 'rate=', 'decay=', 'epsilon='])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage_str)
            return
        elif opt == '--rnn':
            is_rnn = True
        elif opt == '--lstm':
            is_lstm = True
        elif opt == '--lstmp':
            is_lstmp = True
        elif opt == '--gru':
            is_gru = True

        elif opt == '--hidden':
            n_hidden = int(arg)

        elif opt == '--length':
            length = int(arg)

        elif opt == '--optimizer':
            optimizer = arg

        elif opt == '--rate':
            rate = float(arg)
        elif opt == '--decay':
            decay = float(arg)
        elif opt == '--epsilon':
            epsilon = float(arg)

    n_in, n_out = 1, 1

    if is_rnn:
        model = recurrent.RNN(np.random, n_in, n_hidden, n_out)
        experiment(model, length, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_lstm:
        model = recurrent.LSTM(np.random, n_in, n_hidden, n_out)
        experiment(model, length, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_lstmp:
        model = recurrent.LSTMP(np.random, n_in, n_hidden, n_out)
        experiment(model, length, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_gru:
        model = recurrent.GRU(np.random, n_in, n_hidden, n_out)
        experiment(model, length, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
