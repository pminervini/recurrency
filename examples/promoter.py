#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import energy.rnn as rnn
import energy.lstm as lstm
import energy.lstmp as lstmp

import optim.optimization as O

import data.promoter.promoter as promoter

import collections

import logging
import getopt
import sys


def experiment(model, optimizer='sgd', rate=0.1, decay=0.95, epsilon=1e-6):
    np.random.seed(123)

    x = T.matrix(dtype=theano.config.floatX)
    y, _ = model(x)
    output = y[-1]

    f = O.minimize(x, output, model.params, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)

    dataset = promoter.Promoter()
    N = len(dataset.labels)

    NV = 10
    NT = N - NV

    order = np.random.permutation(N)
    train_idx, valid_idx = order[NV:], order[:NV]

    fy = theano.function([x], [output])

    for epoch in range(10000):
        loss_train, loss_valid, order = .0, .0, np.random.permutation(NT)

        for i in range(NT):
            sequence = dataset.sequences[train_idx[order[i]]]
            label = dataset.labels[train_idx[order[i]]]
            loss_train += f(sequence, label, rate)[0]

        for i in range(NV):
            sequence = dataset.sequences[valid_idx[i]]
            label = dataset.labels[valid_idx[i]][0]
            yi = fy(sequence)[0]
            loss_valid += np.abs(yi - label)

        logging.info('[%s %i]\t%s\t%s' % (model.name, epoch, loss_train, loss_valid))

def main(argv):
    is_rnn, is_lstm, is_lstmp = False, False, False
    optimizer, rate, decay, epsilon = 'sgd', 0.1, 0.95, 1e-6
    n_hidden = 10

    usage_str = ('Usage: %s [-h] [--rnn] [--lstm] [--lstmp] [--hidden=<hidden>] [--optimizer=<optimizer>] [--rate=<rate>] [--decay=<decay>] [--epsilon=<epsilon>]' % (sys.argv[0]))

    try:
        opts, args = getopt.getopt(argv, 'h', ['rnn', 'lstm', 'lstmp', 'hidden=', 'optimizer=', 'rate=', 'decay=', 'epsilon='])
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

        elif opt == '--hidden':
            n_hidden = int(arg)
        elif opt == '--optimizer':
            optimizer = arg

        elif opt == '--rate':
            rate = float(arg)
        elif opt == '--decay':
            decay = float(arg)
        elif opt == '--epsilon':
            epsilon = float(arg)

    if is_rnn:
        model = rnn.RNN(np.random, 4, n_hidden, 1)
        experiment(model, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_lstm:
        model = lstm.LSTM(np.random, 4, n_hidden, 1)
        experiment(model, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_lstmp:
        model = lstmp.LSTMP(np.random, 4, n_hidden, 1)
        experiment(model, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
