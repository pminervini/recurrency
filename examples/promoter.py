#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import recurrency.layers.recurrent as recurrent
import recurrency.optimization.optimization as optimization

import data.promoter.promoter as promoter

import collections

import logging
import getopt
import sys


def experiment(model, optimizer='sgd', rate=0.1, decay=0.95, epsilon=1e-6):
    np.random.seed(123)

    x = T.matrix(dtype=theano.config.floatX)
    y, _ = model(x)

    t = T.vector()
    loss = T.abs_(t - y[-1]).mean(axis = 0).sum()

    f = optimization.minimize([x, t], loss, model.params, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)

    dataset = promoter.Promoter()
    N = len(dataset.labels)

    NV = 10
    NT = N - NV

    order = np.random.permutation(N)
    train_idx, valid_idx = order[NV:], order[:NV]

    fy = theano.function([x, t], [loss])

    for epoch in range(10000):
        loss_train, loss_valid, order = .0, .0, np.random.permutation(NT)

        for i in range(NT):
            sequence = dataset.sequences[train_idx[order[i]]]
            label = dataset.labels[train_idx[order[i]]]
            loss_train += f(sequence, label)[0]

        for i in range(NV):
            sequence = dataset.sequences[valid_idx[i]]
            label = dataset.labels[valid_idx[i]]
            yi = fy(sequence, label)[0]
            loss_valid += np.abs(yi - label)

        logging.info('[%s %i]\t%s\t%s' % (model.name, epoch, loss_train, loss_valid))

def main(argv):
    is_rnn, is_lstm, is_lstmp, is_gru = False, False, False, False
    optimizer, rate, decay, epsilon = 'sgd', 0.1, 0.95, 1e-6
    n_hidden = 10

    usage_str = ('Usage: %s [-h] [--rnn] [--lstm] [--lstmp] [--gru] [--hidden=<hidden>] [--optimizer=<optimizer>] [--rate=<rate>] [--decay=<decay>] [--epsilon=<epsilon>]' % (sys.argv[0]))

    try:
        opts, args = getopt.getopt(argv, 'h', ['rnn', 'lstm', 'lstmp', 'gru', 'hidden=', 'optimizer=', 'rate=', 'decay=', 'epsilon='])
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
        elif opt == '--optimizer':
            optimizer = arg

        elif opt == '--rate':
            rate = float(arg)
        elif opt == '--decay':
            decay = float(arg)
        elif opt == '--epsilon':
            epsilon = float(arg)

    n_in, n_out = 4, 1

    if is_rnn:
        model = recurrent.RNN(np.random, n_in, n_hidden, n_out, act=T.nnet.sigmoid)
        experiment(model, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_lstm:
        model = recurrent.LSTM(np.random, n_in, n_hidden, n_out, act=T.nnet.sigmoid)
        experiment(model, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_lstmp:
        model = recurrent.LSTMP(np.random, n_in, n_hidden, n_out, act=T.nnet.sigmoid)
        experiment(model, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)
    if is_gru:
        model = recurrent.GRU(np.random, n_in, n_hidden, n_out, act=T.nnet.sigmoid)
        experiment(model, optimizer=optimizer, rate=rate, decay=decay, epsilon=epsilon)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
