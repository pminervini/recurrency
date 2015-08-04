#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T
import layer

import logging


class RNN(layer.Layer):

    def get_parameters(self, rng, n_in, n_hidden, n_out):
        b_h = self.initialize(rng, size=(n_hidden,), tag='b_h')
        W_ih = self.initialize(rng, size=(n_in, n_hidden), tag='W_ih')
        W_hh = self.initialize(rng, size=(n_hidden, n_hidden), tag='W_hh')
        W_ho = self.initialize(rng, size=(n_hidden, n_out), tag='Who')
        b_o = self.initialize(rng, size=(n_out,), tag='b_o')
        h0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), name='h0')
        return [W_ih, W_hh, b_h, W_ho, b_o], [h0]

    # sequences: x_t, prior results: h_tm1, non-sequences: W_ih, W_hh, W_ho, b_h
    def step(self, x_t, h_tm1, W_ih, W_hh, b_h, W_ho, b_o):
        # e(W_ih x_t + W_hh h_tm1 + bh)
        h_t = T.tanh(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh) + b_h)
        # g(W_ho h_t + b_o)
        y_t = theano.dot(h_t, W_ho) + b_o
        y_t = self.act(y_t)
        return [h_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out):
        self.name = 'RNN(%i, %i, %i)' % (n_in, n_hidden, n_out)

        self.act = T.nnet.sigmoid
        self.non_sequences, self.sequences = self.get_parameters(rng, n_in, n_hidden, n_out)

        self.params = self.non_sequences + self.sequences

    def __call__(self, x):
        # hidden and outputs of the entire sequence
        [h_vals, y_vals], _ = theano.scan(fn=self.step,
                                            sequences=dict(input=x, taps=[0]),
                                            outputs_info=self.sequences + [None], # initialiation
                                            non_sequences=self.non_sequences) # unchanging variables
        return y_vals, [h_vals]
