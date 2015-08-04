#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T
import layer

import logging


#
# Sak, H. et al. - Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
#
class LSTM(layer.Layer):

    def get_parameters(self, rng, n_in, n_out, n_i, n_c, n_o, n_f, n_hidden):
        W_xi = self.initialize(rng, size=(n_in, n_i), tag='W_xi')
        W_hi = self.initialize(rng, size=(n_hidden, n_i), tag='W_hi')
        W_ci = self.initialize(rng, size=(n_c, n_i), tag='W_ci')
        b_i = self.initialize(rng, size=(n_i,), tag='b_i')

        W_xf = self.initialize(rng, size=(n_in, n_f), tag='W_xf')
        W_hf = self.initialize(rng, size=(n_hidden, n_f), tag='W_hf')
        W_cf = self.initialize(rng, size=(n_c, n_f), tag='W_cf')
        b_f = self.initialize(rng, size=(n_f,), tag='b_f')

        W_xc = self.initialize(rng, size=(n_in, n_c), tag='W_xc')
        W_hc = self.initialize(rng, size=(n_hidden, n_c), tag='W_hc')
        b_c = self.initialize(rng, size=(n_c,), tag='b_c')

        W_xo = self.initialize(rng, size=(n_in, n_o), tag='W_xo')
        W_ho = self.initialize(rng, size=(n_hidden, n_o), tag='W_ho')
        W_co = self.initialize(rng, size=(n_c, n_o), tag='W_co')
        b_o = self.initialize(rng, size=(n_o,), tag='b_o')

        W_hy = self.initialize(rng, size=(n_hidden, n_out), tag='W_hy')
        b_y = self.initialize(rng, size=(n_out,), tag='b_y')

        c0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), name='c0')

        return [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y], c0

    # sequences: x_t
    # prior results: h_tm1, c_tm1
    # non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y
    def step(self, x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y):
        # i_t = sigma(W_xi x_t + W_hi h_tm1 + W_ci c_tm1 + b_i)
        i_t = self.sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
        # f_t = sigma(W_xf x_t + W_hf h_tm1 + W_cf c_tm1 + b_f)
        f_t = self.sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)
        # c_t = f_t * c_tm1 + i_t * g(W_xc x_t + W_hc h_tm1 + b_c)
        c_t = f_t * c_tm1 + i_t * self.act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
        # o_t = sigma(W_xo x_t + W_ho h_tm1 + W_co c_t + b_o)
        o_t = self.sigma(theano.dot(x_t, W_xo) + theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co) + b_o)
        # h_t = o_t * h(c_t)
        h_t = o_t * self.act(c_t)
        # y_t = phi(W_hy h_t + b_y)
        y_t = self.sigma(theano.dot(h_t, W_hy) + b_y)
        return [h_t, c_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out):
        self.name = 'LSTM(%i, %i, %i)' % (n_in, n_hidden, n_out)

        self.act = T.nnet.sigmoid
        self.sigma = lambda x : 1 / (1 + T.exp(-x))

        n_i = n_c = n_o = n_f = n_hidden

        self.non_sequences, c0 = self.get_parameters(rng, n_in, n_out, n_i, n_c, n_o, n_f, n_hidden)

        h0 = T.tanh(c0)
        self.sequences = [h0, c0]

        self.params = self.non_sequences + [c0]

    def __call__(self, x):
        # hidden and outputs of the entire sequence
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=self.step,
                                                    sequences=dict(input=x, taps=[0]),
                                                    outputs_info=self.sequences + [None], # corresponds to return type of fn
                                                    non_sequences=self.non_sequences)
        return y_vals, [h_vals, c_vals]
