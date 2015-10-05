# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import recurrency.utils as utils
import recurrency.layers.layer as layer

import recurrency.layers.activation as activation
import recurrency.layers.noise as noise

import logging

"""
    This module implements several Recurrent Neural Network architectures, by extending
    the layer.Layer abstract class.

    Note: the act() constructor parameter is the activation function used for the output variables y_t (e.g. softmax). By default it is the identity function.
"""
sigma, g, h = activation.Sigmoid(), activation.Sigmoid(), activation.Sigmoid()
act = activation.Linear()

class RecurrentLayer(layer.Layer):
    '''
        Abstract class for Recurrent Neural Network layers.
    '''
    def __init__(self):
        super(RecurrentLayer, self).__init__()
        self.params += []

    def __call__(self, x):
        # Input is provided as (n_batch, n_time_steps, n_features): since theano.scan iterates
        # over the first dimension, we dimshuffle to (n_time_steps, n_batch, n_features)
        x = x.dimshuffle(1, 0, 2)
        return x

class RNN(RecurrentLayer):
    '''
        Fully connected Recurrent Neural Network (RNN), as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        h_t = g(W_ih x_t + W_hh h_t-1 + bh),
        y_t = act(W_ho h_t + b_o),

        where g() and act() are element-wise non-linear functions.

        [1] Martens, J. et al. - Learning Recurrent Neural Networks with Hessian-Free Optimization - ICML 2011
    '''

    def get_parameters(self, rng, n_in, n_hidden, n_out):
        b_h = self.initialize(rng, size=(n_hidden,), tag='b_h')
        W_ih = self.initialize(rng, size=(n_in, n_hidden), tag='W_ih')
        W_hh = self.initialize(rng, size=(n_hidden, n_hidden), tag='W_hh', type='le2015')
        W_ho = self.initialize(rng, size=(n_hidden, n_out), tag='W_ho')
        b_o = self.initialize(rng, size=(n_out,), tag='b_o', type='zero')
        h0 = self.initialize(rng, size=(n_hidden,), tag='h_0', type='zero')
        return [W_ih, W_hh, b_h, W_ho, b_o], [h0]

    # sequences: x_t, prior results: h_tm1, non-sequences: W_ih, W_hh, W_ho, b_h
    def step(self, x_t, h_tm1, W_ih, W_hh, b_h, W_ho, b_o):

        # h_t = g(W_ih x_t + W_hh h_tm1 + bh)

        ### Does not work on recurrent layer, see http://arxiv.org/pdf/1311.0701v7.pdf
        h_t = self.g(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh) + b_h)

        # y_t = act(W_ho h_t + b_o)

        ### y_t = self.act(theano.dot(h_t, W_ho) + b_o)
        y_t = self.act(theano.dot(h_t, W_ho) + b_o)

        return [h_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out, g=g, act=act):
        super(RNN, self).__init__()

        self.name = 'RNN(%i, %i, %i)' % (n_in, n_hidden, n_out)
        self.n_hidden = n_hidden
        self.g, self.act = g, act

        self.params += self.g.params + self.act.params

        self.non_sequences, self.sequences = self.get_parameters(rng, n_in, n_hidden, n_out)
        self.params += self.non_sequences + self.sequences

    def __call__(self, x):
        x = super(RNN, self).__call__(x)
        # hidden and outputs of the entire sequence
        [h_vals, y_vals], _ = theano.scan(fn=self.step,
                                            sequences={'input': x, 'taps': [0]},
                                            outputs_info=[T.alloc(self.sequences, x.shape[1], self.n_hidden), None], # initialiation
                                            non_sequences=self.non_sequences) # unchanging variables
        return y_vals, [h_vals]

class LSTM(RecurrentLayer):
    '''
        Long Short-Term Memory (LSTM) Recurrent Neural Network, as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        i_t = σ(W_xi x_t + W_hi h_t-1 + W_ci c_t-1 + b_i),
        f_t = σ(W_xf x_t + W_hf h_t-1 + W_cf c_t-1 + b_f),
        c_t = f_t * c_t-1 + i_t * g(W_xc x_t + W_hc h_t-1 + b_c),
        o_t = σ(W_xo x_t + W_ho h_t-1 + W_co c_t + b_o),
        h_t = o_t * h(c_t), (m_t in the article)
        y_t = act(W_hy h_t + b_y),

        where σ(), g(), h() and act() are element-wise non-linear functions.

        [1] Sak, H. et al. - Long short-term memory recurrent neural network architectures for large scale acoustic modeling. - INTERSPEECH 2014
    '''

    def get_parameters(self, rng, n_in, n_out, n_i, n_c, n_o, n_f, n_hidden):
        W_xi = self.initialize(rng, size=(n_in, n_i), tag='W_xi')
        W_hi = self.initialize(rng, size=(n_hidden, n_i), tag='W_hi')
        W_ci = self.initialize(rng, size=(n_c, n_i), tag='W_ci')
        b_i = self.initialize(rng, size=(n_i,), tag='b_i', type='zero')

        W_xf = self.initialize(rng, size=(n_in, n_f), tag='W_xf')
        W_hf = self.initialize(rng, size=(n_hidden, n_f), tag='W_hf')
        W_cf = self.initialize(rng, size=(n_c, n_f), tag='W_cf')
        b_f = self.initialize(rng, size=(n_f,), tag='b_f', type='zero')

        W_xc = self.initialize(rng, size=(n_in, n_c), tag='W_xc')
        W_hc = self.initialize(rng, size=(n_hidden, n_c), tag='W_hc')
        b_c = self.initialize(rng, size=(n_c,), tag='b_c', type='zero')

        W_xo = self.initialize(rng, size=(n_in, n_o), tag='W_xo')
        W_ho = self.initialize(rng, size=(n_hidden, n_o), tag='W_ho')
        W_co = self.initialize(rng, size=(n_c, n_o), tag='W_co')
        b_o = self.initialize(rng, size=(n_o,), tag='b_o', type='zero')

        W_hy = self.initialize(rng, size=(n_hidden, n_out), tag='W_hy')
        b_y = self.initialize(rng, size=(n_out,), tag='b_y', type='zero')

        c0 = self.initialize(rng, size=(n_hidden,), tag='c_0', type='zero')

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
        c_t = f_t * c_tm1 + i_t * self.g(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)

        # o_t = sigma(W_xo x_t + W_ho h_tm1 + W_co c_t + b_o)
        o_t = self.sigma(theano.dot(x_t, W_xo) + theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co) + b_o)

        # h_t = o_t * h(c_t)
        h_t = o_t * self.h(c_t)

        # y_t = act(W_hy h_t + b_y)
        y_t = self.act(theano.dot(h_t, W_hy) + b_y)

        return [h_t, c_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out, sigma=sigma, g=g, h=h, act=act):
        super(LSTM, self).__init__()

        self.name = 'LSTM(%i, %i, %i)' % (n_in, n_hidden, n_out)
        self.n_hidden = n_hidden
        self.sigma, self.g, self.h = sigma, g, h
        self.act = act

        self.params += self.sigma.params + self.g.params + self.h.params + self.act.params

        n_i = n_c = n_o = n_f = n_hidden

        self.non_sequences, c0 = self.get_parameters(rng, n_in, n_out, n_i, n_c, n_o, n_f, n_hidden)

        h0 = self.h(c0)
        self.sequences = [h0, c0]

        self.params += self.non_sequences + [c0]

    def __call__(self, x):
        x = super(LSTM, self).__call__(x)
        # hidden and outputs of the entire sequence
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=self.step,
                                                    sequences={'input': x, 'taps': [0]},
                                                    outputs_info=[T.alloc(self.sequences[0], x.shape[1], self.n_hidden), T.alloc(self.sequences[1], x.shape[1], self.n_hidden), None], # initialiation
                                                    non_sequences=self.non_sequences)
        return y_vals, [h_vals, c_vals]

class LSTMP(RecurrentLayer):
    '''
        Projected Long Short-Term Meory (LSTMP) Recurrent Neural Network, as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        i_t = σ(W_xi x_t + W_hi h_t-1 + W_ci c_t-1 + b_i),
        f_t = σ(W_xf x_t + W_hf h_t-1 + W_cf c_t-1 + b_f),
        c_t = f_t * c_t-1 + i_t * g(W_xc x_t + W_hc h_t-1 + b_c),
        o_t = σ(W_xo x_t + W_ho h_t-1 + W_co c_t + b_o),
        h_t = o_t * h(c_t),
        r_t = W_hr h_t,
        y_t = act(W_ry r_t + b_y)

        where σ(), g(), h() and act() are element-wise non-linear functions.

        [1] Sak, H. et al. - Long short-term memory recurrent neural network architectures for large scale acoustic modeling. - INTERSPEECH 2014
    '''

    def get_parameters(self, rng, n_in, n_out, n_i, n_c, n_r, n_o, n_f, n_hidden):
        W_xi = self.initialize(rng, size=(n_in, n_i), tag='W_xi')
        W_hi = self.initialize(rng, size=(n_hidden, n_i), tag='W_hi')
        W_ci = self.initialize(rng, size=(n_c, n_i), tag='W_ci')
        b_i = self.initialize(rng, size=(n_i,), tag='b_i', type='zero')

        W_xf = self.initialize(rng, size=(n_in, n_f), tag='W_xf')
        W_hf = self.initialize(rng, size=(n_hidden, n_f), tag='W_hf')
        W_cf = self.initialize(rng, size=(n_c, n_f), tag='W_cf')
        b_f = self.initialize(rng, size=(n_f,), tag='b_f', type='zero')

        W_xc = self.initialize(rng, size=(n_in, n_c), tag='W_xc')
        W_hc = self.initialize(rng, size=(n_hidden, n_c), tag='W_hc')
        b_c = self.initialize(rng, size=(n_c,), tag='b_c', type='zero')

        W_xo = self.initialize(rng, size=(n_in, n_o), tag='W_xo')
        W_ho = self.initialize(rng, size=(n_hidden, n_o), tag='W_ho')
        W_co = self.initialize(rng, size=(n_c, n_o), tag='W_co')
        b_o = self.initialize(rng, size=(n_o,), tag='b_o', type='zero')

        W_hr = self.initialize(rng, size=(n_hidden, n_r), tag='W_hr')
        W_ry = self.initialize(rng, size=(n_r, n_out), tag='W_ry')

        b_y = self.initialize(rng, size=(n_out,), tag='b_y', type='zero')

        c0 = self.initialize(rng, size=(n_hidden,), tag='c_0', type='zero')

        return [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hr, W_ry, b_y], c0

    # sequences: x_t
    # prior results: h_tm1, c_tm1
    # non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hr, W_ry, b_y
    def step(self, x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hr, W_ry, b_y):
        # i_t = sigma(W_xi x_t + W_hi h_tm1 + W_ci c_tm1 + b_i)
        i_t = self.sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) +
        theano.dot(c_tm1, W_ci) + b_i)

        # f_t = sigma(W_xf x_t + W_hf h_tm1 + W_cf c_tm1 + b_f)
        f_t = self.sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)

        # c_t = f_t * c_tm1 + i_t * g(W_xc x_t + W_hc h_tm1 + b_c)
        c_t = f_t * c_tm1 + i_t * self.g(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)

        # o_t = sigma(W_xo x_t + W_ho h_tm1 + W_co c_t + b_o)
        o_t = self.sigma(theano.dot(x_t, W_xo) + theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co) + b_o)

        # h_t = o_t * h(c_t)
        h_t = o_t * self.h(c_t)

        # r_t = W_hr h_t
        r_t = theano.dot(h_t, W_hr)

        # y_t = act(W_ry r_t + b_y)
        y_t = self.act(theano.dot(r_t, W_ry) + b_y)

        return [h_t, c_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out, sigma=sigma, g=g, h=h, act=act):
        super(LSTMP, self).__init__()

        self.name = 'LSTMP(%i, %i, %i)' % (n_in, n_hidden, n_out)
        self.n_hidden = n_hidden
        self.sigma, self.g, self.h = sigma, g, h
        self.act = act

        self.params += self.sigma.params + self.g.params + self.h.params + self.act.params

        n_i = n_c = n_r = n_o = n_f = n_hidden

        self.non_sequences, c0 = self.get_parameters(rng, n_in, n_out, n_i, n_c, n_r, n_o, n_f, n_hidden)

        h0 = T.tanh(c0)
        self.sequences = [h0, c0]

        self.params += self.non_sequences + [c0]

    def __call__(self, x):
        x = super(LSTMP, self).__call__(x)
        # hidden and outputs of the entire sequence
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=self.step,
                                                    sequences={'input': x, 'taps': [0]},
                                                    outputs_info=[T.alloc(self.sequences[0], x.shape[1], self.n_hidden), T.alloc(self.sequences[1], x.shape[1], self.n_hidden), None], # initialiation
                                                    non_sequences=self.non_sequences)
        return y_vals, [h_vals, c_vals]

class GRU(RecurrentLayer):
    '''
        Gated Recurrent Unit (GRU) Recurrent Neural Network, as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        z_t = σ(W_z x_t + U_z h_t-1)
        r_t = σ(W_r x_t + U_r h_t-1)
        ~h_t = g(W x_t + r_t * U h_t-1)
        h_t = (1 - z_t) * h_t-1 + z_t * ~h_t
        y_t = act(W_hy h_t + b_y),

        where σ() and phi() are element-wise non-linear functions.

        [1] Cho, K. et al. - Learning phrase representations using RNN encoder-decoder for statistical machine translation. - arXiv:1406.1078
    '''

    def get_parameters(self, rng, n_in, n_out, n_z, n_r, n_t, n_h):
        W_xz = self.initialize(rng, size=(n_in, n_z), tag='W_xz')
        U_hz = self.initialize(rng, size=(n_h, n_z), tag='U_xz')
        b_z = self.initialize(rng, size=(n_z,), tag='b_z', type='zero')

        W_xr = self.initialize(rng, size=(n_in, n_r), tag='W_xr')
        U_hr = self.initialize(rng, size=(n_h, n_r), tag='U_hr')
        b_r = self.initialize(rng, size=(n_r,), tag='b_r', type='zero')

        W_xt = self.initialize(rng, size=(n_in, n_t), tag='W_xt')
        U_ht = self.initialize(rng, size=(n_h, n_t), tag='U_ht')

        W_hy = self.initialize(rng, size=(n_h, n_out), tag='W_hy')
        b_y = self.initialize(rng, size=(n_out,), tag='b_y', type='zero')

        h0 = self.initialize(rng, size=(n_h,), tag='h_0', type='zero')

        return [W_xz, U_hz, b_z, W_xr, U_hr, b_r, W_xt, U_ht, W_hy, b_y], [h0]

    # sequences: x_t
    # prior results: h_t-1
    # non-sequences: W_xz, U_hz, b_z, W_xr, U_hr, b_r, W_xt, U_ht, W_hy, b_y
    def step(self, x_t, h_tm1, W_xz, U_hz, b_z, W_xr, U_hr, b_r, W_xt, U_ht, W_hy, b_y):
        # z_t = sigma(W_xz x_t + U_hz h_t-1 + b_z)
        z_t = self.sigma(theano.dot(x_t, W_xz) + theano.dot(h_tm1, U_hz) + b_z)

        # r_t = sigma(W_xr x_t + U_hr h_t-1 + b_r)
        r_t = self.sigma(theano.dot(x_t, W_xr) + theano.dot(h_tm1, U_hr) + b_r)

        # ~h_t = g(W_xt x_t + r_t * U_ht h_t-1)
        t_t = self.g(theano.dot(x_t, W_xt) + r_t * theano.dot(h_tm1, U_ht))

        # h_t = (1 - z_t) * h_t-1 + z_t * ~h_t
        h_t = (1. - z_t) * h_tm1 + z_t * t_t

        # y_t = act(W_hy h_t + b_y)
        y_t = self.act(theano.dot(h_t, W_hy) + b_y)

        return [h_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out, sigma=sigma, g=g, act=act):
        super(GRU, self).__init__()

        self.name = 'GRU(%i, %i, %i)' % (n_in, n_hidden, n_out)
        self.n_hidden = n_hidden
        self.sigma, self.g = sigma, g
        self.act = act

        self.params += self.sigma.params + self.g.params + self.act.params

        n_z = n_r = n_t = n_hidden

        self.non_sequences, self.sequences = self.get_parameters(rng, n_in, n_out, n_z, n_r, n_t, n_hidden)

        self.params += self.non_sequences

    def __call__(self, x):
        x = super(GRU, self).__call__(x)
        # hidden and outputs of the entire sequence
        [h_vals, y_vals], _ = theano.scan(fn=self.step,
                                            sequences={'input': x, 'taps': [0]},
                                            outputs_info=[T.alloc(self.sequences, x.shape[1], self.n_hidden), None], # initialiation
                                            non_sequences=self.non_sequences)
        return y_vals, [h_vals]
