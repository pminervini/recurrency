# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

import recurrency.utils.utils as utils
import recurrency.layers.layer as layer

import logging


class RNN(layer.Layer):
    '''
        Fully connected Recurrent Neural Network (RNN), as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        h_t = e(W_ih x_t + W_hh h_t-1 + bh),
        y_t = g(W_ho h_t + b_o),

        where e() and g() are element-wise non-linear functions.

        [1] Sak, H. et al. - Long short-term memory recurrent neural network architectures for large scale acoustic modeling. - INTERSPEECH 2014
    '''

    def get_parameters(self, rng, n_in, n_hidden, n_out):
        b_h = self.initialize(rng, size=(n_hidden,), tag='b_h')
        W_ih = self.initialize(rng, size=(n_in, n_hidden), tag='W_ih')
        W_hh = self.initialize(rng, size=(n_hidden, n_hidden), tag='W_hh')
        W_ho = self.initialize(rng, size=(n_hidden, n_out), tag='W_ho')
        b_o = self.initialize(rng, size=(n_out,), tag='b_o')
        h0 = utils.shared_zeros(n_hidden, name='h_0')
        return [W_ih, W_hh, b_h, W_ho, b_o], [h0]

    # sequences: x_t, prior results: h_tm1, non-sequences: W_ih, W_hh, W_ho, b_h
    def step(self, x_t, h_tm1, W_ih, W_hh, b_h, W_ho, b_o):

        # h_t = e(W_ih x_t + W_hh h_tm1 + bh)
        h_t = T.tanh(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh) + b_h)

        # y_t g(W_ho h_t + b_o)
        y_t = self.act(theano.dot(h_t, W_ho) + b_o)

        return [h_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out):
        super(RNN, self).__init__()
        self.name = 'RNN(%i, %i, %i)' % (n_in, n_hidden, n_out)

        self.act = T.nnet.sigmoid
        self.non_sequences, self.sequences = self.get_parameters(rng, n_in, n_hidden, n_out)

        self.params += self.non_sequences + self.sequences

    def __call__(self, x):
        # hidden and outputs of the entire sequence
        [h_vals, y_vals], _ = theano.scan(fn=self.step,
                                            sequences=dict(input=x, taps=[0]),
                                            outputs_info=self.sequences + [None], # initialiation
                                            non_sequences=self.non_sequences) # unchanging variables
        return y_vals, [h_vals]



class LSTM(layer.Layer):
    '''
        Long Short-Term Meory (LSTM) Recurrent Neural Network, as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        i_t = σ(W_xi x_t + W_hi h_t-1 + W_ci c_t-1 + b_i),
        f_t = σ(W_xf x_t + W_hf h_t-1 + W_cf c_t-1 + b_f),
        c_t = f_t * c_t-1 + i_t * g(W_xc x_t + W_hc h_t-1 + b_c),
        o_t = σ(W_xo x_t + W_ho h_t-1 + W_co c_t + b_o),
        h_t = o_t * h(c_t),
        y_t = phi(W_hy h_t + b_y),

        where σ(), g(), h() and phi() are element-wise non-linear functions.

        [1] Sak, H. et al. - Long short-term memory recurrent neural network architectures for large scale acoustic modeling. - INTERSPEECH 2014
    '''

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

        c0 = utils.shared_zeros(n_hidden, name='c_0')

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
        super(LSTM, self).__init__()

        self.name = 'LSTM(%i, %i, %i)' % (n_in, n_hidden, n_out)

        self.act = T.nnet.sigmoid
        self.sigma = lambda x : 1 / (1 + T.exp(-x))

        n_i = n_c = n_o = n_f = n_hidden

        self.non_sequences, c0 = self.get_parameters(rng, n_in, n_out, n_i, n_c, n_o, n_f, n_hidden)

        h0 = T.tanh(c0)
        self.sequences = [h0, c0]

        self.params += self.non_sequences + [c0]

    def __call__(self, x):
        # hidden and outputs of the entire sequence
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=self.step,
                                                    sequences=dict(input=x, taps=[0]),
                                                    outputs_info=self.sequences + [None], # corresponds to return type of fn
                                                    non_sequences=self.non_sequences)
        return y_vals, [h_vals, c_vals]



class LSTMP(layer.Layer):
    '''
        Projected Long Short-Term Meory (LSTMP) Recurrent Neural Network, as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        i_t = σ(W_xi x_t + W_hi h_t-1 + W_ci c_t-1 + b_i),
        f_t = σ(W_xf x_t + W_hf h_t-1 + W_cf c_t-1 + b_f),
        c_t = f_t * c_t-1 + i_t * g(W_xc x_t + W_hc h_t-1 + b_c),
        o_t = σ(W_xo x_t + W_ho h_t-1 + W_co c_t + b_o),
        h_t = o_t * h(c_t),
        r_t = W_hr h_t,
        y_t = phi(W_ry r_t + b_y)

        where σ(), g(), h() and phi() are element-wise non-linear functions.

        [1] Sak, H. et al. - Long short-term memory recurrent neural network architectures for large scale acoustic modeling. - INTERSPEECH 2014
    '''

    def get_parameters(self, rng, n_in, n_out, n_i, n_c, n_r, n_o, n_f, n_hidden):

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

        W_hr = self.initialize(rng, size=(n_hidden, n_r), tag='W_hr')
        W_ry = self.initialize(rng, size=(n_r, n_out), tag='W_ry')

        b_y = self.initialize(rng, size=(n_out,), tag='b_y')

        c0 = utils.shared_zeros(n_hidden, name='c_0')

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
        c_t = f_t * c_tm1 + i_t * self.act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)

        # o_t = sigma(W_xo x_t + W_ho h_tm1 + W_co c_t + b_o)
        o_t = self.sigma(theano.dot(x_t, W_xo) + theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co) + b_o)

        # h_t = o_t * h(c_t)
        h_t = o_t * self.act(c_t)

        # r_t = W_hr h_t
        r_t = theano.dot(h_t, W_hr)

        # y_t = phi(W_ry r_t + b_y)
        y_t = self.sigma(theano.dot(r_t, W_ry) + b_y)

        return [h_t, c_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out):
        super(LSTMP, self).__init__()

        self.name = 'LSTMP(%i, %i, %i)' % (n_in, n_hidden, n_out)

        self.act = T.nnet.sigmoid
        self.sigma = lambda x : 1 / (1 + T.exp(-x))

        n_i = n_c = n_r = n_o = n_f = n_hidden

        self.non_sequences, c0 = self.get_parameters(rng, n_in, n_out, n_i, n_c, n_r, n_o, n_f, n_hidden)

        h0 = T.tanh(c0)
        self.sequences = [h0, c0]

        self.params += self.non_sequences + [c0]

    def __call__(self, x):
        # hidden and outputs of the entire sequence
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=self.step,
                                                    sequences=dict(input=x, taps=[0]),
                                                    outputs_info=self.sequences + [None], # corresponds to return type of fn
                                                    non_sequences=self.non_sequences)
        return y_vals, [h_vals, c_vals]



class GRU(layer.Layer):
    '''
        Gated Recurrent Unit (GRU) Recurrent Neural Network, as described in [1].
        Given an input sequence <x_1, .., x_t>, the output sequence <y_1, .., y_t> is calculated as follows:

        z_t = σ(W_z x_t + U_z h_t-1)
        r_t = σ(W_r x_t + U_r h_t-1)
        ~h_t = tanh(W x_t + r_t * U h_t-1)
        h_t = (1 - z_t) * h_t-1 + z_t * ~h_t
        y_t = phi(W_hy h_t + b_y),

        where σ() and phi() are element-wise non-linear functions.

        [1] Cho, K. et al. - Learning phrase representations using RNN encoder-decoder for statistical machine translation. - arXiv:1406.1078
    '''

    def get_parameters(self, rng, n_in, n_out, n_z, n_r, n_t, n_h):

        W_xz = self.initialize(rng, size=(n_in, n_z), tag='W_xz')
        U_hz = self.initialize(rng, size=(n_h, n_z), tag='U_xz')
        b_z = self.initialize(rng, size=(n_z,), tag='b_z')

        W_xr = self.initialize(rng, size=(n_in, n_r), tag='W_xr')
        U_hr = self.initialize(rng, size=(n_h, n_r), tag='U_hr')
        b_r = self.initialize(rng, size=(n_r,), tag='b_r')

        W_xt = self.initialize(rng, size=(n_in, n_t), tag='W_xt')
        U_ht = self.initialize(rng, size=(n_h, n_t), tag='U_ht')

        W_hy = self.initialize(rng, size=(n_h, n_out), tag='W_hy')
        b_y = self.initialize(rng, size=(n_out,), tag='b_y')

        z0 = utils.shared_zeros(n_z, name='z_0')
        r0 = utils.shared_zeros(n_z, name='r_0')
        h0 = utils.shared_zeros(n_z, name='h_0')
        t0 = utils.shared_zeros(n_z, name='t_0')

        return [W_xz, U_hz, b_z, W_xr, U_hr, b_r, W_xt, U_ht, W_hy, b_y], h0

    # sequences: x_t
    # prior results: h_t-1
    # non-sequences: W_xz, U_hz, b_z, W_xr, U_hr, b_r, W_xt, U_ht, W_hy, b_y
    def step(self, x_t, h_tm1, W_xz, U_hz, b_z, W_xr, U_hr, b_r, W_xt, U_ht, W_hy, b_y):

        # z_t = sigma(W_xz x_t + U_hz h_t-1 + b_z)
        z_t = self.sigma(theano.dot(x_t, W_xz) + theano.dot(h_tm1, U_hz) + b_z)

        # r_t = sigma(W_xr x_t + U_hr h_t-1 + b_r)
        r_t = self.sigma(theano.dot(x_t, W_xr) + theano.dot(h_tm1, U_hr) + b_r)

        # ~h_t = g(W_xt x_t + r_t * U_ht h_t-1)
        t_t = self.act(theano.dot(x_t, W_xt) + r_t * theano.dot(h_tm1, U_ht))

        # h_t = (1 - z_t) * h_t-1 + z_t * ~h_t
        h_t = (1. - z_t) * h_tm1 + z_t * t_t

        # y_t = phi(W_hy h_t + b_y)
        y_t = self.sigma(theano.dot(h_t, W_hy) + b_y)

        return [h_t, y_t]

    def __init__(self, rng, n_in, n_hidden, n_out):
        super(GRU, self).__init__()

        self.name = 'GRU(%i, %i, %i)' % (n_in, n_hidden, n_out)

        self.act = T.nnet.sigmoid
        self.sigma = lambda x : 1 / (1 + T.exp(-x))

        n_z = n_r = n_t = n_hidden

        self.non_sequences, h0 = self.get_parameters(rng, n_in, n_out, n_z, n_r, n_t, n_hidden)

        self.sequences = [h0]

        self.params += self.non_sequences

    def __call__(self, x):
        # hidden and outputs of the entire sequence
        [h_vals, y_vals], _ = theano.scan(fn=self.step,
                                                    sequences=dict(input=x, taps=[0]),
                                                    outputs_info=self.sequences + [None], # corresponds to return type of fn
                                                    non_sequences=self.non_sequences)
        return y_vals, [h_vals]
