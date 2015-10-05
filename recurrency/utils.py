# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def shared_scalar(val=0., dtype=theano.config.floatX, name=None):
    return theano.shared(np.cast[dtype](val))

def shared_ones(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape), dtype=dtype, name=name)

def alloc_zeros_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](0.), *dims)

def ndim_tensor(ndim):
    ret = None
    if ndim == 2:
        ret = T.matrix()
    elif ndim == 3:
        ret = T.tensor3()
    elif ndim == 4:
        ret = T.tensor4()
    else:
        ret = T.matrix()
    return ret
