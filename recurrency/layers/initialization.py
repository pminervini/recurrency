#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import theano
import theano.tensor as T

import numpy as np

from recurrency.utils import sharedX, shared_zeros, shared_ones

import logging

class Initializer(object):
    __metaclass__ = ABCMeta
    def __init__(self, rng):
        self.rng = rng
    def get_fans(self, shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out
    @abstractmethod
    def __call__(self, shape, name=None):
        pass

class Uniform(Initializer):
    def __init__(self, rng, scale=0.05):
        super(Uniform, self).__init__(rng)
        self.scale = scale
    def __call__(self, shape, name=None):
        return sharedX(self.rng.uniform(low=-self.scale, high=self.scale, size=shape), name=name)

class Normal(Initializer):
    def __init__(self, rng, scale=0.05):
        super(Normal, self).__init__(rng)
        self.scale = scale
    def __call__(self, shape, name=None):
        return sharedX(self.rng.randn(*shape) * self.scale, name=name)

class LeCunUniform(Initializer):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    def __init__(self, rng):
        super(LeCunUniform, self).__init__(rng)
    def __call__(self, shape, name=None):
        fan_in, fan_out = self.get_fans(shape)
        scale = np.sqrt(3. / fan_in)
        uniform = Uniform(self.rng, scale=scale)
        return uniform(shape, name=name)

class GlorotNormal(Initializer):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    def __init__(self, rng):
        super(GlorotNormal, self).__init__(rng)
    def __call__(self, shape, name=None):
        fan_in, fan_out = self.get_fans(shape)
        scale = np.sqrt(2. / (fan_in + fan_out))
        normal = Normal(self.rng, scale=scale)
        return normal(shape, name=name)

class GlorotUniform(Initializer):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    def __init__(self, rng):
        super(GlorotUniform, self).__init__(rng)
    def __call__(self, shape, name=None):
        fan_in, fan_out = self.get_fans(shape)
        scale = np.sqrt(6. / (fan_in + fan_out))
        uniform = Uniform(self.rng, scale=scale)
        return uniform(shape, name=name)

class HeNormal(Initializer):
    ''' Reference: He et al., http://arxiv.org/abs/1502.01852
    '''
    def __init__(self, rng):
        super(HeNormal, self).__init__(rng)
    def __call__(self, shape, name=None):
        fan_in, fan_out = self.get_fans(shape)
        scale = np.sqrt(2. / fan_in)
        normal = Normal(self.rng, scale=scale)
        return normal(shape, name=name)

class HeUniform(Initializer):
    ''' Reference: He et al., http://arxiv.org/abs/1502.01852
    '''
    def __init__(self, rng):
        super(HeUniform, self).__init__(rng)
    def __call__(self, shape, name=None):
        fan_in, fan_out = self.get_fans(shape)
        scale = np.sqrt(6. / fan_in)
        uniform = Uniform(self.rng, scale=scale)
        return uniform(shape, name=name)

class Orthogonal(Initializer):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def __init__(self, rng, scale=1.1):
        super(Orthogonal, self).__init__(rng)
        self.scale = scale
    def __call__(self, shape, name=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = self.rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return sharedX(self.scale * q[:shape[0], :shape[1]], name=name)

class Homogeneous(Initializer):
    def __init__(self, scale=0.0):
        super(Homogeneous, self).__init__(rng=None)
        self.scale = scale
    def __call__(self, shape, name=None):
        return sharedX(self.scale * np.ones(shape), dtype=theano.config.floatX, name=name)

class Diagonal(Initializer):
    def __init__(self, scale=1.):
        super(Diagonal, self).__init__(rng=None)
        self.scale = scale
    def __call__(self, shape, name=None):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise Exception("Identity matrix initialization can only be used for 2D square matrices")
        else:
            return sharedX(self.scale * np.identity(shape[0]), name=name)

def initialize(rng, size, name='', type='glorot_uniform'):
    initializer = None
    if type in ['glorot_normal']:
        initializer = GlorotNormal(rng)
    elif type in ['glorot_uniform']:
        initializer = GlorotUniform(rng)
    elif type in ['he_normal']:
        initializer = HeNormal(rng)
    elif type in ['he_uniform']:
        initializer = HeUniform(rng)
    elif type in ['lecun_uniform']:
        initializer = LeCunUniform(rng)
    elif type in ['orthogonal']:
        initializer = Orthogonal(rng)
    elif type in ['zero', 'zeros']:
        initializer = Homogeneous(scale=0.)
    elif type in ['identity', 'le2015']:
        initializer = Diagonal(scale=1.)
    if initializer is None:
        raise ValueError('Unknown initialization: %s' % (type))
    return initializer(size, name=name)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for type in ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_uniform', 'orthogonal', 'zeros', 'ones']:
        logging.info('Type: %s' % type)
        np.random.seed(123)
        V = initialize(np.random, size=(5,5), type=type)
        logging.info(V.get_value())
