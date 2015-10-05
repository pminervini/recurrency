# -*- coding: utf-8 -*-

import recurrency.layers.activation as activation

def get_activation(name, shape=None):
    act = None
    if name == 'Linear':
        act = activation.Linear()
    if name == 'Sigmoid':
        act = activation.Sigmoid()
    if name == 'Tanh':
        act = activation.Tanh()
    if name == 'ReLU':
        act = activation.ReLU()
    if name == 'PReLU':
        act = activation.PReLU(shape)
    return act

def create_onehot_matrix(n_columns, list_idx):
    n_rows = len(listidx)
    cooData, cooRowIdxs, cooColIdxs = np.ones(n, dtype=theano.config.floatX), range(n_rows), list_idx
    mat = sp.sparse.coo_matrix((cooData, (cooRowIdxs, cooColIdxs)), shape=(n_rows, n_columns))
    return mat.toarray()
