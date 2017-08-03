import math

import numpy as np

import theano
import theano.tensor as T

def get_shared_shape(x):
    return x.get_value(borrow=True, return_internal_type=True).shape

def get_shared_zeros(shape):
    return theano.shared(np.zeros(shape))
    
"""
cf. Glorot and Bengio (2010)
"""
def glorot_uniform(shape, fan_in, fan_out):
    coef = math.sqrt(12 / (fan_in + fan_out))
    init_value = coef * (np.array(np.random.rand(
        *shape
    ), 'float64') - 0.5)
    return init_value

def init_var(method, shape, name):        
    if method == 'zero':
        v = np.zeros(shape)
    elif method == 'glorot_uniform':
        v = glorot_uniform(shape, shape[0], shape[1])
    else: # defaults to zero
        v = np.zero(shape)
        
    return theano.shared(v, name=name)
    
def loss_seq_cross_entropy(correct, predicted):
    """Calculates the negative log likelihood loss between
            two word sequences correct and predicted.
    
    Sequences are assumed to be encoded as 1-of-K encoding (aka one-hot enc.).
    
    Keyword arguments:
    correct -- T.dmatrix (len_seq_t1, K)
    predicted -- T.dmatrix (len_seq_t2, K)
    """
    
    # Trim the longer one so that both seq have the same length
    _correct = correct[:predicted.shape[0]]
    _predicted = predicted[:correct.shape[0]]

    # Calculate loss
    negative_log = T.maximum(-T.log(_predicted + 10e-8), 0.0)
    negative_log_likelihood = _correct * negative_log
    loss = T.sum(negative_log_likelihood)
    
    # If the predicted seq is shorter than the correct one,
    # we assume that it assigned epsilon probability to the remaining part
    # rem_loss = 0.0
    diff = correct.shape[0] - predicted.shape[0]
    diff = diff.clip(0, 10000000)
    k = correct.shape[1]
    rem_loss = diff * (-T.log(1.0/k))
    
    return loss + rem_loss

def pad_vector(a, b):
    dim_diff = b.shape[0] - a.shape[0]
    dim_diff.name = 'dim_diff'
    result = theano.ifelse.ifelse(T.lt(dim_diff, 0),
        (a, T.concatenate([b, T.zeros((-dim_diff,a.shape[1]))])),
        (T.concatenate([a, T.zeros((dim_diff,a.shape[1]))]), b))
    return result
    
def convert_to_one_hot(index_vec, dim):
    """
    Args:
        index_vec (list of int)
        dim (int)
    """    
    
    m = np.zeros((len(index_vec), dim))
    for i in range(len(index_vec)):
        m[i][index_vec[i]] = 1
    return m
    
def convert_to_index(soft_one_hot):
    return np.argmax(soft_one_hot, axis=1)
