import numpy as np
import theano
import theano.tensor as T

from util import get_shared_shape, get_shared_zeros

def update_sgd(loss, params, learning_rate):
    updates = []
    grads = T.grad(loss, params)

    for (param, grad) in zip(params, grads):
        new_param = param - learning_rate * grad
        updates.append((param, new_param))

    return updates 

def update_adam(loss, params):
    """ Update by Adam, cf. Kingma and Ba (2015)
    See also Keras code
    """
    ###
    # Hyperparamers
    ###
    alpha = 0.001   # (effective upper bound of) stepsize (learning rate)
    beta_1 = 0.9    # controlling decay rates
    beta_2 = 0.999  # controlling decay rates
    epsilon = 1e-8  # required because T.sqrt(v) can be 0

    ###
    # Initialize shared variables
    ###
    timestep = theano.shared(1.0)
    shapes = [get_shared_shape(p) for p in params]
    ms = [get_shared_zeros(shape) for shape in shapes]
    vs = [get_shared_zeros(shape) for shape in shapes]

    ###
    # Main
    ###
    updates = []
    grads = T.grad(loss, params)
    coef = alpha * T.sqrt(1.0 - beta_2 ** timestep) / (1.0 - beta_1 ** timestep)
    for (param_previous, grad, m_previous, v_previous) in zip(params,
          grads, ms, vs):
        # Update biased first moment estimate
        m = beta_1 * m_previous + (1.0 - beta_1) * grad
        # Update biased second raw moment estimate
        v = beta_2 * v_previous + (1.0 - beta_2) * T.sqr(grad)

        # Update parameters with efficient bias correction
        param = param_previous - coef * m / (T.sqrt(v) + epsilon)
        
        updates.append((param_previous, param))
        updates.append((m_previous, m))
        updates.append((v_previous, v))

    updates.append((timestep, timestep + 1.0))

    return updates
    

def update_adamax(loss, params):
    """ Update by AdaMax (Adam + infinity norm reg.), cf. Kingma and Ba (2015)
    See also Keras code
    """
    ###
    # Hyperparamers
    ###
    alpha = 0.002   # (effective upper bound of) stepsize (learning rate)
    beta_1 = 0.9    # controlling decay rates
    beta_2 = 0.999  # controlling decay rates
    epsilon = 1e-8  # required because T.sqrt(v) can be 0

    ###
    # Initialize shared variables
    ###
    timestep = theano.shared(1.0)
    shapes = [get_shared_shape(p) for p in params]
    ms = [get_shared_zeros(shape) for shape in shapes]
    us = [get_shared_zeros(shape) for shape in shapes]

    ###
    # Main
    ###
    updates = []
    grads = T.grad(loss, params)
    coef = alpha / (1.0 - beta_1 ** timestep)
    for (param_previous, grad, m_previous, u_previous) in zip(params,
          grads, ms, us):
        # Update biased first moment estimate
        m = beta_1 * m_previous + (1.0 - beta_1) * grad
        # Update biased second raw moment estimate
        u = T.maximum(beta_2 * u_previous, T.abs_(grad))

        # Update parameters with efficient bias correction
        param = param_previous - coef * (m / (u + epsilon))
        
        updates.append((param_previous, param))
        updates.append((m_previous, m))
        updates.append((u_previous, u))

    updates.append((timestep, timestep + 1.0))

    return updates


def update_adam_l1(loss, params):
    """ Update by Adam, cf. Kingma and Ba (2015)
    See also Keras code
    """
    ###
    # Hyperparamers
    ###
    alpha = 0.001   # (effective upper bound of) stepsize (learning rate)
    beta_1 = 0.9    # controlling decay rates
    beta_2 = 0.999  # controlling decay rates
    epsilon = 1e-8  # required because T.sqrt(v) can be 0
    l1_coeff = 0.001 # from Glorot et al. (2011), p. 320

    ###
    # Initialize shared variables
    ###
    timestep = theano.shared(1.0)
    shapes = [get_shared_shape(p) for p in params]
    ms = [get_shared_zeros(shape) for shape in shapes]
    vs = [get_shared_zeros(shape) for shape in shapes]

    ###
    # Main
    ###
    updates = []
    l1_loss = 0.0
    for param in params:
        l1_loss += T.sum(T.abs_(param))
    grads = T.grad(loss + l1_coeff * l1_loss, params)
    coef = alpha * T.sqrt(1.0 - beta_2 ** timestep) / (1.0 - beta_1 ** timestep)
    for (param_previous, grad, m_previous, v_previous) in zip(params,
          grads, ms, vs):
        # Update biased first moment estimate
        m = beta_1 * m_previous + (1.0 - beta_1) * grad
        # Update biased second raw moment estimate
        v = beta_2 * v_previous + (1.0 - beta_2) * T.sqr(grad)

        # Update parameters with efficient bias correction
        param = param_previous - coef * m / (T.sqrt(v) + epsilon)
        
        updates.append((param_previous, param))
        updates.append((m_previous, m))
        updates.append((v_previous, v))

    updates.append((timestep, timestep + 1.0))

    return updates

def update_adamax_l1(loss, params):
    """ Update by AdaMax (Adam + infinity norm reg.), cf. Kingma and Ba (2015)
    with L1 loss
    See also Keras code
    """
    ###
    # Hyperparamers
    ###
    alpha = 0.002   # (effective upper bound of) stepsize (learning rate)
    beta_1 = 0.9    # controlling decay rates
    beta_2 = 0.999  # controlling decay rates
    epsilon = 1e-8  # required because T.sqrt(v) can be 0
    l1_coeff = 0.001 # from Glorot et al. (2011), p. 320
    # l1_coeff = 0.00001

    ###
    # Initialize shared variables
    ###
    timestep = theano.shared(1.0)
    shapes = [get_shared_shape(p) for p in params]
    ms = [get_shared_zeros(shape) for shape in shapes]
    us = [get_shared_zeros(shape) for shape in shapes]

    ###
    # Main
    ###
    updates = []
    l1_loss = 0.0
    for param in params:
        l1_loss += T.sum(T.abs_(param))
    grads = T.grad(loss + l1_coeff * l1_loss, params)
    coef = alpha / (1.0 - beta_1 ** timestep)
    for (param_previous, grad, m_previous, u_previous) in zip(params,
          grads, ms, us):
        # Update biased first moment estimate
        m = beta_1 * m_previous + (1.0 - beta_1) * grad
        # Update biased second raw moment estimate
        u = T.maximum(beta_2 * u_previous, T.abs_(grad))

        # Update parameters with efficient bias correction
        param = param_previous - coef * (m / (u + epsilon))
        
        updates.append((param_previous, param))
        updates.append((m_previous, m))
        updates.append((u_previous, u))

    updates.append((timestep, timestep + 1.0))

    return updates
