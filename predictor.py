import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

import pickle

import network

class Predictor(object):
    def __init__(self, dim_input, dim_output,
            embedding_dim, hidden_dim, alignment_dim,
            optimization_method = 'adamax', activation_method = 'tanh'):
        """ Create a neural network to train a new model
    
        Args:
            dim_input (int)
            dim_output (int)
            embedding_dim (int): Bigger is better
                300 is a reasonable choice, according to Kann and Schütze (2016)
            hidden_dim (int): 100 achieved the best (among 50, 100, 200, 400)
                in Kann and Schütze (2016)
            alignment_dim (int)
        """

        self._dim_input = dim_input
        self._dim_output = dim_output
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._alignment_dim = alignment_dim

        if optimization_method == 'adamax':
            self._optimization_method = network.update_adamax
        elif optimization_method == 'adam':
            self._optimization_method = network.update_adam
        elif optimization_method == 'adam_l1':
            self._optimization_method = network.update_adam_l1
        elif optimization_method == 'adamax_l1':
            self._optimization_method = network.update_adamax_l1
        else:
            raise ValueError('invalid optimization method')

        if activation_method == 'tanh':
            self._activation_method = T.tanh
        elif activation_method == 'relu':
            self._activation_method = nnet.relu
        else:
            raise ValueError('invalid activation method')

        self._update, self._predict = _create_model(dim_input,
            dim_output, embedding_dim, hidden_dim, alignment_dim,
            self._optimization_method,
            self._activation_method)

        self._out_init_zeros = np.zeros((self._dim_output,))
        self._context_init_zeros = np.zeros((self._hidden_dim * 2,))
    
    def train(self, in_v, out_v):
        """ Train a model with online upddating
        
        Args:
            in_v: one-hot encoded input vector
            out_v: one-hot encoded output vector
        
        Returns:
            int: loss incurred by this update
        """        
        current_loss = self._update(in_v, out_v,
            self._context_init_zeros, self._out_init_zeros)

        return current_loss

    def predict(self, in_v):
        return self._predict(in_v,
            self._context_init_zeros, self._out_init_zeros)

def _create_model(dim_input, dim_output,
        embedding_dim, hidden_dim, alignment_dim,
        optimization_method, activation):
    """ Create a neural network to train a new model
    
    Args:
        dim_input (int)
        dim_output (int)
        embedding_dim (int): Bigger is better
            300 is a reasonable choice, according to Kann and Schütze (2016)
        hidden_dim (int): 100 achieved the best (among 50, 100, 200, 400)
            in Kann and Schütze (2016)
        alignment_dim (int)
    """
    
    input_seq = T.dmatrix('input_seq')
    output_seq = T.dmatrix('output_seq')

    h_init = theano.shared(np.zeros(hidden_dim))
    h_init_rev = theano.shared(np.zeros(hidden_dim))
    out_init = T.dvector('out_init')
    context_init = T.dvector('context_init')

    encoder = network.Encoder(dim_input, embedding_dim, hidden_dim,
        with_bidirectional=True)
    decoder = network.Decoder(dim_output, embedding_dim, hidden_dim * 2,
        with_attention=True, alignment_dim=alignment_dim)

    params = encoder.params + decoder.params
    
    ###
    # Loop for encoder
    ###
    prediction, _ = theano.scan(fn=network.Encoder.create_step_bidi(activation),
        sequences=[input_seq, input_seq[::-1]],
        outputs_info=[context_init, h_init, h_init_rev],
        non_sequences=encoder.params,
        strict=True)

    context_vec = prediction[0][-1]
    
    ###
    # Loop for decoder
    ###
    annotations = prediction[0]
    prediction, _ = theano.scan(fn=network.Decoder.create_step_attention(activation),
        outputs_info=[out_init, context_vec],
        non_sequences=[annotations] + decoder.params,
        n_steps=128,
        strict=True)

    predicted_seq = prediction[0]
    
    ###
    # Compute loss tensor
    ###
    loss = network.loss_seq_cross_entropy(output_seq, predicted_seq)
    loss.name = 'loss'
    
    ###
    # Update weights
    ###
    updating = optimization_method(loss, params)
    
    current_loss = 0.0
    
    ###
    # Realize the function
    ###
    _update = theano.function(inputs=[input_seq, output_seq,
                context_init, out_init],
                outputs=loss,
                updates=updating)

    _predict = theano.function(inputs=[input_seq, context_init, out_init],
                outputs=predicted_seq)
    
    out_init_zeros = np.zeros((dim_output,))
    context_init_zeros = np.zeros((hidden_dim * 2,))
    h_init_zeros = np.zeros((hidden_dim,))

    return (_update, _predict)