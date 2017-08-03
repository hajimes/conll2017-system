import math

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

from layer import Layer
from update import update_sgd, update_adam, update_adamax, update_adam_l1, update_adamax_l1
from util import get_shared_zeros, glorot_uniform, init_var, loss_seq_cross_entropy, pad_vector

NEGATIVE_LOG_EPSILON = -math.log(10e-2)

def get_attention_context_vector(W_a, h_previous, annotations, U_a, v_a):
    # Compute W_{a} s_{i-1} in Bahdanau2015
    # (alignment_dim,)
    Wa_hprev = T.dot(W_a, h_previous)
    # (input_length, alignment_dim)
    Wa_hprev_set = T.tile(Wa_hprev, (annotations.shape[0], 1))

    # (input_length, alignment_dim)
    # U_{a} h_{j} in Bahdanau2015
    # annotations is a (input_length, hidden_dim)-sized matrix
    # U_a is a (hidden_dim, alignment_dim)-sized matrix.
    # Note that this def. of U_a is the transposed ver. of Bahdanau2015
    U_h = T.dot(annotations, U_a)

    # (input_length,):
    # Alignment vector for the i-th step
    # j-th element corresponds to "e_{ij}" in Bahdanau2015
    # Wa_sprev and U_h are (input_length, alignment_dim) matrices and
    # v_a is a (alignment_dim,)-sized vector
    # so the following dot produces an (input_length,)-sized vector
    alignments_tmp = T.tanh(Wa_hprev_set + U_h)
    alignments = T.dot(alignments_tmp, v_a)
    
    # (input_length,):
    # Attention vector for the i-th step
    # j-th element corresponds to "alpha_{ij}" in Bahdanau2015.
    # (brackets are required since nnet.softmax takes 2d and outputs 2d)
    attentions = nnet.softmax([alignments])[0]

    # (hidden_dim,)
    # Context_vector for the i-th step
    # attentions is a (input_length,)-sized vector while
    # annotations is a (input_length, hidden_dim)-sized matrix
    c = T.dot(annotations.T, attentions)

    return c

class Encoder(Layer):
    def __init__(self, input_dim, embedding_dim, hidden_dim,
            with_bidirectional=False):
        ###
        # Tensors for input embedding
        ###
        W_in = init_var('glorot_uniform',
            (embedding_dim, input_dim), 'enc_W_in')
        b_in = init_var('zero', (embedding_dim,), 'enc_b_in')

        self.params = [W_in, b_in]
        
        total_embedding_dim = embedding_dim
        
        ###
        # Tensors for encoder GRU
        ###
        self.add_gru(hidden_dim, total_embedding_dim, prefix='enc')
            
        if (with_bidirectional):
            self.add_gru(hidden_dim, total_embedding_dim, prefix='enc_reversed')

    def create_step_bidi(activation):
        def step_bidi(in_vec, in_vec_reverse, # sequences
                out_previous, h_previous, h_previous_rev, # outputs_info
                W_in, b_in, # non_sequences
                W_gru_r, U_gru_r, b_gru_r, # non_sequences
                W_gru_z, U_gru_z, b_gru_z, # non_sequences
                W_gru, U_gru, b_gru, # non_sequences
                W_gru_r_rev, U_gru_r_rev, b_gru_r_rev, # non_sequences
                W_gru_z_rev, U_gru_z_rev, b_gru_z_rev, # non_sequences
                W_gru_rev, U_gru_rev, b_gru_rev # non_sequences
            ):

            ###
            # Input word embedding
            ###
            emb = T.dot(W_in, in_vec) + b_in
            emb_rev = T.dot(W_in, in_vec_reverse) + b_in
        
            ###
            # GRU computation
            ###
            # GRU: reset gate
            h = Layer.step_gru(emb, h_previous,
                W_gru_r, U_gru_r, b_gru_r,
                W_gru_z, U_gru_z, b_gru_z,
                W_gru, U_gru, b_gru,
                T.zeros_like(b_gru), T.zeros_like(b_gru), T.zeros_like(b_gru),
                activation)

            ###
            # GRU computation for reversed direction
            ###
            h_rev = Layer.step_gru(emb_rev, h_previous_rev,
                W_gru_r_rev, U_gru_r_rev, b_gru_r_rev,
                W_gru_z_rev, U_gru_z_rev, b_gru_z_rev,
                W_gru_rev, U_gru_rev, b_gru_rev,
                T.zeros_like(b_gru), T.zeros_like(b_gru), T.zeros_like(b_gru),
                activation)

            ###
            # Out
            ###
            h_total = T.concatenate([h, h_rev])
            out = activation(h_total)

            return [out, h, h_rev]
        return step_bidi

class Decoder(Layer):
    def __init__(self, output_dim, embedding_dim, hidden_dim,
            with_attention=False, alignment_dim=1000):
        ###
        # Tensors for embedding of previous outputs
        ###
        W_in = init_var('glorot_uniform',
            (embedding_dim, output_dim), 'dec_W_in')
        b_in = init_var('zero', (embedding_dim,), 'dec_b_in')

        ###
        # Tensors for context vector
        ###
        C = init_var('glorot_uniform', (hidden_dim, hidden_dim), 'C')
        C_r = init_var('glorot_uniform', (hidden_dim, hidden_dim), 'C_r')
        C_z = init_var('glorot_uniform', (hidden_dim, hidden_dim), 'C_z')

        self.params = [W_in, b_in,
            C, C_r, C_z]

        ###
        # Tensors for encoder GRU
        ###
        self.add_gru(hidden_dim, embedding_dim, 'dec')
    
        ###
        # Tensors for output
        ###
        W_out = init_var('glorot_uniform', (output_dim, hidden_dim), 'W_out')
        b_out = init_var('zero', (output_dim,), 'b_out')    
    
        self.params += [W_out, b_out]
                
        ###
        # Tensors for attention
        ###
        if (with_attention):
            v_a = init_var('zero', (alignment_dim,), 'v_a')
            W_a = init_var('glorot_uniform', (alignment_dim, hidden_dim), 'W_a')
            U_a = init_var('glorot_uniform', (hidden_dim, alignment_dim), 'U_a')
            self.params += [v_a, W_a, U_a]

    def create_step_attention(activation):
        def _step_attention(out_previous, h_previous, # outputs_info
                annotations, # non_sequences
                W_in, b_in, # non_sequences
                C, C_r, C_z, # non_sequences
                W_gru_r, U_gru_r, b_gru_r, # non_sequences
                W_gru_z, U_gru_z, b_gru_z, # non_sequences
                W_gru, U_gru, b_gru, # non_sequences
                W_out, b_out, # non_sequences
                v_a, W_a, U_a # non_sequences
            ):
            ###
            # Word embedding for the previous output
            ###
            emb = T.dot(W_in, out_previous) + b_in

            ###
            # Attention computation
            ###
            c = get_attention_context_vector(W_a, h_previous, annotations, U_a, v_a)

            ###
            # GRU computation
            ###
            c_r = T.dot(C_r, c)
            c_z = T.dot(C_z, c)
            c = T.dot(C, c)
            h = Layer.step_gru(emb, h_previous,
                       W_gru_r, U_gru_r, b_gru_r,
                       W_gru_z, U_gru_z, b_gru_z,
                       W_gru, U_gru, b_gru,
                       c_r, c_z, c,
                       activation)

            ###
            # Out
            ###
            # brackets are required since nnet.softmax takes 2d and outputs 2d
            out = nnet.softmax([T.dot(W_out, activation(h)) + b_out])[0]

            # NB: The END symbol is hard-coded (1); change this one in the future
            return [out, h], theano.scan_module.until(T.eq(out.argmax(), 1))
        return _step_attention