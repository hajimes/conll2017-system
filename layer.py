import theano.tensor as T
import theano.tensor.nnet as nnet
from util import init_var
        
class Layer(object):
    def add_gru(self, hidden_dim, embedding_dim, prefix=''):
        W_gru_r = init_var('glorot_uniform',
            (hidden_dim, embedding_dim), prefix + '_W_gru_r')
    
        U_gru_r = init_var('glorot_uniform',
            (hidden_dim, hidden_dim), prefix + '_U_gru_r')
        b_gru_r = init_var('zero', (hidden_dim,), prefix + '_b_gru_r')

        W_gru_z = init_var('glorot_uniform',
            (hidden_dim, embedding_dim), prefix + '_W_gru_z')    
        U_gru_z = init_var('glorot_uniform',
            (hidden_dim, hidden_dim), prefix + '_U_gru_z')
        b_gru_z = init_var('zero', (hidden_dim,), prefix + '_b_gru_z')

        W_gru = init_var('glorot_uniform',
            (hidden_dim, embedding_dim), prefix + '_W_gru')    
        U_gru = init_var('glorot_uniform',
            (hidden_dim, hidden_dim), prefix + '_U_gru')
        b_gru = init_var('zero', (hidden_dim,), prefix + '_b_gru')
        
        self.params += [
            W_gru_r, U_gru_r, b_gru_r,
            W_gru_z, U_gru_z, b_gru_z,
            W_gru, U_gru, b_gru
        ]

    def step_gru(emb, h_previous,
            W_gru_r, U_gru_r, b_gru_r,
            W_gru_z, U_gru_z, b_gru_z,
            W_gru, U_gru, b_gru,
            c_r, c_z, c, # use T.zeros_like(b_gru) if these args are not needed
            activation
        ):
        """This class method computes a tensor for GRU
        """
        r = nnet.sigmoid(T.dot(W_gru_r, emb) +
            T.dot(U_gru_r, h_previous) + c_r + b_gru_r)
    
        # GRU: update gate
        z = nnet.sigmoid(T.dot(W_gru_z, emb) +
            T.dot(U_gru_z, h_previous) + c_z + b_gru_z)
    
        # GRU: temp variable

        h_bar = activation(T.dot(W_gru, emb) +
            T.dot(U_gru, r * h_previous) + c + b_gru)

        # GRU: hidden units
        h = z * h_bar + (1 - z) * h_previous
        
        return h