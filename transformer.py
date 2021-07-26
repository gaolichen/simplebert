import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy

def clone(obj, N):
    return [copy.deepcopy(obj) for i in range(N)]

class SequenceEmbedding(keras.layers.Layer):
    """ Encode integer sequences into vectors of float type.

    """
    def __init__(self, vacb_size, dim, name = None):
        super(SequenceEmbedding, self).__init__(name = name)

        self.dim = dim
        self.embedding = keras.layers.Embedding(input_dim = vacb_size, output_dim = dim)


    def call(self, x, **kwargs):
        return self.embedding(x) * tf.math.sqrt(tf.cast(self.dim, dtype='float32'))
        

class PositionEncoder(keras.layers.Layer):
    """Encode position information of input.

    PE(pos, 2*i) = sin(pos/10000**(2*i/dim))
    PE(pos, 2*i+1) = cos(pos/10000**(2*i/dim))
    """
    def __init__(self, dim, dropout = 0.1, max_sequence_len = 5000, name = None):
        super(PositionEncoder, self).__init__(name = name)

        self.max_len = max_sequence_len
        self.dim = dim
        self.dropout = keras.layers.Dropout(rate = dropout)

        # initialize positional encoding
        self.pe = tf.Variable(initial_value = tf.constant(0.0, shape = (self.max_len, self.dim)), trainable = False)
        
        exponents = -tf.range(0, self.dim, 2, dtype = 'float32') * tf.math.log(10000.0)
        deno = tf.expand_dims(tf.math.exp(-exponents / self.dim), 0)
        pos = tf.expand_dims(tf.range(0, self.max_len, dtype = 'float32'), -1)
        args = pos * deno
        self.pe[:, 0::2].assign(tf.math.sin(args))
        if self.dim % 2 == 0:
            self.pe[:, 1::2].assign(tf.math.cos(args))
        else:
            self.pe[:, 1::2].assign(tf.math.cos(args[:,:-1]))


    def call(self, x, **kwargs):
        x = tf.add(x, self.pe[:x.shape[-2], :])
        return self.dropout(x)

class LayerNorm(tf.Module):
    def __init__(self, eps = 1e-6, name = None):
        super(LayerNorm, self).__init__(name = name)

        self.eps = tf.constant(eps, shape = (1,))

    def __call__(self, x, **kwargs):
        mean = tf.reduce_mean(x, axis = -1, keepdims = True)
        std = tf.math.reduce_std(x, axis = -1, keepdims = True)
        output = tf.math.subtract(x, mean)
        output = tf.math.divide(output, tf.math.add(std, self.eps))

        return output


def attention(querys, keys, values, mask = None, dropout = None):
    """ Computes attention.
    """
    prod = tf.matmul(querys, keys, transpose_b = True)
    norm_factor = tf.math.sqrt(tf.cast(keys.shape[-1], 'float32'))
    prod = tf.math.divide(prod, norm_factor)

    if mask is not None:
        # prod += (1 - mask) * -1e9
        prod = tf.math.add(prod, tf.math.multiply(tf.math.subtract(1.0, mask), -1e9))

    p_attn = tf.nn.softmax(prod, axis = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
                
    return tf.matmul(p_attn, values), p_attn


class MultiHeadedAttention(tf.Module):
    """ Performs multi-head attention.

        MultiHead(Q, K, V) = Concat(head_1, head_2,..., head_h)W_O,
        where head_i = Attention(Q[i], K[i], V[i]),
        Q[i] = matmul(Q, W_Q[i]), K[i] = matmul(K, W_K[i]), V[i] = matmul(V, W_V[i])
    """
    def __init__(self, h, dim, dropout = 0.1, name = None):
        super(MultiHeadedAttention, self).__init__(name = name)
        assert dim % h == 0

        self.d_k = dim // h
        self.h = h
        self.dim = dim
        
        self.w_q = tf.Variable(tf.random.normal([dim, dim], stddev = 0.02), name='wq', )
        self.w_k = tf.Variable(tf.random.normal([dim, dim], stddev = 0.02), name='wk')
        self.w_v = tf.Variable(tf.random.normal([dim, dim], stddev = 0.02), name='wv')
        self.w_o = tf.Variable(tf.random.normal([dim, dim], stddev = 0.02), name='wo')

        self.dropout = keras.layers.Dropout(rate = dropout)

    def __call__(self, querys, keys, values, mask = None):
        """ Computes multiple attention scores.

            querys.shape = (batches, sequence_len, d_model)
            keys.shape = (batches, sequence_len, d_model)
            values.shape = (batches, sequence_len, d_model)
        """
        batches = keys.shape[0]

        # q_i = Q * W^Q_i
        querys = tf.reshape(tf.matmul(querys, self.w_q), shape = (batches, -1, self.h, self.d_k))
        # At this point, the shapes of querys, keys, and values are (batches, sequence_len, h, d_k)
        # we need to change the shapes to (batches, h, sequence_len, d_k)
        querys = tf.transpose(querys, perm = [0, 2, 1, 3])

        # the same operations on keys and values
        # k_i = K * W^K_i
        keys = tf.reshape(tf.matmul(keys, self.w_k), shape = (batches, -1, self.h, self.d_k))
        keys = tf.transpose(keys, perm = [0, 2, 1, 3])
        
        # v_i = V * W^V_i
        values = tf.reshape(tf.matmul(values, self.w_v), shape = (batches, -1, self.h, self.d_k))
        values = tf.transpose(values, perm = [0, 2, 1, 3])

        v, self.attn = attention(querys, keys, values, mask = mask, dropout = self.dropout)

        # the shape of v is (batches, h, sequence_len, d_k)
        # we need to change it to (batches, sequence_len, h, d_k)
        # and then concatenate last two dims (batches, sequence_len, h * d_k)
        v = tf.reshape(tf.transpose(v, perm = [0, 2, 1, 3]), (batches, -1, self.h * self.d_k))

        # finally apply linear transformation w_o
        return tf.matmul(v, self.w_o)
        

def get_initializer(init_range = 0.02):
    return keras.initializers.TruncatedNormal(stddev=init_range)

class PositionwiseFeedForward(tf.Module):
    def __init__(self, d_out, d_hidden, dropout = 0.1, name = None):
        super(PositionwiseFeedForward, self).__init__(name = name)
        
        self.dense1 = keras.layers.Dense(units = d_hidden, activation = 'relu', name = 'FF_dense1', kernel_initializer = get_initializer())
        self.dropout = keras.layers.Dropout(dropout)
        self.dense2 = keras.layers.Dense(units = d_out, name = 'FF_dense2', kernel_initializer = get_initializer())

    def __call__(self, x):
        output = self.dense1(x)
        output = self.dropout(output)
        return self.dense2(x)

class SublayerConnection(tf.Module):
    """
    
    """
    def __init__(self, dropout_rate):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm()
        self.dropout = keras.layers.Dropout(rate = dropout_rate)

    def __call__(self, x, output, **kwargs):
        return tf.add(x, self.dropout(self.norm(output)))


class EncoderLayer(keras.layers.Layer):
    """
    Base class of transformer layers
    """
    def __init__(self,
                 attn,
                 feed_forward,
                 dropout_rate,
                 name = None):
        super(EncoderLayer, self).__init__(name = name)
        
        self.self_attn = attn
        self.feed_forward = feed_forward
        self.conns = clone(SublayerConnection(dropout_rate = dropout_rate), 2)

    def call(self, x, attention_mask, **kwargs):
        output1 = self.self_attn(x, x, x, mask = attention_mask)
        #print('after attention:', output1)
        output1 = self.conns[0](x, output1)
        output2 = self.feed_forward(output1)
        return self.conns[1](output1, output2)

class Encoder(tf.Module):
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 d_model,
                 intermediate_size,
                 dropout_rate,
                 name = None):
        super(Encoder, self).__init__(name = name)
        
        attn = MultiHeadedAttention(h = num_attention_heads,
                                    dim = d_model,
                                    name = 'self_attention')
        feed_forward = PositionwiseFeedForward(d_out = d_model,
                                               d_hidden = intermediate_size,
                                               name = 'positionwise_feed_forward')
        layer = EncoderLayer(attn = attn,
                             feed_forward = feed_forward,
                             dropout_rate = dropout_rate)
        
        self.main_layers = clone(layer, num_hidden_layers)
        self.norm = LayerNorm()

    def __call__(self, x, attention_mask):
        for layer in self.main_layers:
            x = layer(x, attention_mask)

        #print('before norm: x = ', x)

        #print('after norm: x = ', self.norm(x))
            
        return self.norm(x)

class TransformerHead(keras.layers.Layer):
    def __init__(self, name = None):
        super(TransformerHead, self).__init__(name = name)
        
        self.dense = keras.layers.Dense(units = 1, activation = 'relu')

    def call(self, x):
        return self.dense(x)

class Transformer(keras.models.Model):
    """
    base class of transformers
    """
    def __init__(self,
                 num_hidden_layers,
                 num_attention_heads,
                 hidden_size,
                 max_position_embeddings,
                 intermediate_size,
                 vacb_size,
                 causal_attention = False,
                 **kwargs
                 ):
        super(Transformer, self).__init__(**kwargs)
        
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.input_embedding = SequenceEmbedding(vacb_size = vacb_size, dim = hidden_size)
        self.position_encoder = PositionEncoder(dim = hidden_size, max_sequence_len = max_position_embeddings)
        self.encoder = Encoder(num_hidden_layers = num_hidden_layers,
                               num_attention_heads = num_attention_heads,
                               d_model = hidden_size,
                               intermediate_size = intermediate_size,
                               dropout_rate = 0.1,
                               name = 'encoder')
                               
                               
                               
        self.output_layer = TransformerHead()
        self.causal_attention = causal_attention
        
    
    def build(self, input_shape, **kwargs):
        pass

    def call(self, input_ids,
             embedding_ids = None,
             attention_mask = None):
        output = self.input_embedding(input_ids)
        output = self.position_encoder(output)

        if attention_mask is None:
            attention_mask = tf.contant(1.0, shape = input_ids.shape, dtype = 'float32')

        # attention_mask now has shape (batches, sequence_len),
        # we need to covert it to (batches, 1, 1, sequence_len)
        # so that later it will broadcast to (batches, num_attention_heads, sequence_len, sequence_len)
        attention_mask = tf.reshape(attention_mask, shape = (input_ids.shape[0], 1, 1, input_ids.shape[-1]))
        print('output=', output)
        #print('attention_mask=', attention_mask)
            
        output = self.encoder(output, attention_mask)
            
        return output


def test_PositionEncoder():
    pe = PositionEncoder(dim = 3, dropout = 0.5, max_sequence_len = 100)
    t = tf.constant([[1.2, 2.0,0.4],[3.0,4.6,-2]])
    print('t=', t)
    print('pe(t)=', pe(t, training = False))

    print('pe(t)=', pe(t, training = True))


def test_SequenceEmbedding():
    se = SequenceEmbedding(vacb_size = 100, dim = 10)
    print(se(tf.constant([0,1,2,4])))

def test_Transformer():
    tm = Transformer(num_hidden_layers = 12,
                     num_attention_heads = 16,
                     hidden_size = 768,
                     max_position_embeddings = 512,
                     vacb_size = 100,
                     intermediate_size = 1024)
    input_ids = tf.constant([[4,1,2,3,0],[4,1,2,3,0]])
    attention_mask = tf.cast(tf.math.not_equal(input_ids, 0), dtype = 'float32')
    output = tm(input_ids, attention_mask = attention_mask)
    print(output)

def test_LayerNorm():
    x = tf.random.uniform(shape = [3, 5], minval = 0, maxval = 5)
    norm = LayerNorm()
    print('x=', x)
    print('norm(x)=', norm(x))

if __name__ == '__main__':
    test_Transformer()
