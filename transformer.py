#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import copy
import json
import h5py
from tensorflow.python.keras.saving import hdf5_format

def get_initializer(init_range):
    return keras.initializers.RandomNormal(mean = 0.0, stddev = init_range)

class CheckpointCache(object):
    def __init__(self, ckp_or_h5_path):
        super(CheckpointCache, self).__init__()

        self.ckp_or_h5_path = ckp_or_h5_path

        if ckp_or_h5_path.endswith('.h5'):
            self.shape_from_key = {}
            self.dtype_from_key = {}
            self.values_from_key = {}
            with h5py.File(ckp_or_h5_path, 'r') as f:
                layers_name = set(hdf5_format.load_attributes_from_hdf5_group(f, "layer_names"))
                for layer_name in layers_name:
                    layer_object = f[layer_name]
                    for weight_name in hdf5_format.load_attributes_from_hdf5_group(layer_object, "weight_names"):
                        key = '/'.join(weight_name.split('/')[1:])
                        #print(key)
                        weights = np.asarray(layer_object[weight_name])
                        self.shape_from_key[key] = weights.shape
                        self.dtype_from_key[key] = weights.dtype
                        self.values_from_key[key] = weights
        else:
            reader = tf.train.load_checkpoint(ckp_or_h5_path)
            self.shape_from_key = reader.get_variable_to_shape_map()
            self.dtype_from_key = reader.get_variable_to_dtype_map()

    def keys(self):
        return list(self.shape_from_key.keys())

    def get_shape(self, key):
        return self.shape_from_key[key]

    def get_dtype(self, key):
        return self.dtype_from_key[key]

    def get_values(self, key):
        if self.ckp_or_h5_path.endswith('.h5'):
            return self.values_from_key[key]
        else:
            return tf.train.load_variable(self.ckp_or_h5_path, key)

class BertConfig(object):
    
    def __init__(self, path, **kwargs):
        with open(path) as f:
            self._config = json.load(f)

        for key, value in kwargs.items():
            self._config[key] = value

        self.attention_probs_dropout_prob = self.get('attention_probs_dropout_prob', 0.1)
        self.hidden_act = self.get('hidden_act', 'gelu')
        self.hidden_dropout_prob = self.get('hidden_dropout_prob', 0.1)
        self.hidden_size = self.get('hidden_size', 768)
        self.initializer_range = self.get('initializer_range', 0.02)
        self.intermediate_size = self.get('intermediate_size', 3072)
        self.max_position_embeddings = self.get('max_position_embeddings', 512)
        self.num_attention_heads = self.get('num_attention_heads', 12)
        self.num_hidden_layers = self.get('num_hidden_layers', 12)
        self.type_vocab_size = self.get('type_vocab_size', 2)
        self.vocab_size = self.get('vocab_size', None)

    def get(self, key, default_value = None):
        if default_value is None:
            return self._config[key]
        else:
            return self._config.get(key, default_value)


class SequenceEmbedding(keras.layers.Layer):
    """ Encode integer sequences into vectors of float type.

    """
    def __init__(self, vocab_size, dim, name = None):
        super(SequenceEmbedding, self).__init__(name = name)

        self.dim = dim
        self.embedding = keras.layers.Embedding(input_dim = vocab_size, output_dim = dim, name = 'embedding')


    def call(self, x):
        return self.embedding(x) #* tf.math.sqrt(tf.cast(self.dim, dtype='float32'))

class PositionEmbedding(keras.layers.Layer):
    """Encode position information of input.

    PE(pos, 2*i) = sin(pos/10000**(2*i/dim))
    PE(pos, 2*i+1) = cos(pos/10000**(2*i/dim))
    """
    def __init__(self, dim, max_position_embeddings = 5000, pad_id = 0, name = None):
        super(PositionEmbedding, self).__init__(name = name)

        self.max_len = max_position_embeddings
        self.dim = dim
        self.pad_id = pad_id

    def build(self, input_shape):
        # initialize positional encoding
        self.embeddings = self.add_weight(shape = (self.max_len, self.dim), trainable = False,
                                          initializer = keras.initializers.Constant(), name = 'embeddings')
        
        exponents = -tf.range(0, self.dim, 2, dtype = 'float32') * tf.math.log(10000.0)
        deno = tf.expand_dims(tf.math.exp(-exponents / self.dim), 0)
        pos = tf.expand_dims(tf.range(0, self.max_len, dtype = 'float32'), -1)
        args = pos * deno
        self.embeddings[:, 0::2].assign(tf.math.sin(args))
        if self.dim % 2 == 0:
            self.embeddings[:, 1::2].assign(tf.math.cos(args))
        else:
            self.embeddings[:, 1::2].assign(tf.math.cos(args[:,:-1]))


    def call(self, input_ids):
        #has_id = tf.cast(tf.math.not_equal(input_ids, self.pad_id), dtype = input_ids.dtype)
        #position_ids = tf.math.multiply(tf.cumsum(has_id, axis = 1), has_id) + self.pad_id
        position_ids = tf.expand_dims(tf.range(start = 0, limit = input_ids.shape[-1]), axis = 0)
        output = tf.gather(self.embeddings, indices = position_ids)
        return tf.tile(input = output, multiples = (input_ids.shape[0], 1, 1)) 

class TransformerEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, dim, max_position_embeddings, name = None):
        super(TransformerEmbedding, self).__init__(name = name)

        with tf.name_scope('embeddings') as scope:
            self.word_embeddings = SequenceEmbedding(vocab_size = vocab_size, dim = dim, name = 'word_embeddings')
            self.position_embeddings = PositionEmbedding(dim = dim,
                                                       max_position_embeddings = max_position_embeddings,
                                                       name = "position_embeddings")
            self.token_type_embeddings = keras.layers.Embedding(input_dim = 2, output_dim = dim, name = "token_type_embeddings")
            self.sum_layer = keras.layers.Add(name = 'add')
            self.norm = keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
            self.dropout = keras.layers.Dropout(rate=0.1)

    def call(self, input_ids, token_type_ids = None, training = False):
        x1 = self.word_embeddings(input_ids)
        x2 = self.position_embeddings(input_ids)
        #x2 = tf.tile(input=x2, multiples=(input_ids.shape[0], 1, 1))

        if token_type_ids is None:
            token_type_ids = tf.constant(0, shape = input_ids.shape)
        x3 = self.token_type_embeddings(token_type_ids)

        output = self.sum_layer([x1, x2, x3])
        output = self.norm(output, training = training)
        return self.dropout(output, training = training)

    def save_weight(self, file_path):
        symbolic_weights = self.trainable_weights + self.non_trainable_weights
        with open(file_path, 'w') as f:
            objs = []
            for weight in symbolic_weights:
                print(weight.name)
                objs.append({"name": weight.name, "value":weight.numpy().tolist()[0]})
            f.write(json.dumps(objs))


class SublayerConnection(tf.keras.layers.Layer):
    """    
    """
    def __init__(self, d_out, dropout_rate, **kwargs):
        super(SublayerConnection, self).__init__(**kwargs)
        
        self.dense = keras.layers.Dense(units = d_out, kernel_initializer = get_initializer(0.02), name = 'dense')
        self.norm = keras.layers.LayerNormalization(name = 'LayerNorm')
        self.dropout = keras.layers.Dropout(rate = dropout_rate)

    def call(self, x, output, **kwargs):
        #output = self.dense(output)
        #return tf.add(x, self.dropout(self.norm(output)))
        output = self.dropout(self.dense(output))
        return self.norm(x + output)



def attention(querys, keys, values, mask = None, dropout = None, training = False):
    """ Computes attention.
    """
    prod = tf.matmul(querys, keys, transpose_b = True)
    norm_factor = tf.math.sqrt(tf.cast(keys.shape[-1], 'float32'))
    prod = tf.math.divide(prod, norm_factor)

    if mask is not None:
        # prod += (1 - mask) * -1e9
        prod = tf.math.add(prod, tf.math.multiply(tf.math.subtract(1.0, tf.cast(mask, 'float32')), -1e4))

    p_attn = tf.nn.softmax(prod, axis = -1)

    if dropout is not None:
        p_attn = dropout(p_attn, training = training)
                
    return tf.matmul(p_attn, values), p_attn


class MultiHeadedAttention(keras.layers.Layer):
    """ Performs multi-head attention.

        MultiHead(Q, K, V) = Concat(head_1, head_2,..., head_h)W_O,
        where head_i = Attention(Q[i], K[i], V[i]),
        Q[i] = matmul(Q, W_Q[i]), K[i] = matmul(K, W_K[i]), V[i] = matmul(V, W_V[i])
    """
    def __init__(self, num_attention_heads, hidden_size, dropout = 0.1, name = None):
        super(MultiHeadedAttention, self).__init__(name = name)
        assert hidden_size % num_attention_heads == 0

        self.d_k = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        
        self.dropout = keras.layers.Dropout(rate = dropout)
        
        self.w_q = keras.layers.Dense(units = self.hidden_size,
                                      kernel_initializer = get_initializer(0.02),
                                      name='query')
        self.w_k = keras.layers.Dense(units = self.hidden_size,
                                      kernel_initializer = get_initializer(0.02),
                                      name='key')
        self.w_v = keras.layers.Dense(units = self.hidden_size,
                                      kernel_initializer = get_initializer(0.02),
                                      name='value')
        #self.w_o = keras.layers.Dense(units = self.dim, kernel_initializer = get_initializer(0.02), name='output')

        
    def call(self, querys, keys, values, mask = None, training = False):
        """ Computes multiple attention scores.

            querys.shape = (batches, sequence_len, d_model)
            keys.shape = (batches, sequence_len, d_model)
            values.shape = (batches, sequence_len, d_model)
        """
        batches = keys.shape[0]

        # q_i = Q * W^Q_i
        querys = self.w_q(querys)
        querys = tf.reshape(querys, shape = (batches, -1, self.num_attention_heads, self.d_k))
        # At this point, the shapes of querys, keys, and values are (batches, sequence_len, num_attention_heads, d_k)
        # we need to change the shapes to (batches, num_attention_heads, sequence_len, d_k)
        querys = tf.transpose(querys, perm = [0, 2, 1, 3])

        # the same operations on keys and values
        # k_i = K * W^K_i
        keys = self.w_k(keys)
        keys = tf.reshape(keys, shape = (batches, -1, self.num_attention_heads, self.d_k))
        keys = tf.transpose(keys, perm = [0, 2, 1, 3])
        
        # v_i = V * W^V_i
        values = self.w_v(values)
        values = tf.reshape(values, shape = (batches, -1, self.num_attention_heads, self.d_k))
        values = tf.transpose(values, perm = [0, 2, 1, 3])

        v, self.attn = attention(querys, keys, values, mask = mask, dropout = self.dropout)

        # the shape of v is (batches, h, sequence_len, d_k)
        # we need to change it to (batches, sequence_len, num_attention_heads, d_k)
        # and then concatenate last two dims (batches, sequence_len, num_attention_heads * d_k)
        return tf.reshape(tf.transpose(v, perm = [0, 2, 1, 3]), (batches, -1, self.num_attention_heads * self.d_k))


class AttentionLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

        self.self_attn = MultiHeadedAttention(num_attention_heads = config.num_attention_heads,
                                              hidden_size = config.hidden_size,
                                              dropout = config.attention_probs_dropout_prob,
                                              name = 'self')
        self.conn = SublayerConnection(d_out = config.hidden_size, dropout_rate = config.hidden_dropout_prob, name = 'output')

    def call(self, x, attention_mask, training = False):
        output1 = self.self_attn(x, x, x, mask = attention_mask, training = training)
        return self.conn(x, output1, training = training)
        

class PositionwiseFeedForward(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        
        self.dense = keras.layers.Dense(units = config.intermediate_size,
                                        activation = config.hidden_act,
                                        kernel_initializer = get_initializer(config.initializer_range),
                                        name = 'dense')
        
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, x, training = False):
        output = self.dense(x)
        return self.dropout(output, training = training)

class FeedForwardLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(FeedForwardLayer, self).__init__(**kwargs)

        self.feed_forward = PositionwiseFeedForward(config, name = 'intermediate')
        self.conn = SublayerConnection(d_out = config.hidden_size,
                                       dropout_rate = config.hidden_dropout_prob,
                                       name = 'output')

    def call(self, x, training = False):
        output = self.feed_forward(x, training = training)
        return self.conn(x, output, training = training)


class EncoderLayer(keras.layers.Layer):
    """
    Base class of transformer layers
    """
    def __init__(self, config, name = None):
        super(EncoderLayer, self).__init__(name = name)
        
        self.self_attn = AttentionLayer(config, name = 'attention')
        self.feed_forward = FeedForwardLayer(config, name = 'feedforward')
        
    def call(self, x, attention_mask, training = False, **kwargs):
        output = self.self_attn(x, attention_mask, training = training)
        return self.feed_forward(output, training = training)

class Encoder(keras.layers.Layer):
    def __init__(self, config, name = None):
        super(Encoder, self).__init__(name = name)

        self.config = config        
        self.main_layers = []
        
        for i in range(config.num_hidden_layers):
            layer = EncoderLayer(config = config, name = f'layer_{i}')                        
            self.main_layers.append(layer)

    def call(self, x, attention_mask, output_hidden_states = False, training = False):
        hidden_states = []
        for layer in self.main_layers:
            x = layer(x, attention_mask, training = training)
            if output_hidden_states:
                hidden_states.append(x)
            
        return x, hidden_states

class TransformerPooler(keras.layers.Layer):
    def __init__(self, config, name = None):
        super(TransformerPooler, self).__init__(name = name)
        
        self.dense = keras.layers.Dense(units = config.hidden_size,
                                        kernel_initializer=get_initializer(config.initializer_range),
                                        activation = 'tanh', name = 'dense')

    def call(self, hidden_states):
        return self.dense(hidden_states[:,0])    

class Transformer(keras.models.Model):
    """
    base class of transformers
    """
    def __init__(self, config, add_pooler = False, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.config = config
        self.embeddings = TransformerEmbedding(vocab_size = config.vocab_size,
                                               dim = config.hidden_size,
                                               max_position_embeddings = config.max_position_embeddings,
                                               name = 'embeddings')

        self.encoder = Encoder(config, name = 'encoder')
        
        self.pooler = TransformerPooler(config, name = 'pooler') if add_pooler else None        
    
    def call(self, inputs, output_hidden_states = False, training = False):
        if isinstance(inputs, (list, tuple)):
            input_ids = inputs[0]
            token_type_ids = inputs[1] if len(inputs) > 1 else None
            attention_mask = inputs[2] if len(inputs) > 2 else None
            
        elif isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            token_type_ids = inputs.get('token_type_ids', None)
            attention_mask = inputs.get('attention_mask', None)
        else:
            raise ValueError('The type of inputs should be list or dictionary.')
        
        hidden_states = []
        output = self.embeddings(input_ids, token_type_ids, training = training)
        if output_hidden_states:
            hidden_states.append(output)

        if attention_mask is None:
            attention_mask = tf.constant(1.0, shape = input_ids.shape, dtype = 'float32')

        # attention_mask now has shape (batches, sequence_len),
        # we need to covert it to (batches, 1, 1, sequence_len)
        # so that later it will broadcast to (batches, num_attention_heads, sequence_len, sequence_len)
        attention_mask = tf.reshape(attention_mask, shape = (input_ids.shape[0], 1, 1, input_ids.shape[-1]))

            
        last_hidden_state, layer_outputs = self.encoder(output, attention_mask, output_hidden_states = output_hidden_states, training = training)
        if output_hidden_states:
            hidden_states.extend(layer_outputs)
            
        pooler_output = self.pooler(output) if self.pooler else None

        return (last_hidden_state, pooler_output, hidden_states)

    def _clean_weight_name(self, weight_name):
        if not weight_name.startswith(self.name + '/'):
            return weight_name
        else:
            parts = weight_name.split('/')
            return '/'.join(parts[1:])

    def _append_root_name(self, weight_name):
        return self.name + '/' + weight_name


    def from_checkpoint(self, ckp_path):
        ckc = CheckpointCache(ckp_path)
        dymmy_inputs = np.array([[0,1,2]])
        self([dymmy_inputs])
        
        symbolic_weights = self.trainable_weights + self.non_trainable_weights
        
        variable_keys = [self._clean_weight_name(symbolic_weight.name) for symbolic_weight in symbolic_weights]
        variable_keys = [self._convert_variable_name(key) for key in variable_keys]

        unloaded_keys = set(ckc.keys()) - set(variable_keys)
        print('unloaded keys:', unloaded_keys)
        
        values = [ckc.get_values(key) for key in variable_keys]
        
        name_value_pair = []

        for weight, value in zip(symbolic_weights, values):
            if weight.shape != value.shape:
                raise ValueError(f'The shape of {weight.name} is {weight.shape} but shape from checkpoint is {value.shape}.')
            
            name_value_pair.append((weight, value))
        
        K.batch_set_value(name_value_pair)


    def _convert_variable_name(self, key):
        raise NotImplementedError


class BertModel(Transformer):
    def __init__(self, config, **kwargs):
        super(BertModel, self).__init__(config, **kwargs)

        self.embedding_name_mapping = {
            'embeddings/word_embeddings/embedding/embeddings': 'embeddings/word_embeddings',
            'embeddings/position_embeddings/embeddings': 'embeddings/position_embeddings',
            'embeddings/token_type_embeddings/embeddings': 'embeddings/token_type_embeddings'}

    def _convert_variable_name(self, key):
        # remove last ':0'
        if key.endswith(':0'):
            key = key[:-2]

        prefix = 'bert/'
        if key.startswith('encoder/layer_'):
            return prefix + key.replace('/feedforward', '')
        
        elif key.startswith('embeddings/'):
            return prefix + self.embedding_name_mapping.get(key, key)
        elif key.startswith('pooler/'):
            return prefix + key
        else:
            raise ValueError(f'Invalid variable name {key}')

class HuggingFaceBertModel(Transformer):
    def __init__(self, config, **kwargs):
        super(HuggingFaceBertModel, self).__init__(config, **kwargs)

        self.embedding_name_mapping = {
            'embeddings/word_embeddings/embedding/embeddings:0': 'embeddings/word_embeddings/weight:0',
            'embeddings/position_embeddings/embeddings:0': 'embeddings/position_embeddings/embeddings:0',
            'embeddings/token_type_embeddings/embeddings:0': 'embeddings/token_type_embeddings/embeddings:0'}

    def _convert_variable_name(self, key):
        
        prefix = 'bert/'            
        if key.startswith('encoder/layer_'):
            key = key.replace('layer_', 'layer_._')
            return prefix + key.replace('/feedforward', '')
        
        elif key.startswith('embeddings/'):
            return prefix + self.embedding_name_mapping.get(key, key)
        elif key.startswith('pooler/'):
            return prefix + key
        else:
            raise ValueError(f'Invalid variable name {key}')


