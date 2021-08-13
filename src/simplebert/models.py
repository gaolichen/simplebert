#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : models.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import copy
import json
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from simplebert.pretrained import CheckpointCache, CheckpointManager

def get_initializer(init_range):
    return keras.initializers.RandomNormal(mean = 0.0, stddev = init_range)

def gelu(features, name = None):
    return tf.nn.gelu(features, approximate = False, name = name)

def lower_triangle_matrix(n):
    return np.tril([1] * n, k = 0)

#@tf.function
def shape_list(tensor):
    return tensor.shape

class ModelConfig(object):
    """The config class for model.
    """
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
    def __init__(self, vocab_size, dim, initializer_range = 0.02, name = None):
        super(SequenceEmbedding, self).__init__(name = name)

        self.dim = dim
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range


    def build(self, input_shape):
        self.weight = self.add_weight(shape = (self.vocab_size, self.dim), trainable = True,
                                          initializer = get_initializer(self.initializer_range), name = 'weight')
    
        
    def call(self, x):
        return tf.gather(self.weight, indices = x)
    

class PositionEmbedding(keras.layers.Layer):
    """Encode position information of input.

    PE(pos, 2*i) = sin(pos/10000**(2*i/dim))
    PE(pos, 2*i+1) = cos(pos/10000**(2*i/dim))
    """
    def __init__(self, dim, max_position_embeddings = 5000, name = None):
        super(PositionEmbedding, self).__init__(name = name)

        self.max_len = max_position_embeddings
        self.dim = dim

    def build(self, input_shape):
        # initialize positional encoding
        self.embeddings = self.add_weight(shape = (self.max_len, self.dim), trainable = True,
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
        input_shape = shape_list(input_ids)
        position_ids = tf.expand_dims(tf.range(start = 0, limit = input_shape[-1]), axis = 0)
        output = tf.gather(self.embeddings, indices = position_ids)
        if input_shape[0] is not None:
            return tf.tile(input = output, multiples = (input_shape[0], 1, 1))
        else:
            return output

class TransformerEmbedding(keras.layers.Layer):
    def __init__(self, config, name = None, **kwargs):
        super(TransformerEmbedding, self).__init__(name = name, **kwargs)

        with tf.name_scope('embeddings') as scope:
            self.word_embeddings = SequenceEmbedding(vocab_size = config.vocab_size,
                                                     dim = config.hidden_size,
                                                     initializer_range = config.initializer_range,
                                                     name = 'word_embeddings')
            
            self.position_embeddings = PositionEmbedding(dim = config.hidden_size,
                                                       max_position_embeddings = config.max_position_embeddings,
                                                       name = "position_embeddings")
            
            self.token_type_embeddings = keras.layers.Embedding(input_dim = 2,
                                                                output_dim = config.hidden_size,
                                                                name = "token_type_embeddings")

            self.sum_layer = keras.layers.Add(name = 'add')
            self.norm = keras.layers.LayerNormalization(epsilon = 1e-12, name = "LayerNorm")
            self.dropout = keras.layers.Dropout(rate = config.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids = None, training = False):
        input_shape = shape_list(input_ids)
        
        x1 = self.word_embeddings(input_ids)
        x2 = self.position_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = tf.fill(tf.shape(input_ids), 0)
        x3 = self.token_type_embeddings(token_type_ids)

        output = self.sum_layer([x1, x2, x3])
        output = self.norm(output, training = training)
        return self.dropout(output, training = training)


class SublayerConnection(tf.keras.layers.Layer):
    """Performs add-and-norm operations on sublayers.
    """
    def __init__(self, config, **kwargs):
        super(SublayerConnection, self).__init__(**kwargs)
        
        self.dense = keras.layers.Dense(units = config.hidden_size,
                                        kernel_initializer = get_initializer(config.initializer_range),
                                        name = 'dense')
        self.norm = keras.layers.LayerNormalization(name = 'LayerNorm')
        self.dropout = keras.layers.Dropout(rate = config.hidden_dropout_prob)
        

    def call(self, x, output, **kwargs):
        output = self.dropout(self.dense(output))
        return self.norm(x + output)



def attention(querys, keys, values, mask = None, dropout = None, training = False):
    """ Computes attentions.

        Args:
            querys:
                Querys of shape (batch_size, query_length, dim_k)
            keys:
                Keys of shape (batch_size, sequence_length, dim_k)
            values:
                Values of shape (batch_size, sequence_length, dim_v)
            mask:
                Attention mask.
            dropout:
                Dropout operator.
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
    def __init__(self, config, name = None, **kwargs):
        super(MultiHeadedAttention, self).__init__(name = name, **kwargs)
        assert config.hidden_size % config.num_attention_heads == 0

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.d_k = config.hidden_size // config.num_attention_heads
        
        self.dropout = keras.layers.Dropout(rate = config.attention_probs_dropout_prob)
        
        self.w_q = keras.layers.Dense(units = self.hidden_size,
                                      kernel_initializer = get_initializer(config.initializer_range),
                                      name='query')
        self.w_k = keras.layers.Dense(units = self.hidden_size,
                                      kernel_initializer = get_initializer(config.initializer_range),
                                      name='key')
        self.w_v = keras.layers.Dense(units = self.hidden_size,
                                      kernel_initializer = get_initializer(config.initializer_range),
                                      name='value')
        #self.w_o = keras.layers.Dense(units = self.dim, kernel_initializer = get_initializer(config.initializer_range), name='output')

        
    def call(self, querys, keys, values, mask = None, training = False):
        """ Computes multi-head attention scores.

            querys.shape = (batches, sequence_len, d_model)
            keys.shape = (batches, sequence_len, d_model)
            values.shape = (batches, sequence_len, d_model)
        """

        sequence_len = shape_list(keys)[1]

        # q_i = Q * W^Q_i
        #print('query.shape=', querys.shape)
        querys = self.w_q(querys)
        querys = tf.reshape(querys, shape = (-1, sequence_len, self.num_attention_heads, self.d_k))
        # At this point, the shapes of querys, keys, and values are (batches, sequence_len, num_attention_heads, d_k)
        # we need to change the shapes to (batches, num_attention_heads, sequence_len, d_k)
        querys = tf.transpose(querys, perm = [0, 2, 1, 3])

        # the same operations on keys and values
        # k_i = K * W^K_i
        keys = self.w_k(keys)
        keys = tf.reshape(keys, shape = (-1, sequence_len, self.num_attention_heads, self.d_k))
        keys = tf.transpose(keys, perm = [0, 2, 1, 3])
        
        # v_i = V * W^V_i
        values = self.w_v(values)
        values = tf.reshape(values, shape = (-1, sequence_len, self.num_attention_heads, self.d_k))
        values = tf.transpose(values, perm = [0, 2, 1, 3])

        v, self.attn = attention(querys, keys, values, mask = mask, dropout = self.dropout)

        # the shape of v is (batches, h, sequence_len, d_k)
        # we need to change it to (batches, sequence_len, num_attention_heads, d_k)
        # and then concatenate last two dims (batches, sequence_len, num_attention_heads * d_k)
        return tf.reshape(tf.transpose(v, perm = [0, 2, 1, 3]), (-1, sequence_len, self.num_attention_heads * self.d_k))


class AttentionLayer(keras.layers.Layer):
    """Sublay for attention computation.
    """
    def __init__(self, config, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

        self.self_attn = MultiHeadedAttention(config, name = 'self')
        self.conn = SublayerConnection(config, name = 'output')

    def call(self, x, attention_mask, training = False):
        output1 = self.self_attn(x, x, x, mask = attention_mask, training = training)
        return self.conn(x, output1, training = training)
        

class PositionwiseFeedForward(keras.layers.Layer):
    """Positionwise FeedForward computation.
    """
    def __init__(self, config, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)

        # gelu activation has approximation and exact versions.
        # For simplicity, we always use exact version
        self.dense = keras.layers.Dense(units = config.intermediate_size,
                                        activation = config.hidden_act,
                                        kernel_initializer = get_initializer(config.initializer_range),
                                        name = 'dense')
        
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, x, training = False):
        output = self.dense(x)
        return self.dropout(output, training = training)

class FeedForwardLayer(keras.layers.Layer):
    """Sublayer for positionwise FeedForward computation.
    """
    def __init__(self, config, **kwargs):
        super(FeedForwardLayer, self).__init__(**kwargs)

        self.feed_forward = PositionwiseFeedForward(config, name = 'intermediate')
        self.conn = SublayerConnection(config, name = 'output')

    def call(self, x, training = False):
        output = self.feed_forward(x, training = training)
        return self.conn(x, output, training = training)


class EncoderLayer(keras.layers.Layer):
    """Base class of transformer layers
    """
    def __init__(self, config, name = None):
        super(EncoderLayer, self).__init__(name = name)
        
        self.self_attn = AttentionLayer(config, name = 'attention')
        self.feed_forward = FeedForwardLayer(config, name = 'feedforward')
        
    def call(self, x, attention_mask, training = False, **kwargs):
        output = self.self_attn(x, attention_mask, training = training)
        return self.feed_forward(output, training = training)

class Encoder(keras.layers.Layer):
    """The encoder containing a list of main layers
    """
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
    """The pooler model head computing the pooler output of the model.
    """
    def __init__(self, config, name = None):
        super(TransformerPooler, self).__init__(name = name)
        
        self.dense = keras.layers.Dense(units = config.hidden_size,
                                        kernel_initializer=get_initializer(config.initializer_range),
                                        activation = 'tanh', name = 'dense')

    def call(self, hidden_states):
        return self.dense(hidden_states)

class LanguageModelHead(keras.layers.Layer):
    """The model head compute final output of language model.
    """
    def __init__(self, config, word_embedding, name = None):
        super(LanguageModelHead, self).__init__(name = name)

        self.word_embedding = word_embedding
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.initializer_range = config.initializer_range

    def build(self, input_shape):
        with tf.name_scope('transform'):
            self.dense = keras.layers.Dense(
                units=self.hidden_size,
                input_shape = (self.hidden_size,),
                kernel_initializer=get_initializer(self.initializer_range),
                activation = self.hidden_act,
                name="transform/dense",
            )
            
            self.norm = keras.layers.LayerNormalization(epsilon = 1e-12, name="transform/LayerNorm")
            
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        
        super().build(input_shape)
        
    def call(self, x, training = False):
        hidden_states = self.dense(x)
        hidden_states = self.norm(hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(hidden_states, shape = (-1, self.hidden_size))
        hidden_states = tf.matmul(a = hidden_states, b = self.word_embedding.weight, transpose_b = True)
        hidden_states = tf.reshape(hidden_states, shape=(-1, seq_length, self.vocab_size))
        hidden_states = tf.nn.bias_add(value = hidden_states, bias = self.bias)

        return hidden_states

class Transformer(keras.models.Model):
    """Base class of transformers
    """
    def __init__(self, config,
                 model_head = None,
                 causal_attention = False,
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.config = config
        self.causal_attention = causal_attention
        
        self.embeddings = TransformerEmbedding(config = config, name = 'embeddings')
        self.encoder = Encoder(config, name = 'encoder')

        if model_head is None:
            model_head = []
        elif isinstance(model_head, str):
            model_head = [model_head]

        if not isinstance(model_head, (list, tuple)):
            raise ValueError('model_head should be of type str or list of str')

        self.pooler = TransformerPooler(config, name = 'pooler') if 'pooler' in model_head else None
        self.lm_head = LanguageModelHead(config,
                                     word_embedding = self.embeddings.word_embeddings,
                                     name = 'predictions') if 'lm' in model_head else None
    
    def call(self, inputs, output_hidden_states = False, training = False):
        """call method of the model.

            Args:
                inputs:
                    Can be of list/tuple type as `[input_ids, token_type_ids, attention_mask]` or
                    of dict type as `{"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}`
                output_hidden_state:
                    It it is set to True, output of all hidden layers will be returned.
                training:
                    True if in training mode, False otherwise.
                    
            Returns:
                Dict type of the form `{"sequence_output": sequence_output, "pooler_output": pooler_output,
                "logits": logits, "hidden_states": hidden_states}.`
        """
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
        
        input_shape = shape_list(input_ids)
        
#        last_hidden_state = tf.ones(input_shape + (self.config.hidden_size))
#        output = tf.ones(input_shape + (self.config.hidden_size,))
#        logits = tf.ones(input_shape + (self.config.vocab_size,))
#        pooler_output = tf.ones((input_shape[0], self.config.hidden_size))
        
        hidden_states = [] if output_hidden_states else None
        output = self.embeddings(input_ids, token_type_ids, training = training)
        
        if output_hidden_states:
            hidden_states.append(output)

        if self.causal_attention:
            attention_mask = tf.constant(lower_triangle_matrix(input_shape[-1]))
            attention_mask = tf.reshape(attention_mask, shape = (1, 1, input_shape[-1], input_shape[-1]))
            
        else:
            if attention_mask is None:
                attention_mask = tf.constant(1.0, shape = input_shape, dtype = 'float32')
            # attention_mask now has shape (batches, sequence_len),
            # we need to covert it to (batches, 1, 1, sequence_len)
            # so that later it will broadcast to (batches, num_attention_heads, sequence_len, sequence_len)
            attention_mask = tf.reshape(attention_mask, shape = (input_shape[0], 1, 1, input_shape[-1]))

            
        
        last_hidden_state, layer_outputs = self.encoder(output, attention_mask, output_hidden_states = output_hidden_states, training = training)
        if output_hidden_states:
            hidden_states.extend(layer_outputs)
            
        pooler_output = self.pooler(tf.gather(last_hidden_state, indices = 0, axis = 1)) if self.pooler else None
        logits = self.lm_head(last_hidden_state) if self.lm_head else None

        res = {'sequence_output': last_hidden_state,
               'pooler_output': pooler_output,
               'logits': logits,
               'hidden_states': hidden_states}

        return {k : v for k, v in res.items() if v is not None}


    def _clean_weight_name(self, weight_name):
        if not weight_name.startswith(self.name + '/'):
            return weight_name
        else:
            parts = weight_name.split('/')
            return '/'.join(parts[1:])

    def _append_root_name(self, weight_name):
        return self.name + '/' + weight_name


    def load_checkpoint(self, checkpoint_path, silent = False):
        """Loads weights from checkpoint.

            Args:
                checkpoint_path:
                    Local path to the checkpoint. The checkpoint can be either .cpk or .h5 file type.

            Returns:
                List of unused weights of the checkpoint.
        """
        ckc = CheckpointCache(checkpoint_path)

        dymmy_inputs = np.array([[0,1,2]])
        self([dymmy_inputs])
        
        symbolic_weights = self.trainable_weights + self.non_trainable_weights
        
        variable_keys = [self._clean_weight_name(symbolic_weight.name) for symbolic_weight in symbolic_weights]
        variable_keys = [self._convert_variable_name(key) for key in variable_keys]

        unloaded_keys = set(ckc.keys()) - set(variable_keys)
        if not silent:
            print('unused keys:', unloaded_keys)
        
        values = [ckc.get_values(key) for key in variable_keys]
        
        name_value_pair = []

        for weight, value in zip(symbolic_weights, values):
            if weight.shape != value.shape:
                raise ValueError(f'The shape of {weight.name} is {weight.shape} but shape from checkpoint is {value.shape}.')
            if weight.dtype != value.dtype:
                raise ValueError(f'The type of {weight.name} is {weight.dtype} but type from checkpoint is {value.dtype}.')
            
            name_value_pair.append((weight, value))
        
        K.batch_set_value(name_value_pair)
        
        return unloaded_keys
        

    def _convert_variable_name(self, key):
        raise NotImplementedError


class BertModel(Transformer):
    """Model class for the original BERT implementation.
    """
    def __init__(self, config, **kwargs):
        super(BertModel, self).__init__(config, **kwargs)

        self.embedding_name_mapping = {
            'embeddings/word_embeddings/weight': 'embeddings/word_embeddings',
            'embeddings/position_embeddings/embeddings': 'embeddings/position_embeddings',
            'embeddings/token_type_embeddings/embeddings': 'embeddings/token_type_embeddings',
            'predictions/bias': 'predictions/output_bias'}

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
        elif key.startswith('predictions/'):
            return 'cls/' + self.embedding_name_mapping.get(key, key)
        else:
            raise ValueError(f'Invalid variable name {key}')

class HuggingFaceBertModel(Transformer):
    """Model class for the HuggingFace implementations of BERT
    """
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
        elif key.startswith('predictions/'):
            return 'mlm___cls/' + key
        else:
            raise ValueError(f'Invalid variable name {key}')


name_to_class = {}

def register_class(cls):
    name_to_class[cls.__name__] = cls

register_class(BertModel)
register_class(HuggingFaceBertModel)


def model_from_pretrained(model_name,
                          model_head = None,
                          causal_attention = False,
                          **kwargs):
    """
    Creates a model and initializes its weights from a pretrained checkpoint.

    Args:
        model_name:
            Name of the pretrained model.
        model_head:
            The name of head on top of the main layer. Its type can be either `str` or `list[str]` 
        causal_attention:
            Lower triangle attention mask is applied if it is True.
        **kwargs:
            Other

    Returns:
        The pretrained transformer model.

    """
    checkpoint_manager = CheckpointManager()
    class_name = checkpoint_manager.get_class(model_name)
    if not class_name in name_to_class:
        raise ValueError(f'{class_name} is not a valid Transformer class.')
    
    cls = name_to_class[class_name]
    config = ModelConfig(checkpoint_manager.get_config_path(model_name))
    model = cls(config = config, model_head = model_head, causal_attention = causal_attention, **kwargs)

    checkpoint_path = checkpoint_manager.get_checkpoint_path(model_name)
    model.load_checkpoint(checkpoint_path)
    
    return model

if __name__ == '__main__':
    print(name_to_class)
    
