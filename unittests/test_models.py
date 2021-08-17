#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_models.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import unittest
import numpy as np
import json
import tensorflow as tf
import tensorflow.keras as keras

import sys
import os
from transformers import TFBertModel, BertTokenizer, TFBertForMaskedLM
  
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
simple_bert_dir=os.path.join(parent, 'src')

# adding the parent directory to 
# the sys.path.
sys.path.append(simple_bert_dir)
  
# now we can import the module in the parent
# directory.
from simplebert.tokenizers import Tokenizer
from simplebert.models import ModelConfig, BertModel, HuggingFaceBertModel, model_from_pretrained
from simplebert.pretrained import CheckpointManager

config_path = os.path.join(current, './testdata/bert_config.json')
#hf_config_path = os.path.join(current, './testdata/hf_bert_config.json')

en_cased_vocab_path = os.path.join(current, './testdata/bert-base-cased-vocab.txt')
en_uncased_vocab_path = os.path.join(current, './testdata/bert-base-uncased-vocab.txt')
cn_vocab_path = os.path.join(current, './testdata/bert-base-chinese-vocab.txt')


def average_norm(tensor, axis = -1):
    deno = tf.sqrt(float(tensor.shape[axis]))
    return tf.norm(tensor, axis = axis) / deno


class ModelConfigTestCase(unittest.TestCase):

    def setUp(self):
        self.config_path = config_path

    def test_constructor(self):
        config = ModelConfig(path = self.config_path)
        self.assertEqual(config.attention_probs_dropout_prob, 0.1)
        self.assertEqual(config.hidden_act, 'gelu')
        self.assertEqual(config.hidden_dropout_prob, 0.1)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.initializer_range, 0.02)
        self.assertEqual(config.intermediate_size, 3072)
        self.assertEqual(config.max_position_embeddings, 512)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.type_vocab_size, 2)
        self.assertEqual(config.vocab_size, 28996)

    def test_constructor_with_extra_args(self):
        config = ModelConfig(path = self.config_path, hidden_act = 'relu', attention_probs_dropout_prob = 0.3)
        self.assertEqual(config.attention_probs_dropout_prob, 0.3)
        self.assertEqual(config.hidden_act, 'relu')

class TransformerTestCase(unittest.TestCase):
    
    def setUp(self):
        self.eps = 1e-2

        cm = CheckpointManager()
        self.config_path = cm.get_config_path('bert-base-cased')
        self.checkpoint_path = cm.get_checkpoint_path('bert-base-cased')

        self.hf_config_path = cm.get_config_path('huggingface-bert-base-chinese')
        self.h5file_path = cm.get_checkpoint_path('huggingface-bert-base-chinese')


    def test_constructor(self):
        config = ModelConfig(path = config_path)
        bert = BertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.checkpoint_path, silent = True)
        #summary = bert.summary()

    def test_call(self):
        config = ModelConfig(path = self.config_path)
        bert = BertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.checkpoint_path, silent = True)

        tokenizer = Tokenizer(en_cased_vocab_path)
        maxlen = 20
        text = "Train the model for three epochs."
        inputs = tokenizer([text], return_np = True, return_dict = True)

        output = bert(inputs, output_hidden_states = True)
        self.assertFalse(output['sequence_output'] is None)
        self.assertFalse(output['hidden_states'][0] is None)


    def test_call_lm(self):
        config = ModelConfig(path = self.config_path)
        bert = BertModel(config = config, model_head = 'lm', name = 'bert')
        bert.load_checkpoint(self.checkpoint_path, silent = True)

        tokenizer = Tokenizer(en_cased_vocab_path)
        maxlen = 20
        text = "Train the model for three epochs."
        inputs = tokenizer([text], return_np = True, return_dict = True)

        output = bert(inputs, output_hidden_states = False)
        self.assertFalse(output['sequence_output'] is None)
        self.assertFalse(output['logits'] is None)
        

    def test_call_HuggingFace(self):
        text = u'原标题：打哭伊藤美诚！孙颖莎一个词形容：过瘾！……'
        
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = True)

        hf_hidden_states = huggingface_model_output(text)

        for i in range(len(hf_hidden_states)):
            diff = output['hidden_states'][i] - hf_hidden_states[i]
            norm = average_norm(diff, axis = -1)
            self.assertTrue(tf.reduce_all(norm < self.eps).numpy(), f'i={i}, norm = {norm}')
            norm = average_norm(output['hidden_states'][i], axis = -1)
            self.assertTrue(tf.reduce_all(norm > self.eps).numpy(), norm)


    def test_call_HuggingFace_textpair(self):
        text1 = u'磁铁会吸引某些金属，但也会排斥其他磁铁，那么为什么人们只能感觉到地心引力呢？'
        text2 = u'1915年，阿尔伯特·爱因斯坦发表了著名的广义相对论，找到了其中的答案。'
        
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text1], second_text = [text2], return_np = True, return_dict = True, maxlen = 80)
        output = bert(inputs, output_hidden_states = True)

        hf_hidden_states = huggingface_model_output(text1, text2, maxlen = 80)

        for i in range(len(hf_hidden_states)):
            diff = output['hidden_states'][i] - hf_hidden_states[i]
            norm = average_norm(diff, axis = -1)
            self.assertTrue(tf.reduce_all(norm < self.eps).numpy(), f'i={i}, norm = {norm}')
            norm = average_norm(output['hidden_states'][i], axis = -1)
            self.assertTrue(tf.reduce_all(norm > self.eps).numpy(), norm)


    def test_LMHead(self):
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lm', name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)

        text = u'我有一个[MASK]想。'
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = False)
        hf_logits = higgingface_lm_model_output(text)

        norm = average_norm(output['logits'] - hf_logits, axis = -1)
        self.assertTrue(tf.reduce_all(norm < 1e-2).numpy(), f'norm = {norm}')


    def test_causal_lm(self):
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lm', causal_attention = True, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)

        text = u'我有一个梦想。'
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = False)

        tokens = logits_to_tokens(output['logits'], tokenizer, topk = 5)
        self.assertEqual(tokens.shape, (inputs['input_ids'].shape[1], 5))

    def test_causal_attention(self):
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lm', causal_attention = True, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)

        text = u'我有一个梦想。'
        tokenizer = Tokenizer(cn_vocab_path)
        input_size = len(tokenizer.tokenize(text)) + 1

        maxlen = 20

        inputs = tokenizer([text], return_np = True, return_dict = True, maxlen = maxlen)
        output = bert(inputs, output_hidden_states = False)['logits']

        seq_len = output.shape[1]

        text2 = u'我有一个梦想。你有吗？'
        inputs = tokenizer([text2], return_np = True, return_dict = True, maxlen = maxlen)
        output2 = bert(inputs, output_hidden_states = False)['logits']
        
        
        diff_norm = average_norm(output - output2, axis = -1)[0]
        epsilon = tf.constant(1e-5, shape = diff_norm.shape)
        bool_res = tf.less(diff_norm, 1e-5)
        self.assertTrue(tf.reduce_all(bool_res[:input_size]).numpy())
        self.assertFalse(tf.reduce_any(bool_res[input_size:]).numpy())
                

    def test_call_batch(self):
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'pooler', name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)

        text1 = u'磁铁会吸引某些金属，但也会排斥其他磁铁，那么为什么人们只能感觉到地心引力呢？'
        text2 = u'1915年，阿尔伯特·爱因斯坦发表了著名的广义相对论，找到了其中的答案。'
        tokenizer = Tokenizer(cn_vocab_path)
        inputs = tokenizer([text1, text2], return_np = True, return_dict = True, maxlen = 64)
        output = bert(inputs, output_hidden_states = False)

        hf_pooler_output = higgingface_pooler_output([text1, text2])
        
        diff = output['pooler_output'] - hf_pooler_output
        diff_norm = average_norm(diff)
        bool_res = tf.less(diff_norm, 1e-2)
        self.assertTrue(tf.reduce_all(bool_res).numpy(), f'diff_norm={diff_norm}')
        

    def test_build_model(self):
        input_dim = 64
        config = ModelConfig(path = self.config_path)
        bert_model = BertModel(config, model_head = 'lm', causal_attention = True, name = 'bert')
        bert_model.load_checkpoint(self.checkpoint_path, silent = True)

        inputs = dict(input_ids=keras.layers.Input(shape=(input_dim,), dtype=tf.int32),
                        attention_mask=keras.layers.Input(shape=(input_dim, ), dtype=tf.int32))
            
        output = bert_model(inputs)
        
        model = keras.models.Model(inputs, output['logits'], name = 'TFBert')
          
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = 3e-5), loss = 'mse')

        tokenizer = Tokenizer(en_cased_vocab_path)
        text = "Train the model for three epochs."
        model_inputs = tokenizer([text], return_np = True, return_dict = True, maxlen = input_dim)
        model_out = model(model_inputs)
        self.assertEqual(model_out.shape, model_inputs['input_ids'].shape + (tokenizer.vocab_size,))

    def test_save_load_weights(self):
        tokenizer = Tokenizer(en_cased_vocab_path, cased = True)
        print(tokenizer.vocab_size)
        config = ModelConfig(path = self.config_path)
        bert_model = BertModel(config, model_head = 'lm', name = 'bert')
        self.assertFalse(bert_model.built)
        bert_model.load_checkpoint(self.checkpoint_path, silent = True)
        self.assertTrue(bert_model.built)
        text = 'The capital of France is [MASK].'
        inputs = tokenizer(text)
        logits = bert_model(inputs)['logits']
        
        save_path = r'.\tmp.ckpt'
        print(f'saving weights to {save_path}')
        bert_model.save_weights(save_path)
        del bert_model
        
        bert_model = BertModel(config, model_head = 'lm', name = 'bert')
        print(f'loading weights')
        bert_model.load_weights(save_path)
        print(f'removing weights')
        os.remove(save_path + '.data-00000-of-00001')
        os.remove(os.path.join(os.path.dirname(save_path), 'checkpoint'))
        os.remove(save_path + '.index')

        logits2 = bert_model(inputs)['logits']

        diff_norm = average_norm(logits2 - logits)
        bool_res = tf.less(diff_norm, 1e-6)
        self.assertTrue(tf.reduce_all(bool_res).numpy(), f'diff_norm={diff_norm}')


def logits_to_tokens(logits, tokenizer, topk):
    prob = tf.nn.softmax(logits, axis = -1)
    indices = tf.argsort(prob, axis = -1, direction = 'DESCENDING')
    indices = indices[:, :, :topk]

    flat_indices = tf.reshape(indices, [-1]).numpy()
    tokens = tokenizer.ids_to_tokens(flat_indices)
    tokens = np.array(tokens)
    tokens = np.reshape(tokens, (-1, topk))
    return tokens

        
def huggingface_model_output(text1, text2 = None, maxlen = None):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = TFBertModel.from_pretrained('bert-base-chinese')
    if text2 is None:
        inputs = tokenizer([text1], return_tensors = 'np')
    else:
        inputs = tokenizer([text1], [text2], padding = 'max_length', max_length = maxlen, return_tensors = 'np')

    output = model(inputs, output_hidden_states = True)
    return output.hidden_states

def higgingface_lm_model_output(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = TFBertForMaskedLM.from_pretrained('bert-base-chinese')
    inputs = tokenizer([text], return_tensors = 'np')
    output = model(inputs, output_hidden_states = False)
    return output.logits


def higgingface_pooler_output(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = TFBertModel.from_pretrained('bert-base-chinese')
    inputs = tokenizer(texts, return_tensors = 'tf', max_length = 64, padding = 'max_length')
    output = model(inputs, output_hidden_states = False)
    return output.pooler_output

        
def suite():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModelConfigTestCase)
    suite.addTest(ModelConfigTestCase('test_constructor'))
    suite.addTest(ModelConfigTestCase('test_constructor_with_extra_args'))
    
    suite.addTest(TransformerTestCase('test_constructor'))
    suite.addTest(TransformerTestCase('test_call'))
    suite.addTest(TransformerTestCase('test_call_lm'))
    suite.addTest(TransformerTestCase('test_call_HuggingFace'))
    suite.addTest(TransformerTestCase('test_call_HuggingFace_textpair'))
    suite.addTest(TransformerTestCase('test_LMHead'))
    suite.addTest(TransformerTestCase('test_causal_lm'))
    suite.addTest(TransformerTestCase('test_call_batch'))
    suite.addTest(TransformerTestCase('test_build_model'))
    suite.addTest(TransformerTestCase('test_causal_attention'))

    suite.addTest(TransformerTestCase('test_save_load_weights'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

        
