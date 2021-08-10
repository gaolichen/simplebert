#!/usr/bin/env python
# -*- coding: utf-8 -*- 
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
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
  
# now we can import the module in the parent
# directory.
from transformer import BertConfig, Transformer, BertModel, HuggingFaceBertModel
from tokenizer import Tokenizer

config_path = './testdata/bert_config.json'
hf_config_path = './testdata/hf_bert_config.json'

en_cased_vocab_path = './testdata/bert-base-cased-vocab.txt'
en_uncased_vocab_path = './testdata/bert-base-uncased-vocab.txt'
cn_vocab_path = './testdata/bert-base-chinese-vocab.txt'


def average_norm(tensor, axis = -1):
    deno = tf.sqrt(float(tensor.shape[axis]))
    return tf.norm(tensor, axis = axis) / deno


class BertConfigTestCase(unittest.TestCase):

    def setUp(self):
        self.config_path = config_path

    def test_constructor(self):
        config = BertConfig(path = self.config_path)
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
        config = BertConfig(path = self.config_path, hidden_act = 'relu', attention_probs_dropout_prob = 0.3)
        self.assertEqual(config.attention_probs_dropout_prob, 0.3)
        self.assertEqual(config.hidden_act, 'relu')

class TransformerTestCase(unittest.TestCase):

    def setUp(self):
        self.eps = 1e-2
        self.config_path = config_path
        self.checkpoint_path = r'E:\deeplearning-data\cased_L-12_H-768_A-12\bert_model.ckpt'
        self.h5file_path = r'E:\deeplearning-data\huggingface-bert-chinese\tf_bert_chinese.h5'


    def test_constructor(self):
        config = BertConfig(path = config_path)
        bert = BertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.checkpoint_path)
        print(bert.summary())

    def test_call(self):
        config = BertConfig(path = self.config_path)
        bert = BertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.checkpoint_path)

        tokenizer = Tokenizer(en_cased_vocab_path)
        maxlen = 20
        text = "Train the model for three epochs."
        print(tokenizer.tokenize(text))
        inputs = tokenizer([text], return_np = True, return_dict = True)

        print(inputs['input_ids'].shape)
        output = bert(inputs, output_hidden_states = True)
        print(output['sequence_output'], output['hidden_states'][0])


    def test_call_lml(self):
        config = BertConfig(path = self.config_path)
        bert = BertModel(config = config, model_head = 'lml', name = 'bert')
        bert.load_checkpoint(self.checkpoint_path)

        tokenizer = Tokenizer(en_cased_vocab_path)
        maxlen = 20
        text = "Train the model for three epochs."
        print(tokenizer.tokenize(text))
        inputs = tokenizer([text], return_np = True, return_dict = True)

        print(inputs['input_ids'].shape)
        output = bert(inputs, output_hidden_states = True)
        print(output['sequence_output'], output['hidden_states'][0])


    def test_call_HuggingFace(self):
        text = u'原标题：打哭伊藤美诚！孙颖莎一个词形容：过瘾！……'
        
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.h5file_path)
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = True)

        hf_hidden_states = huggingface_model_output(text)

        save_output(output['hidden_states'], num_layers = 2, file_name = './my_bert')
        save_output(hf_hidden_states, num_layers = 2, file_name = './hf_bert')

        for i in range(len(hf_hidden_states)):
            diff = output['hidden_states'][i] - hf_hidden_states[i]
            norm = average_norm(diff, axis = -1)
            self.assertTrue(tf.reduce_all(norm < self.eps).numpy(), f'i={i}, norm = {norm}')
            norm = average_norm(output['hidden_states'][i], axis = -1)
            self.assertTrue(tf.reduce_all(norm > self.eps).numpy(), norm)


    def test_call_HuggingFace_textpair(self):
        text1 = u'磁铁会吸引某些金属，但也会排斥其他磁铁，那么为什么人们只能感觉到地心引力呢？'
        text2 = u'1915年，阿尔伯特·爱因斯坦发表了著名的广义相对论，找到了其中的答案。'
        
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.h5file_path)
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text1], second_text = [text2], return_np = True, return_dict = True, maxlen = 80)
        #print('my_model inputs:', inputs)
        output = bert(inputs, output_hidden_states = True)

        hf_hidden_states = huggingface_model_output(text1, text2, maxlen = 80)

        for i in range(len(hf_hidden_states)):
            diff = output['hidden_states'][i] - hf_hidden_states[i]
            norm = average_norm(diff, axis = -1)
            self.assertTrue(tf.reduce_all(norm < self.eps).numpy(), f'i={i}, norm = {norm}')
            norm = average_norm(output['hidden_states'][i], axis = -1)
            self.assertTrue(tf.reduce_all(norm > self.eps).numpy(), norm)


    def test_LMHead(self):
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lml', name = 'bert')
        bert.load_checkpoint(self.h5file_path)

        text = u'我有一个[MASK]想。'
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = False)
        hf_logits = higgingface_lmlmodel_output(text)

        norm = average_norm(output['logits'] - hf_logits, axis = -1)
        self.assertTrue(tf.reduce_all(norm < 1e-2).numpy(), f'norm = {norm}')


    def test_causal_lml(self):
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lml', causal_attention_mask = True, name = 'bert')
        bert.load_checkpoint(self.h5file_path)

        text = u'我有一个梦想。'
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = False)

        tokens = logits_to_tokens(output['logits'], tokenizer, topk = 5)
        print(tokens)

    def test_causal_attention(self):
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lml', causal_attention_mask = True, name = 'bert')
        bert.load_checkpoint(self.h5file_path)

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
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'pooler', name = 'bert')
        bert.load_checkpoint(self.h5file_path)

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
        config = BertConfig(path = self.config_path)
        bert_model = BertModel(config, model_head = 'lml', causal_attention_mask = True, name = 'bert')
        bert_model.load_checkpoint(self.checkpoint_path)

        inputs = dict(input_ids=keras.layers.Input(shape=(input_dim,), dtype=tf.int32),
                        attention_mask=keras.layers.Input(shape=(input_dim, ), dtype=tf.int32))
            
        output = bert_model(inputs)
        
        model = keras.models.Model(inputs, output['logits'], name = 'TFBert')
          
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = 3e-5), loss = 'mse')

        print(model.summary())

        tokenizer = Tokenizer(en_cased_vocab_path)
        text = "Train the model for three epochs."
        model_inputs = tokenizer([text], return_np = True, return_dict = True, maxlen = input_dim)
        model_out = model(model_inputs)
        print(model_out.shape)


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
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir = r'C:\Users\Gaoli\.cache\huggingface\transformers')
    model = TFBertModel.from_pretrained('bert-base-chinese', cache_dir = r'C:\Users\Gaoli\.cache\huggingface\transformers')
    if text2 is None:
        inputs = tokenizer([text1], return_tensors = 'np')
    else:
        inputs = tokenizer([text1], [text2], padding = 'max_length', max_length = maxlen, return_tensors = 'np')

    #print('hf_model inputs:', inputs)

    output = model(inputs, output_hidden_states = True)
    return output.hidden_states

def higgingface_lmlmodel_output(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir = r'C:\Users\Gaoli\.cache\huggingface\transformers')
    model = TFBertForMaskedLM.from_pretrained('bert-base-chinese', cache_dir = r'C:\Users\Gaoli\.cache\huggingface\transformers')
    inputs = tokenizer([text], return_tensors = 'np')
    output = model(inputs, output_hidden_states = False)
    return output.logits

def higgingface_pooler_output(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir = r'C:\Users\Gaoli\.cache\huggingface\transformers')
    model = TFBertModel.from_pretrained('bert-base-chinese', cache_dir = r'C:\Users\Gaoli\.cache\huggingface\transformers')
    inputs = tokenizer(texts, return_tensors = 'tf', max_length = 64, padding = 'max_length')
    output = model(inputs, output_hidden_states = False)
    return output.pooler_output

def save_output(hidden_states, num_layers, file_name):
    for k in range(num_layers):
        with open(file_name + f'_layer_{k}.txt', 'w') as f:
            obj = {"value": hidden_states[k].numpy().tolist()[0]}
            f.write(json.dumps(obj))
        
def suite():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BertConfigTestCase)
    suite.addTest(BertConfigTestCase('test_constructor'))
    suite.addTest(BertConfigTestCase('test_constructor_with_extra_args'))
    
    suite.addTest(TransformerTestCase('test_constructor'))
    suite.addTest(TransformerTestCase('test_call'))
    suite.addTest(TransformerTestCase('test_call_lml'))
    suite.addTest(TransformerTestCase('test_call_HuggingFace'))
    suite.addTest(TransformerTestCase('test_call_HuggingFace_textpair'))
    suite.addTest(TransformerTestCase('test_LMHead'))
    suite.addTest(TransformerTestCase('test_causal_lml'))
    suite.addTest(TransformerTestCase('test_call_batch'))
    suite.addTest(TransformerTestCase('test_build_model'))
    suite.addTest(TransformerTestCase('test_causal_attention'))
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

        
