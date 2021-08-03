#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import unittest
import numpy as np
import json
import tensorflow as tf

import sys
import os
from transformers import TFBertModel, BertTokenizer
  
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
        self.config_path = config_path
        self.checkpoint_path = r'E:\deeplearning-data\cased_L-12_H-768_A-12\bert_model.ckpt'
        self.h5file_path = r'E:\deeplearning-data\huggingface-bert-chinese\tf_bert_chinese.h5'

    def test_constructor(self):
        config = BertConfig(path = config_path)
        bert = BertModel(config = config, name = 'bert')
        bert.from_checkpoint(self.checkpoint_path)
        print(bert.summary())

    def test_call(self):
        config = BertConfig(path = self.config_path)
        bert = BertModel(config = config, add_pooler = False, name = 'bert')
        bert.from_checkpoint(self.checkpoint_path)

        tokenizer = Tokenizer(en_cased_vocab_path)
        maxlen = 20
        text = "Train the model for three epochs."
        print(tokenizer.tokenize(text))
        inputs = tokenizer([text], return_np = True, return_dict = True)

        print(inputs['input_ids'].shape)
        output = bert(inputs, output_hidden_states = True)
        print(output[0].shape, output[2][0])


    def test_call_HuggingFace(self):
        text = u'原标题：打哭伊藤美诚！孙颖莎一个词形容：过瘾！……'
        
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, add_pooler = False, name = 'bert')
        bert.from_checkpoint(self.h5file_path)
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        sequence_output, pooler_output, hidden_states = bert(inputs, output_hidden_states = True)

        hf_hidden_states = huggingface_model_output(text)

        save_output(hidden_states, num_layers = 2, file_name = './my_bert')
        save_output(hf_hidden_states, num_layers = 2, file_name = './hf_bert')

        diff0 = hidden_states[0] - hf_hidden_states[0]
        diff1 = hidden_states[1] - hf_hidden_states[1]
        diff2 = hidden_states[-1] - hf_hidden_states[-1]
        print(tf.norm(diff0, axis = -1))
        print(tf.norm(diff1, axis = -1))
        print(tf.norm(diff2, axis = -1))
        print(tf.norm(hidden_states[1], axis = -1) / tf.sqrt(float(hidden_states[1].shape[-1])))

    def test_call_HuggingFace_textpair(self):
        text1 = u'磁铁会吸引某些金属，但也会排斥其他磁铁，那么为什么人们只能感觉到地心引力呢？'
        text2 = u'1915年，阿尔伯特·爱因斯坦发表了著名的广义相对论，找到了其中的答案。'
        
        config = BertConfig(path = hf_config_path)
        bert = HuggingFaceBertModel(config = config, add_pooler = False, name = 'bert')
        bert.from_checkpoint(self.h5file_path)
        tokenizer = Tokenizer(cn_vocab_path)

        inputs = tokenizer([text1], second_text = [text2], return_np = True, return_dict = True, maxlen = 80)
        #print('my_model inputs:', inputs)
        sequence_output, pooler_output, hidden_states = bert(inputs, output_hidden_states = True)

        hf_hidden_states = huggingface_model_output(text1, text2, maxlen = 80)

        diff0 = hidden_states[0] - hf_hidden_states[0]
        diff1 = hidden_states[1] - hf_hidden_states[1]
        diff2 = hidden_states[-1] - hf_hidden_states[-1]
        print(tf.norm(diff0, axis = -1))
        print(tf.norm(diff1, axis = -1))
        print(tf.norm(diff2, axis = -1))
        print(tf.norm(hidden_states[1], axis = -1) / tf.sqrt(float(hidden_states[1].shape[-1])))
        
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
    suite.addTest(TransformerTestCase('test_call_HuggingFace'))
    suite.addTest(TransformerTestCase('test_call_HuggingFace_textpair'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

        
