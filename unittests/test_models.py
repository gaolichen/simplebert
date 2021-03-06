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
from transformers import TFBertModel, TFBertForMaskedLM
from transformers import BertTokenizer as HFBertTokenizer
  
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
from simplebert import model_from_pretrained
from simplebert.tokenizers import BertTokenizer, BpeTokenizer
from simplebert.models import ModelConfig, BertModel, HuggingFaceBertModel, HuggingFaceRobertaModel
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

        tokenizer = BertTokenizer(en_cased_vocab_path)
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

        tokenizer = BertTokenizer(en_cased_vocab_path)
        maxlen = 20
        text = "Train the model for three epochs."
        inputs = tokenizer([text], return_np = True, return_dict = True)

        output = bert(inputs, output_hidden_states = False)
        self.assertFalse(output['sequence_output'] is None)
        self.assertFalse(output['logits'] is None)
        

    def test_call_HuggingFace(self):
        text = u'???????????????????????????????????????????????????????????????????????????'
        
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)
        tokenizer = BertTokenizer(cn_vocab_path)

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
        text1 = u'??????????????????????????????????????????????????????????????????????????????????????????????????????????????????'
        text2 = u'1915???????????????????????????????????????????????????????????????????????????????????????????????'
        
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)
        tokenizer = BertTokenizer(cn_vocab_path)

        inputs = tokenizer([text1], second_text = [text2], return_np = True, return_dict = True, maxlen = 80)
        output = bert(inputs, output_hidden_states = True)

        hf_hidden_states = huggingface_model_output(text1, text2, maxlen = 80)
        attention_mask = [1.0] * (tf.reduce_sum(inputs['attention_mask']).numpy())
        attention_mask += [0.0] * (80 - len(attention_mask))
        attention_mask = tf.expand_dims(tf.constant(attention_mask, 'float32'), axis = 0)

        for i in range(len(hf_hidden_states)):
            diff = output['hidden_states'][i] - hf_hidden_states[i]
            norm = average_norm(diff, axis = -1)
            norm = norm * attention_mask
            self.assertTrue(tf.reduce_all(norm < self.eps).numpy(), f'i={i}, norm = {norm}')
            norm = average_norm(output['hidden_states'][i], axis = -1)
            self.assertTrue(tf.reduce_all(norm > self.eps).numpy(), norm)


    def test_LMHead(self):
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lm', name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)

        text = u'????????????[MASK]??????'
        tokenizer = BertTokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = False)
        hf_logits = higgingface_lm_model_output(text)

        norm = average_norm(output['logits'] - hf_logits, axis = -1)
        self.assertTrue(tf.reduce_all(norm < 1e-2).numpy(), f'norm = {norm}')


    def test_causal_lm(self):
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lm', causal_attention = True, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)

        text = u'?????????????????????'
        tokenizer = BertTokenizer(cn_vocab_path)

        inputs = tokenizer([text], return_np = True, return_dict = True)
        output = bert(inputs, output_hidden_states = False)

        tokens = logits_to_tokens(output['logits'], tokenizer, topk = 5)
        self.assertEqual(tokens.shape, (inputs['input_ids'].shape[1], 5))

    def test_causal_attention(self):
        config = ModelConfig(path = self.hf_config_path)
        bert = HuggingFaceBertModel(config = config, model_head = 'lm', causal_attention = True, name = 'bert')
        bert.load_checkpoint(self.h5file_path, silent = True)

        text = u'?????????????????????'
        tokenizer = BertTokenizer(cn_vocab_path)
        input_size = len(tokenizer.tokenize(text)) + 1

        maxlen = 20

        inputs = tokenizer([text], return_np = True, return_dict = True, maxlen = maxlen)
        output = bert(inputs, output_hidden_states = False)['logits']

        seq_len = output.shape[1]

        text2 = u'?????????????????????????????????'
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

        text1 = u'??????????????????????????????????????????????????????????????????????????????????????????????????????????????????'
        text2 = u'1915???????????????????????????????????????????????????????????????????????????????????????????????'
        tokenizer = BertTokenizer(cn_vocab_path)
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

        tokenizer = BertTokenizer(en_cased_vocab_path)
        text = "Train the model for three epochs."
        model_inputs = tokenizer([text], return_np = True, return_dict = True, maxlen = input_dim)
        model_out = model(model_inputs)
        self.assertEqual(model_out.shape, model_inputs['input_ids'].shape + (tokenizer.vocab_size,))

    def test_save_load_weights(self):
        tokenizer = BertTokenizer(en_cased_vocab_path, cased = True)
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
    tokenizer = HFBertTokenizer.from_pretrained('bert-base-chinese')
    model = TFBertModel.from_pretrained('bert-base-chinese')
    if text2 is None:
        inputs = tokenizer([text1], return_tensors = 'np')
    else:
        inputs = tokenizer([text1], [text2], padding = 'max_length', max_length = maxlen, return_tensors = 'np')

    output = model(inputs, output_hidden_states = True)
    return output.hidden_states

def higgingface_lm_model_output(text):
    tokenizer = HFBertTokenizer.from_pretrained('bert-base-chinese')
    model = TFBertForMaskedLM.from_pretrained('bert-base-chinese')
    inputs = tokenizer([text], return_tensors = 'np')
    output = model(inputs, output_hidden_states = False)
    return output.logits


def higgingface_pooler_output(texts):
    tokenizer = HFBertTokenizer.from_pretrained('bert-base-chinese')
    model = TFBertModel.from_pretrained('bert-base-chinese')
    inputs = tokenizer(texts, return_tensors = 'tf', max_length = 64, padding = 'max_length')
    output = model(inputs, output_hidden_states = False)
    return output.pooler_output


roberta_config_path = os.path.join(current, './testdata/bert_config.json')
roberta_vocab_path = os.path.join(current, './testdata/roberta-vocab.json')
robert_merges_path = os.path.join(current, './testdata/roberta-merges.txt')
roberta_testdata_path = os.path.join(current, './testdata/testdata_robert_model.json')


def load_roberta_testdata(vocab_path, merges_path, testdata_path = None):
    tokenizer = BpeTokenizer(vocab_path, merges_path)

    if testdata_path is None:
        return tokenizer
    else:
        with open(testdata_path, encoding = 'utf-8') as f:
            testdata = json.load(f)

        return tokenizer, testdata
    
class RobertaModelTestCase(unittest.TestCase):
    def test_load_checkpoint(self):
        cm = CheckpointManager()
        checkpoint_path = cm.get_checkpoint_path('huggingface-roberta-base')
        config_path = cm.get_config_path('huggingface-roberta-base')

        config = ModelConfig(config_path)
        roberta = HuggingFaceRobertaModel(config = config, model_head = ['pooler', 'lm'], name = 'roberta')
        roberta.load_checkpoint(checkpoint_path, silent = True)

    def test_call(self):
        cm = CheckpointManager()
        checkpoint_path = cm.get_checkpoint_path('huggingface-roberta-base')
        config_path = cm.get_config_path('huggingface-roberta-base')

        config = ModelConfig(config_path)
        roberta = HuggingFaceRobertaModel(config = config, model_head = ['pooler', 'lm'], name = 'roberta')
        roberta.load_checkpoint(checkpoint_path, silent = True)

        tokenizer, testdatas = load_roberta_testdata(roberta_vocab_path, robert_merges_path, roberta_testdata_path)

        for testdata in testdatas:
            inputs = tokenizer([testdata['text']])
            output = roberta(inputs)
            expected = np.array(testdata['last_hidden_state'], dtype = 'float32')
            actual = output['sequence_output'].numpy()
            
            diff_norm = average_norm(actual - expected, axis = -1)[0]
            expected_norm = average_norm(expected, axis = -1)[0]

            bool_res = tf.less(diff_norm, 1e-2)
            self.assertTrue(tf.reduce_all(bool_res).numpy())
        
        
        
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
    
    suite.addTest(RobertaModelTestCase('test_load_checkpoint'))
    suite.addTest(RobertaModelTestCase('test_call'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

        
