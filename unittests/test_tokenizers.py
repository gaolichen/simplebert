#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_tokenizers.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import unittest
import numpy as np
import json
import copy

import sys
import os
  
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

from simplebert import tokenizer_from_pretrained
from simplebert.tokenizers import BertTokenizer, BpeTokenizer

en_cased_vocab_path = os.path.join(current, './testdata/bert-base-cased-vocab.txt')
testdata_en_cased_path = os.path.join(current, './testdata/testdata_tokenizer_en_cased.json')

en_uncased_vocab_path = os.path.join(current, './testdata/bert-base-uncased-vocab.txt')
testdata_en_uncased_path = os.path.join(current, './testdata/testdata_tokenizer_en_uncased.json')

cn_vocab_path = os.path.join(current, './testdata/bert-base-chinese-vocab.txt')
testdata_cn_path = os.path.join(current, './testdata/testdata_tokenizer_cn.json')


def load_testdata(vocab_path, testdata_path = None, cased = True):
    tokenizer = BertTokenizer(vocab_path, cased = cased)

    if testdata_path is None:
        return tokenizer
    else:
        with open(testdata_path, encoding = 'utf-8') as f:
            testdata = json.load(f)

        return tokenizer, testdata


class BertTokenizerTestCase(unittest.TestCase):

    def _test_constructor(self, tokenizer):
        # test special tokens
        self.assertEqual(tokenizer.pad_token, '[PAD]')
        self.assertEqual(tokenizer.unk_token, '[UNK]')
        self.assertEqual(tokenizer.cls_token, '[CLS]')
        self.assertEqual(tokenizer.sep_token, '[SEP]')
        self.assertEqual(tokenizer.mask_token, '[MASK]')

        # test special token ids
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertTrue(tokenizer.unk_token_id > 0)
        self.assertTrue(tokenizer.cls_token_id > 0)
        self.assertTrue(tokenizer.sep_token_id > 0)
        self.assertTrue(tokenizer.mask_token_id > 0)

        # test special tokens
        self.assertTrue(tokenizer.pad_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.unk_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.cls_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.sep_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.mask_token in tokenizer.special_tokens)
        
    def test_constructor_en_cased(self):
        tokenizer = load_testdata(en_cased_vocab_path)
        self._test_constructor(tokenizer)

    def test_constructor_en_uncased(self):
        tokenizer = load_testdata(en_uncased_vocab_path, cased = False)
        self._test_constructor(tokenizer)

    def test_constructor_cn(self):
        tokenizer = load_testdata(cn_vocab_path)
        self._test_constructor(tokenizer)

    def _test_special_tokens(self, tokenizer):
        tokenizer = BertTokenizer(self.vocab_path)
        special_tokens = [tokenizer.pad_token,
                          tokenizer.unk_token,
                          tokenizer.cls_token,
                          tokenizer.sep_token,
                          tokenizer.mask_token]
        
        special_token_ids = [tokenizer.pad_token_id,
                          tokenizer.unk_token_id,
                          tokenizer.cls_token_id,
                          tokenizer.sep_token_id,
                          tokenizer.mask_token_id]

        self.assertEqual(tokenizer.tokens_to_ids(special_tokens), special_token_ids)
        self.assertEqual(tokenizer.ids_to_tokens(special_token_ids), special_tokens)

    def test_special_tokens_cn(self):
        tokenizer = load_testdata(cn_vocab_path)
        self._test_constructor(tokenizer)

    def test_special_tokens_en_cased(self):
        tokenizer = load_testdata(en_cased_vocab_path)
        self._test_constructor(tokenizer)

    def test_special_tokens_en_uncased(self):
        tokenizer = load_testdata(en_uncased_vocab_path, cased = False)
        self._test_constructor(tokenizer)
        
    def _test_tokenize(self, tokenizer, testdatas):
        for testdata in testdatas:
            tokens = tokenizer.tokenize(testdata['text'], replace_unknown_token = True)
            self.assertEqual(tokens, testdata['tokens'])
            #decoded = testdata['decode'][len(tokenizer.cls_token):-len(tokenizer.sep_token)].strip()
            #self.assertEqual(tokenizer.tokens_to_text(tokens), testdata['text'])

    def test_tokenize_en_cased(self):
        tokenizer, testdata = load_testdata(en_cased_vocab_path, testdata_en_cased_path)
        self._test_tokenize(tokenizer, testdata)

    def test_tokenize_en_uncased(self):
        tokenizer, testdata = load_testdata(en_uncased_vocab_path, testdata_en_uncased_path, cased = False)
        self._test_tokenize(tokenizer, testdata)

    def test_tokenize_cn(self):
        tokenizer, testdata = load_testdata(cn_vocab_path, testdata_cn_path)
        self._test_tokenize(tokenizer, testdata)
            
    def test_tokenize_cjk(self):
        tokenizer = load_testdata(cn_vocab_path)
        text = u'???????????????????????????????????????????????????????????????????????????'
        tokens = tokenizer.tokenize(text, replace_unknown_token = False)

        self.assertEqual(len(tokens), len(text))
        self.assertEqual(tokenizer.tokens_to_text(tokens), text)


    def test_tokenize_cjk_en_mix(self):
        tokenizer = load_testdata(cn_vocab_path)
        text_cn = u'???????????????????????????????????????????????????????????????????????????????????????????????????????????????'
        text_en = "You'll need to keep any subtypes in mind if you've opted for this approach."
        text = text_en + text_cn
        tokens = tokenizer.tokenize(text)

        self.assertEqual(tokenizer.tokens_to_text(tokens), text)

    def test_tokenize_special_tokens_cn(self):
        tokenizer = load_testdata(cn_vocab_path)
        for tok in tokenizer.special_tokens:
            text = f'?????????????????????????????????????????????{tok}?????????????????????'
            tokens = tokenizer.tokenize(text, replace_unknown_token = False)
            self.assertTrue(tok in tokens, f'tok={tok}, tokens={tokens}')

    def test_tokenize_special_tokens_en(self):
        tokenizer = load_testdata(en_uncased_vocab_path)
        for tok in tokenizer.special_tokens:
            text = f"You'll need to keep any subtypes in {tok} if you've opted for this approach."
            tokens = tokenizer.tokenize(text, replace_unknown_token = False)
            self.assertTrue(tok in tokens, f'tok={tok}, tokens={tokens}')
            

    def _test_encode(self, tokenizer, testdatas):
        for testdata in testdatas:
            input_ids = tokenizer.encode(testdata['text'])
            self.assertEqual(input_ids, testdata['token_ids'])

            input_ids = tokenizer.encode([testdata['text']])
            self.assertEqual(input_ids, [testdata['token_ids']])

            input_ids = tokenizer.encode(testdata['text'], maxlen = 20).tolist()
            self.assertEqual(input_ids, testdata['token_ids'][:20])

            for second_data in testdatas:
                input_ids = tokenizer.encode(testdata['text'], second_data['text'])
                expected = testdata['token_ids'] + second_data['token_ids'][1:]
                self.assertEqual(input_ids, expected)
            
    def test_encode_cn(self):
        tokenizer, testdata = load_testdata(cn_vocab_path, testdata_cn_path)
        self._test_encode(tokenizer, testdata)

    def test_encode_en_cased(self):
        tokenizer, testdata = load_testdata(en_cased_vocab_path, testdata_en_cased_path)
        self._test_encode(tokenizer, testdata)

    def test_encode_en_uncased(self):
        tokenizer, testdata = load_testdata(en_uncased_vocab_path, testdata_en_uncased_path, cased = False)
        self._test_encode(tokenizer, testdata)

    def _test_decode(self, tokenizer, testdatas, cased = False):
        for testdata in testdatas:
            input_ids = testdata['token_ids']
            text = tokenizer.decode(input_ids)
            expected = testdata['text'] if cased else testdata['text'].lower()

            # since the tokenizer cannot distinguish period from floating point,
            # we need special handling to compare the results.
            text2 = [t.strip() for t in text.split('.')]
            expected2 = [t.strip() for t in expected.split('.')]
            self.assertEqual(text2, expected2)

            texts = tokenizer.decode([input_ids])
            self.assertEqual(texts, [text])

    def test_decode_cn(self):
        tokenizer, testdata = load_testdata(cn_vocab_path, testdata_cn_path)
        self._test_decode(tokenizer, testdata, cased = False)

    def test_encode_en_cased(self):
        tokenizer, testdata = load_testdata(en_cased_vocab_path, testdata_en_cased_path)
        self._test_decode(tokenizer, testdata, cased = True)

    def test_encode_en_uncased(self):
        tokenizer, testdata = load_testdata(en_uncased_vocab_path, testdata_en_uncased_path, cased = False)
        self._test_decode(tokenizer, testdata, cased = False)

    def _test_call(self, tokenizer, testdatas):
        for testdata in testdatas:
            # simple test.
            input_ids, token_type_ids, attention_mask = tokenizer(testdata['text'], return_dict = False, return_np = False)
            self.assertEqual(token_type_ids[0], [0] * len(input_ids[0]))
            self.assertEqual(attention_mask[0], [1] * len(input_ids[0]))
            self.assertEqual(input_ids[0], testdata['token_ids'])

            # test return dict
            maxlen = 30
            inputs = tokenizer(testdata['text'], return_dict = True, maxlen = maxlen, return_np = False)
            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            attention_mask = inputs['attention_mask']

            self.assertEqual(token_type_ids[0], [0] * len(input_ids[0]))
            expected_mask = [1] * min(maxlen, len(testdata['token_ids'])) + [0] * max(0, maxlen - len(testdata['token_ids']))
            self.assertEqual(attention_mask[0], expected_mask)

            expected_input_ids = copy.copy(testdata['token_ids'])
            if len(expected_input_ids) > maxlen:
                expected_input_ids = expected_input_ids[:maxlen - 1] + expected_input_ids[-1:]
            else:
                expected_input_ids += [tokenizer.pad_token_id] * (maxlen - len(expected_input_ids))

            self.assertEqual(input_ids[0], expected_input_ids)

            # test text pair
            for testdata1 in testdatas:
                for testdata2 in testdatas:
                    maxlen = 60
                    inputs = tokenizer(testdata1['text'], second_text = testdata2['text'],
                                       return_dict = True, maxlen = maxlen, return_np = False)
                    
                    input_ids = inputs['input_ids'][0]
                    token_type_ids = inputs['token_type_ids'][0]
                    attention_mask = inputs['attention_mask'][0]

                    len1 = len(testdata1['token_ids'])
                    len2 = len(testdata2['token_ids'])

                    expected_token_type_ids = [0] * len1 + [1] * min(maxlen - len1, len2 - 1)
                    expected_token_type_ids += [0] * (maxlen - len(expected_token_type_ids))
                    expected_token_type_ids = expected_token_type_ids[:maxlen]

                    #if token_type_ids != expected_token_type_ids:
                    #    print(token_type_ids, expected_token_type_ids)
                        
                    self.assertEqual(token_type_ids, expected_token_type_ids)

                    expected_attention_mask = [1] * (len1 + len2 - 1)
                    expected_attention_mask += [0] * (maxlen - len(expected_attention_mask))
                    expected_attention_mask = expected_attention_mask[:maxlen]
                    self.assertEqual(attention_mask, expected_attention_mask)

                    expected_input_ids = copy.copy(testdata1['token_ids'])
                    expected_input_ids += testdata2['token_ids'][1:]
                    expected_input_ids += [tokenizer.pad_token_id] * (maxlen - len(expected_input_ids))
                    if expected_input_ids[-1] == tokenizer.pad_token_id:
                        expected_input_ids = expected_input_ids[:maxlen]
                    else:
                        expected_input_ids = expected_input_ids[:maxlen - 1] + [tokenizer.sep_token_id]

                    if input_ids != expected_input_ids:
                        print('input_ids=', input_ids)
                        print('expected_input_ids=', expected_input_ids)
                    self.assertEqual(input_ids, expected_input_ids)
                                

    def test_call_en_cased(self):
        tokenizer, testdata = load_testdata(en_cased_vocab_path, testdata_en_cased_path)
        self._test_call(tokenizer, testdata)

    def test_call_en_uncased(self):
        tokenizer, testdata = load_testdata(en_uncased_vocab_path, testdata_en_uncased_path, cased = False)
        self._test_call(tokenizer, testdata)

    def test_call_cn(self):
        tokenizer, testdata = load_testdata(cn_vocab_path, testdata_cn_path)
        self._test_call(tokenizer, testdata)

    def test_call_return_np(self):
        tokenizer = load_testdata(cn_vocab_path)
        text1 = u'?????????????????????????????????????????????????????????????????????'
        text2 = u'?????????????????????'
        maxlen = max(len(text1), len(text2)) + 2

        inputs = tokenizer([text1, text2], return_np = True)

        self.assertEqual(inputs['input_ids'].shape, (2, maxlen))
        self.assertEqual(inputs['token_type_ids'].shape, (2, maxlen))
        self.assertEqual(inputs['attention_mask'].shape, (2, maxlen))

    def test_from_pretrained(self):
        tokenizer = tokenizer_from_pretrained(model_name = 'bert-base-chinese')
        self.assertFalse(tokenizer.cased)

roberta_vocab_path = os.path.join(current, './testdata/roberta-vocab.json')
roberta_merges_path = os.path.join(current, './testdata/roberta-merges.txt')
roberta_testdata_path = os.path.join(current, './testdata/testdata_bpe_tokenizer.json')


def load_roberta_testdata(vocab_path, merges_path, testdata_path = None):
    tokenizer = BpeTokenizer(vocab_path, merges_path)

    if testdata_path is None:
        return tokenizer
    else:
        with open(testdata_path, encoding = 'utf-8') as f:
            testdata = json.load(f)

        return tokenizer, testdata

class BpeTokenizerTestCase(unittest.TestCase):
    def test_constructor(self):
        tokenizer = load_roberta_testdata(roberta_vocab_path, roberta_merges_path)

        # test special tokens
        self.assertEqual(tokenizer.pad_token, '<pad>')
        self.assertEqual(tokenizer.unk_token, '<unk>')
        self.assertEqual(tokenizer.cls_token, '<s>')
        self.assertEqual(tokenizer.sep_token, '</s>')
        self.assertEqual(tokenizer.mask_token, '<mask>')

        # test special tokens
        self.assertTrue(tokenizer.pad_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.unk_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.cls_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.sep_token in tokenizer.special_tokens)
        self.assertTrue(tokenizer.mask_token in tokenizer.special_tokens)
        
    def test_tokenize(self):
        tokenizer, testdatas = load_roberta_testdata(roberta_vocab_path, roberta_merges_path, roberta_testdata_path)
        for testdata in testdatas:
            tokens = tokenizer.tokenize(testdata['text'], replace_unknown_token = True)
            self.assertEqual(tokens, testdata['tokens'])
            #self.assertEqual(tokenizer.tokens_to_text(tokens), testdata['text'])

    def test_encode(self):
        tokenizer, testdatas = load_roberta_testdata(roberta_vocab_path, roberta_merges_path, roberta_testdata_path)
        
        for testdata in testdatas:
            input_ids = tokenizer.encode(testdata['text'])
            self.assertEqual(input_ids, testdata['token_ids'], testdata['text'])

            input_ids = tokenizer.encode([testdata['text']])
            self.assertEqual(input_ids, [testdata['token_ids']])

            input_ids = tokenizer.encode(testdata['text'], maxlen = 19).tolist()
            self.assertEqual(input_ids, testdata['token_ids'][:19])

            for second_data in testdatas:
                input_ids = tokenizer.encode(testdata['text'], second_data['text'])
                expected = testdata['token_ids'] + second_data['token_ids']
                self.assertEqual(input_ids, expected)


def suite():
    suite = unittest.TestSuite()
    
    suite.addTest(BertTokenizerTestCase('test_constructor_en_cased'))
    suite.addTest(BertTokenizerTestCase('test_constructor_en_uncased'))
    suite.addTest(BertTokenizerTestCase('test_constructor_cn'))

    suite.addTest(BertTokenizerTestCase('test_special_tokens_en_uncased'))
    suite.addTest(BertTokenizerTestCase('test_special_tokens_en_cased'))
    suite.addTest(BertTokenizerTestCase('test_special_tokens_cn'))
    
    suite.addTest(BertTokenizerTestCase('test_tokenize_en_uncased'))
    suite.addTest(BertTokenizerTestCase('test_tokenize_en_cased'))
    suite.addTest(BertTokenizerTestCase('test_tokenize_cn'))
    

    suite.addTest(BertTokenizerTestCase('test_tokenize_cjk'))
    suite.addTest(BertTokenizerTestCase('test_tokenize_cjk_en_mix'))
    suite.addTest(BertTokenizerTestCase('test_tokenize_special_tokens_cn'))
    suite.addTest(BertTokenizerTestCase('test_tokenize_special_tokens_en'))

    suite.addTest(BertTokenizerTestCase('test_encode_cn'))
    suite.addTest(BertTokenizerTestCase('test_encode_en_cased'))
    suite.addTest(BertTokenizerTestCase('test_encode_en_uncased'))

    suite.addTest(BertTokenizerTestCase('test_call_en_cased'))
    suite.addTest(BertTokenizerTestCase('test_call_en_uncased'))
    suite.addTest(BertTokenizerTestCase('test_call_cn'))
    
    suite.addTest(BertTokenizerTestCase('test_call_return_np'))
    suite.addTest(BertTokenizerTestCase('test_from_pretrained'))

    suite.addTest(BpeTokenizerTestCase('test_constructor'))
    suite.addTest(BpeTokenizerTestCase('test_tokenize'))
    suite.addTest(BpeTokenizerTestCase('test_encode'))
    
    
    #suite = unittest.defaultTestLoader.loadTestsFromTestCase(TokenizerTestCase)
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

