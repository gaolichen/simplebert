#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_pretrained.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import unittest
import json

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
  
# now we can import the module in the parent
# directory.

from simplebert.pretrained import CheckpointManager, ModuleConfig

class ModuleConfigTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_constructor(self):
        config = ModuleConfig(os.path.join(current, "testdata", 'simple_bert_config.json'))
        self.assertTrue(len(config.cache_dir) > 0)
        self.assertTrue('bert-base-chinese' in config.model_keys())

    def test_construtor_kwargs(self):
        config = ModuleConfig(os.path.join(current, "testdata", 'simple_bert_config.json'),
                              cache_dir = current)

        self.assertEqual(config.cache_dir, current)

    def test_model_info(self):
        config = ModuleConfig(os.path.join(current, "testdata", 'simple_bert_config.json'))
        model_info = config.model_info('bert-base-chinese')
        self.assertEqual(model_info.name, 'bert-base-chinese')
        self.assertEqual(model_info.vocab_file, 'vocab.txt')
        self.assertEqual(model_info.config_file, 'bert_config.json')
        self.assertEqual(model_info.checkpoint_file, 'bert_model.ckpt')
        self.assertTrue('bert-base-chinese' in config.model_keys())



class CheckpointManagerTestCase(unittest.TestCase):
    
    def get_file_monkey(self, fname, origin,
                        extract = False, cache_subdir = 'datasets',
                        cache_dir = '.'):
        self.cache_root = os.path.join(cache_dir, cache_subdir)
        return os.path.join(self.cache_root, origin.split('/')[-1].split('.')[0])

    
    def setUp(self):
        self.config = ModuleConfig(os.path.join(current, "testdata", 'simple_bert_config.json'),
                                   cache_dir = os.path.join(current, 'testdata'))
        CheckpointManager._get_file = staticmethod(self.get_file_monkey)


    def test_constructor(self):
        model_name = 'bert-base-chinese'
        cm = CheckpointManager(self.config)
        
        config_path = cm.get_config_path(model_name)
        model_path = os.path.join(self.cache_root, model_name)
        
        self.assertEqual(config_path, os.path.join(model_path, 'bert_config.json'))
        self.assertEqual(cm.get_vocab_path(model_name), os.path.join(model_path, 'vocab.txt'))
        self.assertEqual(cm.get_checkpoint_path(model_name), os.path.join(model_path, 'bert_model.ckpt'))
        self.assertFalse(cm.get_cased(model_name))
        

    def test_get_config_path(self):
        model_name = 'bert-base-chinese'
        cm = CheckpointManager(self.config)
    

def suite():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModuleConfigTestCase)
    suite.addTest(CheckpointManagerTestCase('test_constructor'))
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
