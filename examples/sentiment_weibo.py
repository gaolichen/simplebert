#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : sentiment_weibo.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/19
# @Desc    :

import sys
import os
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras

if os.environ.get('SIMPLEBERT_LOCAL_SOURCE', '') == "1":
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    simple_bert_dir=os.path.join(parent, 'src')
    sys.path.append(simple_bert_dir)

from simplebert import tokenizer_from_pretrained, model_from_pretrained

# download data from https://github.com/ymcui/Chinese-BERT-wwm/blob/master/data/weibo/weibo.zip
# and unzip the file to directory {data_root}.
data_root = r'/path/to/data/directory'

@dataclass
class Settings(object):
    epochs: int = 1
    batch_size: int = 32
    input_dim: int = 128
    learning_rate: float = 2e-5
    
    model_name: str = 'bert-base-chinese'
    train_path = os.path.join(data_root, 'train.csv')
    dev_path = os.path.join(data_root, 'dev.csv')
    test_path = os.path.join(data_root, 'test.csv')
    

class WeiboDataGenerator(keras.utils.Sequence):
  def __init__(self, df, input_dim, batch_size, tokenizer):
    self.input_dim = input_dim
    self.df = df
    self.indices = list(range(df.shape[0]))
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    
    self.on_epoch_end()

  def __len__(self):
    return (len(self.indices) + self.batch_size - 1) // self.batch_size
  
  def __getitem__(self, index):
    last_pos = min(len(self.indices), (index + 1) * self.batch_size)
    idx = self.indices[index * self.batch_size : last_pos]
    sub_df = self.df.iloc[idx]
    
    input_ids = self.tokenizer.encode(sub_df['review'].values, maxlen = self.input_dim)
    labels = sub_df['label'].values

    attention_mask = (input_ids != 0).astype('int32')
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels
  
  def on_epoch_end(self):
    random.shuffle(self.indices)


def build_model(model_name):
    input_ids = keras.layers.Input(shape=(settings.input_dim,), dtype=tf.int32)
    attention_mask = keras.layers.Input(shape=(settings.input_dim, ), dtype=tf.int32)
    inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

    bert_model = model_from_pretrained(model_name, model_head = 'pooler', name = 'bert', silent = True)
    x = bert_model(inputs)
    
    dense = keras.layers.Dense(units = 1, name = 'classifier')
    output = dense(x['pooler_output'])
    
    model = keras.models.Model(inputs, output, name = 'seti_classifier')
    return model

class ValidCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, valid_step = 10, first_valid_iteration = 10):
        super(ValidCallback, self).__init__()
        
        self.save_path = f'./seti_classifier.ckpt'
        self.val_ds = val_ds
        self.valid_step = valid_step
        self.best = 0.0
        self.best_batch = 0
        self.first_valid_iteration = first_valid_iteration
        self.updated = False
    
    def on_train_batch_end(self, batch, logs=None):
        if batch + 1 >= self.first_valid_iteration * self.valid_step \
            and (batch + 1) % self.valid_step == 0:
            loss, acc = self.model.evaluate(self.val_ds, verbose = 0)
            if acc > self.best:
                print()
                print('val_accuracy=', acc, 'val_loss=', loss)
                self.updated = True
                self.best = acc
                self.best_batch = batch
                self.model.save_weights(self.save_path)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.updated:
            print(f'epoch {epoch} best val_accuracy = {self.best}, best batch = {self.best_batch}')
            self.updated = False

def train(settings):
    train_data = pd.read_csv(settings.train_path)
    dev_data = pd.read_csv(settings.dev_path)
    test_data = pd.read_csv(settings.test_path)

    tokenizer = tokenizer_from_pretrained(settings.model_name)

    train_ds = WeiboDataGenerator(train_data,
                                  input_dim = settings.input_dim,
                                  batch_size = settings.batch_size,
                                  tokenizer = tokenizer)
    
    dev_ds = WeiboDataGenerator(dev_data,
                                  input_dim = settings.input_dim,
                                  batch_size = settings.batch_size,
                                  tokenizer = tokenizer)

    test_ds = WeiboDataGenerator(test_data,
                                  input_dim = settings.input_dim,
                                  batch_size = settings.batch_size,
                                  tokenizer = tokenizer)

    model = build_model(settings.model_name)
    print(model.summary())
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = settings.learning_rate),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy']
                  )
    
    callback = ValidCallback(dev_ds, valid_step = 100)

    model.fit(train_ds, validation_data = dev_ds,
              epochs = settings.epochs, callbacks = [callback])
    
    model.load_weights(callback.save_path)

    test_loss, test_accuracy = model.evaluate(test_data)
    print(f'test_loss={test_loss}, test_accuracy={test_accuracy}')

                                  
if __name__ == '__main__':
    settings = Settings()
    train(settings)
    
