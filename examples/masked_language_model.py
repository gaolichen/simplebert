#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : masked language model.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/13
# @Desc    :

import sys
import os
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
simple_bert_dir=os.path.join(parent, 'src')
sys.path.append(simple_bert_dir)

from simplebert.tokenizers import tokenizer_from_pretrained
from simplebert.models import model_from_pretrained

def language_model_predict(model_name, text):
    
    tokenizer = tokenizer_from_pretrained(model_name)
    model = model_from_pretrained(model_name, model_head = 'lm', silent = True)

    inputs = tokenizer([text])
    output = model(inputs)
    tokens = tokenizer.logits_to_tokens(output['logits'], topk = 1)

    print()
    print(f'=============== {model_name} ===============')
    print('input text:', text)
    print('input_ids:', inputs['input_ids'])
    print('ouput:', np.ravel(tokens)[1:-1])
    

model_names = ['bert-base-chinese', 'huggingface-bert-base-chinese']

text = u'中国的首都是[MASK][MASK]。'
for model_name in model_names:
    language_model_predict(model_name, text)


"""输出:

=============== bert-base-chinese ===============
input text: 中国的首都是[MASK][MASK]。
input_ids: [[ 101  704 1744 4638 7674 6963 3221  103  103  511  102]]
ouput: ['。' '国' '的' '首' '都' '是' '北' '京' '。']

=============== huggingface-bert-base-chinese ===============
input text: 中国的首都是[MASK][MASK]。
input_ids: [[ 101  704 1744 4638 7674 6963 3221  103  103  511  102]]
ouput: ['。' '国' '的' '首' '都' '是' '北' '京' '。']

"""
