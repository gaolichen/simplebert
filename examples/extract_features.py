#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : extract_features.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/13
# @Desc    :

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
simple_bert_dir=os.path.join(parent, 'src')
sys.path.append(simple_bert_dir)

from simplebert import tokenizer_from_pretrained, model_from_pretrained

# 选择要加载的模型的名称
model_name = 'bert-base-chinese'

# 创建并加载分词器
tokenizer = tokenizer_from_pretrained(model_name)

# 创建并加载模型
# 选择lm (LanguageModelHead)和pooler两种model head
model = model_from_pretrained(model_name, model_head = ['lm', 'pooler'], silent = True)

# 调用分词器产生输入
inputs = tokenizer([u'为啥科技公司都想养只机器狗？', u'一些公司已经将四足机器人应用在了业务中。'])

# 调用模型产生输出，输出所有层的结果
output = model(inputs, output_hidden_states = True)

# 输出结果
print(output['sequence_output'].shape)    # 最后一层的输出
print(output['logits'].shape)             # 'lm'产生的输出
print(output['pooler_output'].shape)      # 'pooler'产生的输出
print(output['hidden_states'][-2].shape)  # 倒数第二层产生的输出

"""输出：
(2, 22, 768)
(2, 22, 21128)
(2, 768)
(2, 22, 768)
"""
