# simplebert
基于tensorflow.keras的各类Transformer模型的简单封装。本项目初衷为供本人学习使用，力求提供最简便的API调用方法，也欢迎有需要的人下载使用。项目编写过程中参考了<a href="https://huggingface.co/transformers/model_doc/bert.html">Huggingface Transformers</a>, <a href="https://nlp.seas.harvard.edu/2018/04/03/attention.html">The Annotated Transformer</a>, 以及<a href="https://github.com/bojone/bert4keras">bert4keras</a>等资料和代码。

## 主要功能
- 支持加载Google原版的BERT模型权重
- 支持加载Huggingface的BERT模型权重
- 支持加载Huggingface的RoBERTa模型权重

## 安装
```shell
pip install simplebert
```

## 使用范例
最简单的调用如下。
```python
from simplebert import tokenizer_from_pretrained, model_from_pretrained

# 选择要加载的模型的名称
model_name = 'bert-base-chinese'                 
# 创建并加载分词器
tokenizer = tokenizer_from_pretrained(model_name)
# 创建并加载模型
# 选择lm (LanguageModelHead)和pooler两种model head
model = model_from_pretrained(model_name, model_head = ['lm', 'pooler'])

# 调用分词器产生输入
inputs = tokenizer([u'为啥科技公司都想养只机器狗？', u'一些公司已经将四足机器人应用在了业务中。'])
# 调用模型产生输出，输出所有层的结果
output = model(inputs, output_hidden_states = True)

# 输出结果
print(output['sequence_output'].shape)    # 最后一层的输出
print(output['logits'].shape)             # 'lm'产生的输出
print(output['pooler_output'].shape)      # 'pooler'产生的输出
print(output['hidden_states'][-2].shape)  # 倒数第二层产生的输出
```

可以选择的模型在`pretrained_models.json`文件中配置。

如果已预先下载了有权重文件到本地，可用如下方式调用。
```python
from simplebert.tokenizers import BertTokenizer
from simplebert.models import ModelConfig, BertModel

config_file = '/path/to/bert_config.json'
vocab_file = '/path/to/vocab.txt'
checkpoint_file = '/path/to/checkpoint.ckp'

tokenizer = BertTokenizer(config_file, cased = True)
config = ModelConfig(config_file)
model = BertModel(config, model_head = 'lm')
#...

```
更多用法，参考[Examples](https://github.com/gaolichen/simplebert/tree/main/examples)目录。

## 支持的模型权重
- Google原版[BERT](https://github.com/google-research/bert) 包括：bert-base-uncased, bert-base-cased, bert-base-chinese, bert-base-cased-multi-lang, bert-large-uncased, bert-large-cased, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking
- [Huggingface的BERT](https://huggingface.co/transformers/model_doc/bert.html)模型, 权重名称包括：huggingface-bert-base-cased, huggingface-bert-base-uncased, huggingface-bert-large-uncased, huggingface-bert-large-cased, huggingface-bert-base-chinese
- [Huggingface的RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)模型, 权重名称包括：huggingface-roberta-base, huggingface-roberta-large




