#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

from simplebert.models import BertModel, HuggingFaceBertModel, HuggingFaceRobertaModel, ModelConfig
from simplebert.tokenizers import BertTokenizer, BpeTokenizer
from simplebert.pretrained import CheckpointManager

name_to_class = {}

def register_class(class_key, model_cls, tokenizer_cls):
    name_to_class[class_key] = (model_cls, tokenizer_cls)

register_class('BertModel', BertModel, BertTokenizer)
register_class('HuggingFaceBertModel', HuggingFaceBertModel, BertTokenizer)
register_class('HuggingFaceRobertaModel', HuggingFaceRobertaModel, BpeTokenizer)

def model_from_pretrained(model_name,
                          model_head = None,
                          causal_attention = False,
                          silent = False,
                          **kwargs):
    """
    Creates a model and initializes its weights from a pretrained checkpoint.

    Args:
        model_name:
            Name of the pretrained model.
        model_head:
            The name of head on top of the main layer. Its type can be either `str` or `list[str]` 
        causal_attention:
            Lower triangle attention mask is applied if it is True.
        silent:
            If True, some warming messages are suppressed.
        **kwargs:
            Other

    Returns:
        The pretrained transformer model.

    """
    checkpoint_manager = CheckpointManager()
    class_name = checkpoint_manager.get_class(model_name)
    if not class_name in name_to_class:
        raise ValueError(f'{class_name} is not a valid Transformer class.')
    
    cls = name_to_class[class_name][0]
    config = ModelConfig(checkpoint_manager.get_config_path(model_name))
    model = cls(config = config, model_head = model_head, causal_attention = causal_attention, **kwargs)
    
    checkpoint_path = checkpoint_manager.get_checkpoint_path(model_name)
    model.load_checkpoint(checkpoint_path, silent = silent)
    
    return model

def config_from_pretrained(model_name):
    checkpoint_manager = CheckpointManager()
    return ModelConfig(checkpoint_manager.get_config_path(model_name))


def tokenizer_from_pretrained(model_name):
    """Load tokenizer from pretrained models.

        Args:
            model_name:
                The name of the pretrianed model configured in the pretrain_models.json

        Returns:
            A Tokenizer instance.
    """

    checkpoint_manager = CheckpointManager()
    class_name = checkpoint_manager.get_class(model_name)
    if not class_name in name_to_class:
        raise ValueError(f'{class_name} is not a valid Transformer class.')
    
    cls = name_to_class[class_name][1]    
    checkpoint_manager = CheckpointManager()
    vocab_path = checkpoint_manager.get_vocab_path(model_name)
    merges_path = checkpoint_manager.get_merges_path(model_name)
    if not merges_path:
        return cls(vocab_path, cased = checkpoint_manager.get_cased(model_name))
    else:
        return cls(vocab_path, merges_path, cased = checkpoint_manager.get_cased(model_name))
