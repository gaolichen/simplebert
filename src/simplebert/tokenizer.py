#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from simplebert import utils
from simplebert.checkpoint import checkpoint_manager

class Tokenizer(object):
    def __init__(self, vocab_path, cased = True):
        super(Tokenizer, self).__init__()
            
        self.cased = cased

        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'

        self._token_id_dict = {}
        with open(vocab_path, encoding='utf-8') as f:
            for tok in f.readlines():
                self._token_id_dict[tok.strip()] = len(self._token_id_dict)

        self._id_token_dict = {v : k for k, v in self._token_id_dict.items()}
        self.vocab_size = len(self._token_id_dict)

        self.pad_token_id = self._token_id_dict[self.pad_token]
        self.unk_token_id = self._token_id_dict[self.unk_token]
        self.cls_token_id = self._token_id_dict[self.cls_token]
        self.sep_token_id = self._token_id_dict[self.sep_token]
        self.mask_token_id = self._token_id_dict[self.mask_token]

        self.special_tokens = [self.pad_token,
                               self.unk_token,
                               self.cls_token,
                               self.sep_token,
                               self.mask_token]

        self.special_token_ids = [self.pad_token_id,
                               self.unk_token_id,
                               self.cls_token_id,
                               self.sep_token_id,
                               self.mask_token_id]
        

    def tokenize(self, text, replace_unknown_token = False):
        words = []
        # first split text into words
        word = ''

        # add one whitespace in the end
        text += ' '

        if not self.cased:
            text = text.lower()

        def add_word():
            nonlocal word
            if len(word) > 0:
                words.append(word)
                word = ''
        
        for i, ch in enumerate(text):
            if utils.is_whitespace(ch):
                add_word()
            elif utils.is_cjk_character(ch):
                add_word()
                words.append(ch)
            elif ch == ']':
                if len(words) >= 1 and words[-1] == '[' and '[' + word + ']' in self.special_tokens:
                    # this is a special token
                    words[-1] = '[' + word + ']'
                    word = ''
                else:
                    add_word()
                    words.append(ch)
                
#            elif utils.is_punctuation(ch) and (utils.is_whitespace(text[i+1]) or \
#                                               ch in utils.cjk_punctuation()):
            elif utils.is_punctuation(ch):
                # true punctuation
                add_word()
                words.append(ch)
            else:
                word += ch

        # perform word piece tokenization
        tokens = []
        for word in words:
            tokens.extend(self._tokenize(word, replace_unknown_token))
                
        return tokens

    def _tokenize(self, word, replace_unknown_token):
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            while start < end:
                if start > 0:
                    tok = '##' + word[start:end]
                else:
                    tok = word[start:end]

                if tok in self._token_id_dict:
                    tokens.append(tok)
                    break
                else:
                    end -= 1
            if start < end:
                # find a known token from dictionary
                start = end
            else:
                # did not find a known token
                if replace_unknown_token:
                    return [self.unk_token]
                else:
                    return [word]

        return tokens

    def tokens_to_text(self, tokens):
        if not tokens:
            return ''
        
        text = ''
        for tok in tokens:
            # skip two special tokens.
            #if tok == self.cls_token or tok == self.sep_token or tok == self.pad_token:
            if tok == self.cls_token or tok == self.sep_token or tok == self.pad_token:
                continue
            
            if tok.startswith('##'):
                text += tok[2:]
            else:
                if len(tok) == 1 and (utils.is_punctuation(tok) \
                                       or utils.is_cjk_character(tok)):
                    text += tok
                else:
                    if len(text) > 0 and (text[-1] != "'" and text[-1] != '-'):
                        text += ' ' + tok
                    else:
                        text += tok

        return text.strip()

    def encode(self, first_text, second_text = None):
        input_ids, _= self._to_input_ids(first_text)

        if second_text:
            input_ids2, _ = self._to_input_ids(second_text)
            input_ids[0] += input_ids2[0][1:]

        return input_ids[0]
            

    def __call__(self, first_text, second_text = None,
                 maxlen = None, return_np = False, return_dict = True):
        
        input_ids, segment_ids = self._to_input_ids(first_text, segment_id = 0)

        if second_text is not None:
            input_ids2, segment_ids2 = self._to_input_ids(second_text, segment_id = 1)

            if len(input_ids2) != len(input_ids):
                raise ValueError(f'the number of texts in the two arguments should be the same.')
            
            for i in range(len(input_ids2)):
                input_ids[i] += input_ids2[i][1:]
                segment_ids[i] += segment_ids2[i][1:]

        # build attention mask
        attention_mask = []
        for id_list in input_ids:
            attention_mask.append([1] * len(id_list))

        if maxlen is not None:
            input_ids = self._truncate(input_ids, maxlen, self.pad_token_id, keep_last = True)
            segment_ids = self._truncate(segment_ids, maxlen, 0)
            attention_mask = self._truncate(attention_mask, maxlen, 0)

        if return_np:
            input_ids = np.array(input_ids)
            segment_ids = np.array(segment_ids)
            attention_mask = np.array(attention_mask)

        if return_dict:
            return {'input_ids': input_ids,
                'token_type_ids': segment_ids,
                'attention_mask': attention_mask}
        else:
            return input_ids, segment_ids, attention_mask
        
    def _to_input_ids(self, texts, segment_id = 0):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        segment_ids = []
        for text in texts:
            tokens = self.tokenize(text)
            token_ids = [self.cls_token_id] + self.tokens_to_ids(tokens) + [self.sep_token_id]            
            input_ids.append(token_ids)
            segment_ids.append([segment_id] * len(token_ids))
            
        return input_ids, segment_ids


    def decode(self, input_ids):
        tokens = self.ids_to_tokens(id_list)
        return self.tokens_to_text(tokens)
    

    def tokens_to_ids(self, tokens):
        return [self._token_id_dict[tok] if tok in self._token_id_dict else self.unk_token_id for tok in tokens]


    def ids_to_tokens(self, ids):
        return [self._id_token_dict[id] for id in ids]

    @staticmethod
    def _truncate(input_lists, maxlen, pad_id, keep_last = False):
        ret = []
        for input in input_lists:
            if len(input) > maxlen:
                if keep_last:
                    ret.append(input[:maxlen - 1] + input[-1:])
                else:
                    ret.append(input[:maxlen])
            else:
                ret.append(input + [pad_id] * (maxlen - len(input)))
        return ret

def tokenizer_from_pretrained(model_name):
    vocab_path = checkpoint_manager.get_vocab_path(model_name)
    return Tokenizer(vocab_path, cased = checkpoint_manager.get_cased(model_name))


if __name__ == '__main__':
    mytok = tokenizer_from_pretrained('huggingface-bert-base-cased')
    print(mytok.cased)
