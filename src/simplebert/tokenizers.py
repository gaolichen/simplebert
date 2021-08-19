#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : tokenizers.py
# @Author  : Gaoli Chen
# @Time    : 2021/07/09
# @Desc    :

import os
import json
import numpy as np
import regex as re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections.abc import Iterable
from simplebert import utils
from simplebert.pretrained import CheckpointManager

class TokenizerBase(object):
    """Tokenizer
    """
    def __init__(self,
                 vocab_path,
                 cls_token,
                 sep_token,
                 pad_token,
                 unk_token,
                 mask_token,
                 cased = True,
                 paired_tag = False):
        super(TokenizerBase, self).__init__()
            
        self.cased = cased
        self.paired_tag = paired_tag

        self.pad_token =  pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

        # for BERT the left and right brackets are [ and ]
        # for RoBert and GPT2 they are < and >
        self.left_bracket = pad_token[0]
        self.right_bracket = pad_token[-1]
        
        self._token_id_dict = {}
        with open(vocab_path, encoding='utf-8') as f:
            if vocab_path.endswith('.json'):
                self._token_id_dict = json.load(f)
            else:
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
        # split text into words
        words = self._whitespace_tokenize(text)
        
        # perform word piece tokenization
        tokens = []
        for word in words:
            tokens.extend(self._wordpiece_tokenize(word, replace_unknown_token))
                
        return tokens

    def _whitespace_tokenize(self, text):
        raise NotImplementedError

    def _wordpiece_tokenize(self, word, replace_unknown_token):
        raise NotImplementedError

    def tokens_to_text(self, tokens):
        if not tokens:
            return ''

        if isinstance(tokens, (list, tuple)) and isinstance(tokens[0], (list, tuple)):
            return [self.tokens_to_text(tok_list) for tok_list in tokens]

        return self._tokens_to_text(tokens)
    

    def _tokens_to_text(self, tokens):
        raise NotImplementedError

    def encode(self, first_text, second_text = None, maxlen = None):
        if second_text is not None:
            len1 = 1 if isinstance(first_text, str) else len(first_text)
            len2 = 1 if isinstance(second_text, str) else len(second_text)

            if len1 != len2:
                raise ValueError(f'The length of first_text = {len1} but the length of second_text is {len2}')
        
        input_ids, _= self._to_input_ids(first_text)

        if second_text:
            input_ids2, _ = self._to_input_ids(second_text)
            if not self.paired_tag:
                input_ids[0] += input_ids2[0][1:]
            else:
                input_ids[0] += input_ids2[0]

        if not maxlen is None:
            input_ids = pad_sequences(input_ids, maxlen = maxlen, padding = 'post',
                              truncating = 'post', value = self.pad_token_id)

        if isinstance(first_text, str):
            return input_ids[0]
        else:
            return input_ids
            

    def __call__(self, first_text, second_text = None,
                 maxlen = None, return_np = True, return_dict = True):
        """Generates inputs for transformer model.

            Args:
                first_text:
                    The text or a list of text to be processed.
                second_text:
                    The second text or a list of text to be processed. Used for task with sentence pair inputs.
                maxlen:
                    If set, returned lists are truncated or padded to `maxlen`.
                return_np:
                    If True, returned results are numpy arrays, otherwise returned values are python lists.
                return_dict:
                    If True, the method returns a dict object, otherwise returns a tuple.

            Returns:
                If `return_dict` is `True`, it returns dict of the form
                {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}.
                If `return_dict` is `False`, it returns a tuple (input_ids, token_type_ids, attention_mask).
                If `return_np` is `True`, all lists in the results are numpy arrays. Otherwise, all lists are
                python lists.
        """
        
        input_ids, segment_ids = self._to_input_ids(first_text, segment_id = 0)

        if second_text is not None:
            input_ids2, segment_ids2 = self._to_input_ids(second_text, segment_id = 1)

            if len(input_ids2) != len(input_ids):
                raise ValueError(f'the number of texts in the two arguments should be the same.')
            
            for i in range(len(input_ids2)):
                if not self.paired_tag:
                    input_ids[i] += input_ids2[i][1:]
                    segment_ids[i] += segment_ids2[i][1:]
                else:
                    input_ids[i] += input_ids2[i]
                    segment_ids[i] += segment_ids2[i]

        # build attention mask
        attention_mask = []
        for id_list in input_ids:
            attention_mask.append([1] * len(id_list))

        if maxlen is None and return_np:
            maxlen = max([len(l) for l in input_ids])

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
        tokens = self.ids_to_tokens(input_ids)
        return self.tokens_to_text(tokens)
    

    def tokens_to_ids(self, tokens):
        return [self._token_id_dict[tok] if tok in self._token_id_dict else self.unk_token_id for tok in tokens]


    def ids_to_tokens(self, ids):
        return [self.ids_to_tokens(id) if isinstance(id, Iterable) else self._id_token_dict[id] for id in ids]

    def logits_to_ids(self, logits, topk = 5, flatten = False):
        prob = tf.nn.softmax(logits, axis = -1)
        indices = tf.argsort(prob, axis = -1, direction = 'DESCENDING')

        indices = indices[:, :, :topk]
        if flatten:
            return np.ravel(indices.numpy())
        else:
            return indices

    def logits_to_tokens(self, logits, topk = 5):
        input_shape = logits.shape[:-1]
        token_ids = self.logits_to_ids(logits, topk = topk, flatten = True)
        tokens = self.ids_to_tokens(token_ids)
        return np.reshape(np.array(tokens), input_shape + (topk,))
        

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

class BertTokenizer(TokenizerBase):
    
    def __init__(self, vocab_path, cased = True):
        super(BertTokenizer, self).__init__(vocab_path,
                                            pad_token = '[PAD]',
                                            unk_token = '[UNK]',
                                            cls_token = '[CLS]',
                                            sep_token = '[SEP]',
                                            mask_token = '[MASK]',
                                            cased = cased)


    def _whitespace_tokenize(self, text):
        words = []
        # first split text into words
        word = ''

        # add one whitespace in the end
        text += ' '

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
            elif ch == self.right_bracket:
                if len(words) >= 1 and words[-1] == self.left_bracket and \
                   self.left_bracket + word.upper() + self.right_bracket in self.special_tokens:
                    # this is a special token
                    words[-1] = self.left_bracket + word.upper() + self.right_bracket
                    word = ''
                else:
                    add_word()
                    words.append(ch)
                
            elif utils.is_punctuation(ch):
                # true punctuation
                add_word()
                words.append(ch)
            else:
                word += ch if self.cased else ch.lower()

        return words

    def _wordpiece_tokenize(self, word, replace_unknown_token):
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

    def _tokens_to_text(self, tokens):
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



class BpeTokenizer(TokenizerBase):
    """
    Tokenizer for RoBERT and GPT2 models.
    The source code of the class is primary from open AI and huggingface transformers
    """
    def __init__(self, vocab_path, merge_path, cased = True):
        super(BpeTokenizer, self).__init__(vocab_path,
                                               pad_token = '<pad>',
                                               unk_token = '<unk>',
                                               cls_token = '<s>',
                                               sep_token = '</s>',
                                               mask_token = '<mask>',
                                               cased = cased,
                                               paired_tag = True)

        with open(merge_path, encoding = 'utf-8') as f:
            bpe_pairs = [tuple(line.split()) for line in f.readlines()[1:-1]]

        self.bpe_ranks = dict(zip(bpe_pairs, range(len(bpe_pairs))))

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        self.cache = {}
        self.errors = "replace"
        
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _whitespace_tokenize(self, text):
        bra, start = 0, 0
        positions = []
        
        for i, ch in enumerate(text):
            if ch == self.left_bracket:
                bra = 1
                start = i
            elif ch == self.right_bracket and bra == 1:
                if text[start:i + 1] in self.special_tokens:
                    positions.append((start, i + 1))
                bra = 0

        if not positions:
            return re.findall(self.pat, text)
        else:
            last_pos = 0
            tokens = []
            for pos in positions:
                pre_text = text[last_pos:pos[0]].rstrip()
                if len(pre_text):
                    tokens.extend(re.findall(self.pat, pre_text))
                tokens.append(text[pos[0]:pos[1]])
                last_pos = pos[1]

            if last_pos < len(text):
                tokens.extend(re.findall(self.pat, text[last_pos:].rstrip()))
            return tokens
            

    def _wordpiece_tokenize(self, word, replace_unknown_token):
        if word in self.special_tokens:
            return [word]
        
        bpe_tokens = []
        token = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
        bpe_tokens.extend(bpe_token for bpe_token in self._byte_pair_encode(token).split(" "))

        return bpe_tokens

    def _tokens_to_text(self, tokens):
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors = self.errors)
        return text


    def _byte_pair_encode(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    @staticmethod
    def _get_pairs(word):
        """
        Return set of symbol pairs in a word.

        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs


    @staticmethod
    def _bytes_to_unicode():
        """
        Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
        characters the bpe code barfs on.

        The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
        if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
        decent coverage. This is a signficant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
        tables between utf-8 bytes and unicode strings.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))


if __name__ == '__main__':
    pass
