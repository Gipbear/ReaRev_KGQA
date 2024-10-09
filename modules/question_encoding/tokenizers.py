import re

import numpy as np
from transformers import AutoTokenizer


class BaseTokenizer:
    def __init__(self, max_query_word, num_kb_relation=None, max_rel_words=None):
        self.max_query_word = max_query_word
        self.num_kb_relation = num_kb_relation
        self.max_rel_words = max_rel_words

    def encode_string(self, question: str):
        raise NotImplementedError

    def encode_relation(self, rel_words: list[str]):
        raise NotImplementedError


class LSTMTokenizer(BaseTokenizer):
    def __init__(self, word2id, max_query_word, num_kb_relation=None, max_rel_words=None):
        super().__init__(max_query_word, num_kb_relation, max_rel_words)
        self.word2id = word2id

    def encode_string(self, question: str):
        tokens = self.tokenize_sent(question)
        query_text = np.full(self.max_query_word, len(self.word2id), dtype=int)
        for j, word in enumerate(tokens):
            if j < self.max_query_word:
                if word in self.word2id:
                    query_text[j] = self.word2id[word]
            else:
                query_text[j] = len(self.word2id)
        return query_text

    @staticmethod
    def tokenize_sent(question_text: str):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        for w in question_text.split(' '):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)  # 清理非数字和字母
            if w == '':
                continue
            words += [w]
        return words

    def encode_relation(self, rel_words: list[str]):
        rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
        rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
        for rel_id, tokens in enumerate(rel_words):
            for j, word in enumerate(tokens):
                if j < self.max_rel_words:
                    if word in self.word2id:
                        rel_texts[rel_id, j] = self.word2id[word]
                        rel_texts_inv[rel_id, j] = self.word2id[word]
                    else:
                        rel_texts[rel_id, j] = len(self.word2id)
                        rel_texts_inv[rel_id, j] = len(self.word2id)
        return rel_texts, rel_texts_inv


class BERTsTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_name, max_query_word, num_kb_relation=None, max_rel_words=None):
        super().__init__(max_query_word, num_kb_relation, max_rel_words)
        if tokenizer_name == 'bert':
            tokenizer_name = 'bert-base-uncased'
        elif tokenizer_name == 'roberta':
            tokenizer_name = 'roberta-base'
        elif tokenizer_name == 'sbert':
            tokenizer_name = 'sentence-transformers/all-MiniLM-L6-v2'
        elif tokenizer_name == 'sbert2':
            tokenizer_name = 'sentence-transformers/all-mpnet-base-v2'
        elif tokenizer_name == 't5':
            tokenizer_name = 't5-small'
        else:
            raise NameError("unsupported tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.pad_val = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def encode_string(self, question: str):
        tokens = self.tokenizer.encode_plus(text=question, padding='max_length', truncation=True,
                                            max_length=self.max_query_word, return_attention_mask=False)
        query_text = np.array(tokens['input_ids'])
        return query_text

    def encode_relation(self, rel_words: list[str]):
        rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), self.pad_val, dtype=int)
        rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), self.pad_val, dtype=int)

        for rel_id, words in enumerate(rel_words):
            tokens = self.tokenizer.encode_plus(text=' '.join(words), padding='max_length', truncation=True,
                                                max_length=self.max_rel_words, return_attention_mask=False)
            tokens_inv = self.tokenizer.encode_plus(text=' '.join(words[::-1]), padding='max_length', truncation=True,
                                                    max_length=self.max_rel_words, return_attention_mask=False)
            rel_texts[rel_id] = np.array(tokens['input_ids'])
            rel_texts_inv[rel_id] = np.array(tokens_inv['input_ids'])
        return rel_texts, rel_texts_inv
