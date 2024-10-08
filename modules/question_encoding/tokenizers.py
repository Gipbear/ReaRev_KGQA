import re
import numpy as np

class LSTMTokenizer():
    def __init__(self, word2id, max_query_word):
        super().__init__()
        self.word2id = word2id
        self.max_query_word = max_query_word

    def tokenize(self, question):
        tokens = self.tokenize_sent(question)
        query_text = np.full(self.max_query_word, len(self.word2id), dtype=int)
        #tokens = question.split()
        #if self.data_type == "train":
        #    random.shuffle(tokens)
        for j, word in enumerate(tokens):
            if j < self.max_query_word:
                    if word in self.word2id:
                        query_text[j] = self.word2id[word]
                        
            else:
                query_text[j] = len(self.word2id)

        return query_text

    @staticmethod
    def tokenize_sent(question_text):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        toks = enumerate(question_text.split(' '))
        
        for w_idx, w in toks:
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words
