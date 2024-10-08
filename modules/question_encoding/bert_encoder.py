import os
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer #DistilBertModel, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
# import warnings
# warnings.filterwarnings("ignore")
# try:
#     os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/costas/home/mavro016/.cache'
# except:
#     pass

from .base_encoder import BaseInstruction


class BERTInstruction(BaseInstruction):

    def __init__(self, args, model):
        super().__init__(args)
        
        entity_dim = self.entity_dim
        self.model = model
        if model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.pretrained_weights = 'bert-base-uncased'
        elif model == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            self.pretrained_weights = 'roberta-base'
        elif model == 'sbert':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.pretrained_weights = 'sentence-transformers/all-MiniLM-L6-v2'
        elif model == 'sbert2':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.pretrained_weights = 'sentence-transformers/all-mpnet-base-v2'
        elif model == 't5':
            self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
            self.pretrained_weights = 't5-small'
        self.pad_val = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        self.question_linear = nn.ModuleList([
            nn.Linear(in_features=entity_dim, out_features=entity_dim)
            for _ in range(self.num_ins)
        ])
        self.encoder_def()

    def encoder_def(self):
        # initialize entity embedding
        self.node_encoder = AutoModel.from_pretrained(self.pretrained_weights)
        self.word_dim = self.node_encoder.config.hidden_size
        print('word_dim', self.word_dim)
        print('Total Params', sum(p.numel() for p in self.node_encoder.parameters()))
        if self.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.node_encoder.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')
        self.question_emb = nn.Linear(in_features=self.word_dim, out_features=self.entity_dim)

    def encode_question(self, query_text, store=True):
        if self.model != 't5':
            query_hidden_emb = self.node_encoder(query_text)[0]  # batch_size, max_query_word, word_dim 问题单词的hidden编码
        else:
            query_hidden_emb = self.node_encoder.encoder(query_text)[0]

        if store:
            self.query_hidden_emb = self.question_emb(query_hidden_emb)  # batch_size, max_query_word, entity_dim
            self.query_node_emb = query_hidden_emb.transpose(1,0)[0].unsqueeze(1)  # 取问句中的第一个词
            #print(self.query_node_emb.size())
            self.query_node_emb = self.question_emb(self.query_node_emb)  # batch_szie, 1, entity_dim
            
            self.query_mask = (query_text != self.pad_val).float()
            return query_hidden_emb, self.query_node_emb
        else:
            return  query_hidden_emb 

