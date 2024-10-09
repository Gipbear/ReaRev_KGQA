import os
import time
import json
import warnings
from collections import Counter

import numpy as np
from tqdm import tqdm
from loguru import logger

from modules.question_encoding.tokenizers import LSTMTokenizer, BERTsTokenizer

warnings.filterwarnings("ignore")

try:
    os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/costas/home/mavro016/.cache'
except:
    pass


class BasicDataLoader(object):
    """ 
    Basic Dataloader contains all the functions to read questions and KGs from json files and
    create mappings between global entity ids and local ids that are used during GNN updates.
    """
    def __init__(self, config, word2id, relation2id, entity2id, tokenize_name, data_type="train"):
        self.tokenize_name = tokenize_name
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_file(config, data_type)
        self._load_data()  # 读取 data数据 对 79 行左右的一堆属性进行初始化赋值

    def _parse_args(self, config, word2id, relation2id, entity2id):

        """
        Builds necessary dictionaries and stores arguments.
        1. 从 args 中提取部分参数
        2. 将 xx2id 转换为 id2xx 的形式存储在属性中
        """
        self.data_eff = config['data_eff']  # ?
        self.data_name = config['name']

        if 'use_inverse_relation' in config:
            self.use_inverse_relation = config['use_inverse_relation']
        else:
            self.use_inverse_relation = False
        if 'use_self_loop' in config:
            self.use_self_loop = config['use_self_loop']
        else:
            self.use_self_loop = False

        self.rel_word_emb = config['relation_word_emb']
        self.max_local_entity = 0
        self.max_facts = 0  #最大子图的三元组数量 *2， 自环则加上实体的数量

        logger.info('building word index ...')
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i: entity for entity, i in entity2id.items()}
        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
            self.max_facts = self.max_facts + self.max_local_entity
        logger.info("Entity: {}, Relation in KB: {}, Relation in use: {} ",
                    len(entity2id), len(self.relation2id), self.num_kb_relation)

    def _load_file(self, config, data_type="train"):
        """
        Loads lines (questions + KG subgraphs) from json files.
        ! 由于数据预处理后导致 train_simple.json 的数据量非常大，导致读取文件时小内存(CPU 16G)无法正常运行
        """
        data_file = config['data_folder'] + data_type + "_simple.json"
        self.data_file = data_file
        logger.info('loading data from', data_file)
        # self.data = []
        self.skip_index = set()
        index = 0
        self.max_query_word = 0
        with open(self.data_file) as f_in:
            for line in tqdm(f_in):
                if index == config['max_train'] and data_type == "train": break  #break if we reach max_question_size
                line = json.loads(line)
                if len(line['entities']) == 0:
                    self.skip_index.add(index)
                    continue
                # self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
                self.max_query_word = max(self.max_query_word, line["question"].count(' ')+1)
                index += 1
        logger.warning("skip: {}", self.skip_index)
        logger.info('max_facts: {}', self.max_facts)
        self.num_data = index
        self.batches = np.arange(self.num_data)

    def _load_data(self):
        """
        Creates mappings between global entity ids and local entity ids that are used during GNN updates.
        """
        logger.info('converting global to local entity index ...')
        self.global2local_entity_maps, self.max_local_entity = self.__build_global2local_entity_maps()
        rel_words, self.max_rel_words = self.__collect_rel_words()

        self.num_word, self.tokenizer = self.__init_tokenizer()
        if self.rel_word_emb:
            self.rel_texts, self.rel_texts_inv = self.tokenizer.encode_relation(rel_words)
        else:
            self.rel_texts, self.rel_texts_inv = None, None

        self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)  # 将问题使用 tokenizer 进行编码
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)  # 每个问题的主题词，用 1 标识子图编号
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)  # 每个问题的候选实体，对应 id 使用全局 id
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)  # 每个问题的主题词概率，如果有 k 个，赋值为 1/k
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)  # 局部子图关系列表(head_list, rel_list, tail_list)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)  # 每个问题的答案，用 1 标识子图编号
        self.answer_lists = np.empty(self.num_data, dtype=object)  # 每个问题的答案列表，对应 id 使用全局 id
        self.__prepare_data()

    def __collect_rel_words(self):
        max_rel_words = 0
        rel_words = []
        if 'MetaQA' in self.data_file:
            for rel in self.relation2id:
                words = rel.split('_')
                max_rel_words = max(len(words), max_rel_words)
                rel_words.append(words)
        else:
            for rel in self.relation2id:  # 将关系词语拆分，便于后续使用 tokenizer 进行编码（空格连接后编码）
                rel = rel.strip()
                fields = rel.split('.')
                try:
                    words = fields[-2].split('_') + fields[-1].split('_')  # ? 仅使用部分词作为关系，因该是数据原因
                    max_rel_words = max(len(words), max_rel_words)
                except:
                    words = ['UNK']
                rel_words.append(words)
        assert len(rel_words) == len(self.relation2id)
        return rel_words, max_rel_words

    def __init_tokenizer(self):
        """初始化 tokenizer"""
        logger.info("init tokenizer: {}", self.tokenize_name)
        if self.tokenize_name == 'lstm':
            num_word = len(self.word2id)  # 需要编码的单词数量，lstm 模型中也作为 padding 值
            tokenizer = LSTMTokenizer(self.word2id, self.max_query_word, self.num_kb_relation, self.max_rel_words)
        else:
            self.max_query_word = self.max_query_word + 2  # for cls token and sep
            tokenizer = BERTsTokenizer(self.tokenize_name,
                                       self.max_query_word, self.num_kb_relation,self.max_rel_words)
            num_word = tokenizer.pad_val  # pad 的编码，一般为 0
        return num_word, tokenizer

    def __build_global2local_entity_maps(self):
        """
        Create a map from global entity id to local entity of each sample
        读取 data 属性，并将 entities 转换为 id， 每个 sample（子图） id 从 0 开始编码
        """
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        max_local_entity = self.max_local_entity
        with open(self.data_file) as f_in:
            for line in tqdm(f_in):
                sample = json.loads(line)
                if len(sample['entities']) == 0:
                    continue
                g2l = dict()
                self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
                # construct a map from global entity id to local entity id
                self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)
                global2local_entity_maps[next_id] = g2l
                total_local_entity += len(g2l)
                max_local_entity = max(max_local_entity, len(g2l))
                next_id += 1
        logger.info('avg local entity: {}', total_local_entity / next_id)
        logger.info('max local entity: {}', max_local_entity)
        return global2local_entity_maps, max_local_entity

    def __prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        对问题、答案、实体进行局部子图编码转换
        """
        next_id = 0
        query_entity_nums = []
        with open(self.data_file) as f_in:
            for line in tqdm(f_in):
                sample = json.loads(line)
                if len(sample['entities']) == 0:
                    continue
                g2l = self.global2local_entity_maps[next_id]
                if not g2l:
                    continue
                # 1. 使用 tokenizer 将问题从字符串编码为 token
                self.query_texts[next_id] = self.tokenizer.encode_string(sample['question'])
                # 2 将主题词转换为局部编码
                tp_set = self.__collect_local_ent(g2l, sample['entities'])
                query_entity_nums.append(len(tp_set))
                # 2.1 初始化查询实体矩阵
                self.query_entities[next_id, list(tp_set)] = 1.0
                # 2.2 初始化候选实体矩阵
                for global_entity, local_entity in g2l.items():
                    if self.data_name != 'cwq':
                        # 对应公式 9 后的说明，候选实体不包括主题词
                        if local_entity not in tp_set:  # skip entities in question
                            self.candidate_entities[next_id, local_entity] = global_entity
                    elif self.data_name == 'cwq':
                        self.candidate_entities[next_id, local_entity] = global_entity
                # 2.3 根据主题词初始化实体概率分布
                if len(tp_set) > 0:
                    for local_ent in tp_set:
                        self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
                else:
                    for index in range(len(g2l)):
                        self.seed_distribution[next_id, index] = 1.0 / len(g2l)
                assert np.sum(self.seed_distribution[next_id]) > 0.0
                # 3 将三元组转换为局部 id 存储
                if not self.data_eff:
                    head_list, rel_list, tail_list = self.__collect_local_kg_tuples(g2l, sample['subgraph']['tuples'])
                    self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                                 np.array(rel_list, dtype=int),
                                                 np.array(tail_list, dtype=int))
                # 4 将答案也转换为局部 id 存储
                answer_list, local_answer_list = self.__collect_local_answer(g2l, sample['answers'])
                self.answer_dists[next_id, local_answer_list] = 1.0
                self.answer_lists[next_id] = answer_list
                next_id += 1
        c = Counter(query_entity_nums)
        logger.info("Case with different hops count result: {}", c)

    def __collect_local_ent(self, g2l, entities) -> set:
        tp_set = set()
        for entity in entities:
            try:
                global_entity = self.entity2id[entity['text']]
            except:
                global_entity = entity
            if global_entity not in g2l:
                continue
            local_ent = g2l[global_entity]
            tp_set.add(local_ent)
        return tp_set

    def __collect_local_kg_tuples(self, g2l, tuples):
        head_list, rel_list, tail_list = [], [], []
        for (sbj, rel, obj) in tuples:
            try:
                head = g2l[self.entity2id[sbj['text']]]
                rel = self.relation2id[rel['text']]
                tail = g2l[self.entity2id[obj['text']]]
            except:
                head = g2l[sbj]
                rel = int(rel)
                tail = g2l[obj]
            head_list.append(head)
            rel_list.append(rel)
            tail_list.append(tail)
            if self.use_inverse_relation:
                head_list.append(tail)
                rel_list.append(rel + len(self.relation2id))
                tail_list.append(head)
        return head_list, rel_list, tail_list

    def __collect_local_answer(self, g2l, answers):
        answer_list, local_answer_list = [], []
        for answer in answers:
            keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
            answer_ent = self.entity2id[answer[keyword]]
            answer_list.append(answer_ent)
            if answer_ent in g2l:
                local_answer_list.append(g2l[answer_ent])
        return answer_list, local_answer_list

    def get_quest(self):
        q_list = []
        
        sample_ids = self.sample_ids
        for sample_id in sample_ids:
            tp_str = self.decode_text(self.query_texts[sample_id, :])
            # id2word = self.id2word
            # for i in range(self.max_query_word):
            #     if self.query_texts[sample_id, i] in id2word:
            #         tp_str += id2word[self.query_texts[sample_id, i]] + " "
            q_list.append(tp_str)
        return q_list

    def decode_text(self, np_array_x):
        if self.tokenize_name == 'lstm':
            id2word = self.id2word
            tp_str = ""
            for i in range(self.max_query_word):
                if np_array_x[i] in id2word:
                    tp_str += id2word[np_array_x[i]] + " "
        else:
            tp_str = ""
            words = self.tokenizer.convert_ids_to_tokens(np_array_x)
            for w in words:
                if w not in ['[CLS]', '[SEP]', '[PAD]']:
                    tp_str += w + " "
        return tp_str

    def create_kb_adj_mats(self, sample_id):

        """
        Re-build local adj mats if we have data_eff == True (they are not pre-stored).
        """
        sample = self.data[sample_id]
        g2l = self.global2local_entity_maps[sample_id]
        
        # build connection between question and entities in it
        head_list = []
        rel_list = []
        tail_list = []
        for i, tpl in enumerate(sample['subgraph']['tuples']):
            sbj, rel, obj = tpl
            try:
                head = g2l[self.entity2id[sbj['text']]]
                rel = self.relation2id[rel['text']]
                tail = g2l[self.entity2id[obj['text']]]
            except:
                head = g2l[sbj]
                rel = int(rel)
                tail = g2l[obj]
            head_list.append(head)
            rel_list.append(rel)
            tail_list.append(tail)
            if self.use_inverse_relation:
                head_list.append(tail)
                rel_list.append(rel + len(self.relation2id))
                tail_list.append(head)

        return np.array(head_list, dtype=int),  np.array(rel_list, dtype=int), np.array(tail_list, dtype=int)

    def _build_fact_mat(self, sample_ids, fact_dropout):
        """
        Creates local adj mats that contain entities, relations, and structure.
        """
        batch_heads = np.array([], dtype=int)
        batch_rels = np.array([], dtype=int)
        batch_tails = np.array([], dtype=int)
        batch_ids = np.array([], dtype=int)
        #print(sample_ids)
        for i, sample_id in enumerate(sample_ids):
            index_bias = i * self.max_local_entity
            if self.data_eff:
                head_list, rel_list, tail_list = self.create_kb_adj_mats(sample_id) #kb_adj_mats[sample_id]
            else:
                (head_list, rel_list, tail_list) = self.kb_adj_mats[sample_id]
            num_fact = len(head_list)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]

            real_head_list = head_list[mask_index] + index_bias # 保证同一个 batch 内的局部编码不一致
            real_tail_list = tail_list[mask_index] + index_bias
            real_rel_list = rel_list[mask_index]
            batch_heads = np.append(batch_heads, real_head_list)
            batch_rels = np.append(batch_rels, real_rel_list)
            batch_tails = np.append(batch_tails, real_tail_list)
            batch_ids = np.append(batch_ids, np.full(len(mask_index), i, dtype=int))
            if self.use_self_loop:
                num_ent_now = len(self.global2local_entity_maps[sample_id])
                ent_array = np.array(range(num_ent_now), dtype=int) + index_bias
                rel_array = np.array([self.num_kb_relation - 1] * num_ent_now, dtype=int)
                batch_heads = np.append(batch_heads, ent_array)
                batch_tails = np.append(batch_tails, ent_array)
                batch_rels = np.append(batch_rels, rel_array)
                batch_ids = np.append(batch_ids, np.full(num_ent_now, i, dtype=int))
        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_count = Counter(batch_heads)
        # tail_count = Counter(batch_tails)
        weight_list = [1.0 / head_count[head] for head in batch_heads]
        # entity2fact_index = torch.LongTensor([batch_heads, fact_ids])
        # entity2fact_val = torch.FloatTensor(weight_list)
        # entity2fact_mat = torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
        #     [len(sample_ids) * self.max_local_entity, len(batch_heads)]))
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity_global_id in entities:
            try:
                ent = entity2id[entity_global_id['text']]
                if ent not in g2l:
                    g2l[ent] = len(g2l)
            except:
                if entity_global_id not in g2l:
                    g2l[entity_global_id] = len(g2l)

class SingleDataLoader(BasicDataLoader):
    """
    Single Dataloader creates training/eval batches during KGQA.
    """
    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        super().__init__(config, word2id, relation2id, entity2id, tokenize, data_type)
        
    def get_batch(self, iteration, batch_size, fact_dropout, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.query_texts[sample_ids]  # 获取 question 编码结果
        kb_adj_mats = self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)
        
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mats, \
                   q_input, \
                   seed_dist, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mats, \
               q_input, \
               seed_dist, \
               self.answer_dists[sample_ids]


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def load_data(config, tokenize):

    """
    Creates train/val/test dataloaders (seperately).
    """
    # 分别读取 entities.txt/relation.txt/vocab_new.txt 得到 string2id——其中的 id 顺序在后续的子图文件中是一致的
    entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    
    valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
    test_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
    if config["is_eval"]:
        train_data = None
        num_word = test_data.num_word
    else:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="train")
        num_word = train_data.num_word
    relation_texts = test_data.rel_texts
    relation_texts_inv = test_data.rel_texts_inv
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data, #test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id,
        "num_word": num_word,
        "rel_texts": relation_texts,
        "rel_texts_inv": relation_texts_inv,
    }
    return dataset


if __name__ == "__main__":
    st = time.time()
    #args = get_config()
    load_data(args)
