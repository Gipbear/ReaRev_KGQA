
from utils import create_logger
import time
import numpy as np
import os, math
from loguru import logger

import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim

from tqdm import tqdm

from dataset_load import load_data, SingleDataLoader
from models.ReaRev.rearev import ReaRev
from evaluate import Evaluator

class Trainer_KBQA(object):
    def __init__(self, args, model_name, logger_=None):
        self.args = args
        self.logger = logger_
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.best_h1b = 0.0
        self.best_f1b = 0.0
        self.batch_size = args['batch_size']
        self.eps = args['eps']
        self.learning_rate = args['lr']
        self.fact_dropout = args['fact_dropout']
        self.gradient_clip = args['gradient_clip']
        self.test_batch_size = args['test_batch_size']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.reset_time = 0
        self.load_data(args, args['lm'])
        logger.info("Entity: {}, Relation: {}, Word: {}", self.num_entity, self.num_relation, len(self.word2id))

        assert model_name == 'ReaRev'
        self.decay_rate = args['decay_rate'] if 'decay_rate' in args else 0.98
        self.model = ReaRev(self.args, self.num_entity, self.num_relation, self.num_word)  # todo: rename pad_val
        if args['relation_word_emb']:
            self.model.encode_rel_texts(self.rel_texts, self.rel_texts_inv)
        self.model.to(self.device)
        self.evaluator = Evaluator(args=args, model=self.model, entity2id=self.entity2id,
                                       relation2id=self.relation2id, device=self.device)
        self.load_pretrain()
        self.optim_def()

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

    def load_data(self, args, tokenize):
        dataset = load_data(args, tokenize)
        self.train_data: SingleDataLoader = dataset["train"]
        self.valid_data: SingleDataLoader = dataset["valid"]
        self.test_data: SingleDataLoader = dataset["test"]
        self.entity2id: dict = dataset["entity2id"]
        self.relation2id: dict = dataset["relation2id"]
        self.word2id: dict = dataset["word2id"]
        self.num_word = self.test_data.num_word
        self.num_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)
        self.rel_texts = self.test_data.rel_texts
        self.rel_texts_inv = self.test_data.rel_texts_inv

    def load_pretrain(self):
        args = self.args
        if args['load_experiment'] is not None:
            ckpt_path = os.path.join(args['checkpoint_dir'], args['load_experiment'])
            logger.info("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)

    def optim_def(self):
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim_model = optim.Adam(trainable, lr=self.learning_rate)
        if self.decay_rate > 0:
            self.scheduler = ExponentialLR(self.optim_model, self.decay_rate)

    def evaluate(self, data, test_batch_size=20, write_info=False):
        return self.evaluator.evaluate(data, test_batch_size, write_info)

    def train(self, epoch_num):
        eval_every = self.args['eval_every']
        logger.info("Start Training------------------")
        for epoch in range(epoch_num):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()
            if self.decay_rate > 0:
                self.scheduler.step()

            self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch + 1, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(np.mean(h1_list_all), np.mean(f1_list_all)))

            if (epoch + 1) % eval_every == 0:
                eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size)
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                if eval_h1 > self.best_h1:
                    self.best_h1 = eval_h1
                    self.save_ckpt("h1")
                    self.logger.info("BEST EVAL H1: {:.4f}".format(eval_h1))
                if eval_f1 > self.best_f1:
                    self.best_f1 = eval_f1
                    self.save_ckpt("f1")
                    self.logger.info("BEST EVAL F1: {:.4f}".format(eval_f1))
                eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size)
                self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        self.save_ckpt("final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        print("End Training------------------")
        self.evaluate_best()

    def evaluate_best(self):
        filename = os.path.join(self.args['checkpoint_dir'], "{}-h1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, write_info=False)
        self.logger.info("Best h1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-f1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size,  write_info=False)
        self.logger.info("Best f1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        eval_f1, eval_hits = self.evaluate(self.valid_data, self.test_batch_size, write_info=False)
        self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_hits))
        test_f1, test_hits = self.evaluate(self.test_data, self.test_batch_size, write_info=True)
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_hits))

    def train_epoch(self):
        self.model.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        num_batch = math.ceil(self.train_data.fact_num / self.batch_size)
        h1_list_all = []
        f1_list_all = []
        for iteration in tqdm(range(num_batch)):
            batch = self.train_data.get_batch(iteration, self.batch_size, self.fact_dropout)
            self.optim_model.zero_grad()
            loss, _, _, tp_list = self.model(batch, training=True)
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters()], self.gradient_clip)
            self.optim_model.step()
            losses.append(loss.item())
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, reason="h1"):
        model = self.model
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        model_name = os.path.join(self.args['checkpoint_dir'], f"{self.args['experiment_name']}-{reason}.ckpt")
        torch.save(checkpoint, model_name)
        logger.success("Best {}, save model as {}", reason, model_name)

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]
        # self.logger.info("Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())), filename))
        self.model.load_state_dict(model_state_dict, strict=False)

