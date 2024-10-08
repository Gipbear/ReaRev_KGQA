
import torch
import torch.nn.functional as F
import torch.nn as nn

from .base_gnn import BaseGNNLayer

VERY_NEG_NUMBER = -100000000000

class ReasonGNNLayer(BaseGNNLayer):
    """
    GNN Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super().__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.linear_drop = nn.Dropout(p=args['linear_dropout'])
        self.rel_linear = nn.ModuleList([
            nn.Linear(in_features=entity_dim, out_features=entity_dim)
            for _ in range(self.num_gnn)
        ])
        self.e2e_linear = nn.ModuleList([
            nn.Linear(in_features=2*(self.num_ins)*entity_dim + entity_dim, out_features=entity_dim)
        for _ in range(self.num_gnn)
        ])
        self.lin_m =  nn.Linear(in_features=(self.num_ins)*entity_dim, out_features=entity_dim)

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.build_matrix(kb_adj_mat)  # 构造一堆三元组的稀疏矩阵，部分使用转置可优化
        self.query_entities = query_entities

    def reason_layer(self, curr_dist, instruction, rel_linear):
        """
        Aggregates neighbor representations
        """
        fact_rel = torch.index_select(self.rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)  # 公式 5，计算 指令 k 下从当前节点传播的消息
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))  # 根据实体概率计算当前实体对应的三元组的重要性
        fact_val = fact_val * fact_prior  # 公式 6 根据三元组的概率，计算传播的消息
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)  # 公式 6 聚合消息得到指令 k 下 在 l 层节点的聚合消息
        assert not torch.isnan(f2e_emb).any()
        neighbor_rep = f2e_emb.view(self.batch_size, self.max_local_entity, self.entity_dim)
        return neighbor_rep

    def reason_layer_inv(self, curr_dist, instruction, rel_linear):
        fact_rel = torch.index_select(self.rel_features_inv, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
        fact_val = fact_val * fact_prior
        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()
        neighbor_rep = f2e_emb.view(self.batch_size, self.max_local_entity, self.entity_dim)
        return neighbor_rep

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        """
        Compute next probabilistic vectors and current node representations.
        """
        neighbor_reps = []
        for j in range(relational_ins.size(1)):  # i 的数量
            # we do the same procedure for existing and inverse relations
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], self.rel_linear[step])
            neighbor_reps.append(neighbor_rep)
            neighbor_rep = self.reason_layer_inv(current_dist, relational_ins[:,j,:], self.rel_linear[step])
            neighbor_reps.append(neighbor_rep)
        neighbor_reps = torch.cat(neighbor_reps, dim=2)  # 公式 7，第 l 层时，所有指令下的邻居聚合消息
        
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        self.local_entity_emb = F.relu(self.e2e_linear[step](self.linear_drop(next_local_entity_emb)))  # 公式 8 聚合得到当前实体表示

        score_tp = self.score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)  # 公式 9 计算下一跳的实体得分
        answer_mask = self.local_entity_mask
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER  # 将掩码部分置为无穷小
        current_dist = self.softmax_d1(score_tp)  # 公式 9，计算得到当前跳的实体概率分布
        if return_score:
            return score_tp, current_dist
        
        return current_dist, self.local_entity_emb 
