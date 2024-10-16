
import torch
import torch.nn as nn
import torch.nn.functional as F
VERY_NEG_NUMBER = -100000000000
VERY_SMALL_NUMBER = 1e-10


class TypeLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(TypeLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        self.kb_self_linear = nn.Linear(in_features, out_features)
        self.device = device

    def forward(self, local_entity, edge_list, rel_features):
        '''
        input_vector: (batch_size, max_local_entity)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        '''
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features
        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        val_one = torch.ones_like(batch_ids).float().to(self.device)
        
        # print("Prepare data:{:.4f}".format(time.time() - st))
        # Step 1: Calculate value for every fact with rel and head
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        fact_val = self.kb_self_linear(fact_rel)  # Tsize, entity_dim  关系编码作为注意力
        # fact_val = self.kb_self_linear(fact_rel)#self.kb_head_linear(self.linear_drop(fact_ent))

        # Step 3: Edge Aggregation with Sparse MM
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))

        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        f2e_emb = F.relu(torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val))  # Bsize, max_local_entity, Tsize
        assert not torch.isnan(f2e_emb).any()

        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)  # 得到三元组的编码表示

        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)
