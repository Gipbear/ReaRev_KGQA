import torch
from torch.autograd import Variable
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform


class ReaRev(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super().__init__(args, num_entity, num_relation, num_word)
        self.layers(args)
        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.lm = args['lm']
        
        self.private_module_def(args, num_entity, num_relation)
        self.to(self.device)
        self.fusion = Fusion(self.entity_dim)
        self.reform = nn.ModuleList([QueryReform(self.entity_dim) for _ in range(self.num_ins)])

    def layers(self, args):
        self.linear_drop = nn.Dropout(p=args['linear_dropout'])
        self.type_layer = TypeLayer(in_features=self.entity_dim, out_features=self.entity_dim,
                                    linear_drop=self.linear_drop, device=self.device)
        self.self_att_r = AttnEncoder(self.entity_dim)

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        local_entity_emb = self.type_layer(local_entity=local_entity,
                                            edge_list=kb_adj_mat,
                                            rel_features=rel_features)
        return local_entity_emb
   
    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        rel_features = self.instruction.question_emb(self.rel_features)  # Tsize, max_rel_words, entity_dim 对关系短语进行编码
        rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
        rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())  # Tsize, max_rel_words 计算得到关系词的注意力
        rel_features_inv = self.self_att_r(rel_features_inv,  (self.rel_texts != self.instruction.pad_val).float())
        if self.lm == 'lstm':
            rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
            rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())
        return rel_features, rel_features_inv

    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, args['lm'])

    def encode_rel_texts(self, rel_texts, rel_texts_inv):
        self.rel_texts = torch.from_numpy(rel_texts).type('torch.LongTensor').to(self.device)  # Tsize, max_rel_words
        self.rel_texts_inv = torch.from_numpy(rel_texts_inv).type('torch.LongTensor').to(self.device)
        self.instruction.eval()
        with torch.no_grad():
            self.rel_features = self.instruction.encode_question(self.rel_texts, store=False)
            self.rel_features_inv = self.instruction.encode_question(self.rel_texts_inv, store=False)
        self.rel_features.requires_grad = False
        self.rel_features_inv.requires_grad = False

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Initializing Reasoning
        """
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv  = self.get_rel_feature()  # 跟问句类似，将关系短语转换为 单词序列，使用 tokenizer 进行编码得到关系隐藏层特征
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)  # Bsize, max_local_entity, entity_dim 使用三元组初始化每个实体编码
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        self.reasoning.init_reason( 
                                   local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features,
                                   rel_features_inv=rel_features_inv,
                                   query_entities=query_entities)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss
    
    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """

        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)
        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        
        """
        Instruction generations
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)  # 初始化关系编码、实体编码、KG稀疏矩阵
        self.instruction.init_reason(q_input)  # ? 似乎与上一步 self.init_reason 中的 self.instruction(q_input) 步骤重复，为何要再进行相同的计算？非 forward 不传播参数？
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i) 
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        #relation_ins = torch.cat(self.instruction.instructions, dim=1)
        #query_emb = None
        self.dist_history.append(self.curr_dist)

        """
        BFS + GNN reasoning
        """

        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist
            for j in range(self.num_gnn):
                self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
            self.dist_history.append(self.curr_dist)
            qs = []

            """
            Instruction Updates
            """
            for j in range(self.num_ins):
                q = self.reform[j](self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)  # 参考 SGReader 提出的查询重构方法对指令进行重新编码
        """
        Answer Predictions
        """
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        # filter no answer training case
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        
        pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list

    