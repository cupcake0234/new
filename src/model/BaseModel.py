from src.utils import *
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch import Tensor
from copy import deepcopy

class BaseModel(nn.Module):
    def __init__(self, args, kg):
        super(BaseModel, self).__init__()
        self.args = args
        self.kg = kg  # information of snapshot sequence, self.kg.snapshots[i] is the i-th snapshot

        '''initialize the entity and relation embeddings for the first snapshot'''
        # embedding也是有参数的
        if self.args.Plan_yuan == "True":
            logging.info('当前模型:Plan_yuan')
            self.ent_embeddings = nn.Embedding(self.kg.snapshots[0].num_ent, self.args.emb_dim).to(self.args.device).double()
            self.rel_embeddings = nn.Embedding(self.kg.snapshots[0].num_rel, self.args.emb_dim).to(self.args.device).double()
            xavier_normal_(self.ent_embeddings.weight)
            xavier_normal_(self.rel_embeddings.weight)
        elif self.args.Plan_weight == 'True':
            logging.info('当前模型:Plan_weight')
            self.ent_embeddings = nn.Parameter(Tensor(self.kg.snapshots[0].num_ent, self.args.emb_dim).to(self.args.device).double())
            self.ent_embeddings_list = nn.ParameterList()
            self.ent_embeddings_list.append(self.ent_embeddings)

            self.expand_ent_embeddings = None

            self.rel_embeddings = nn.Parameter(Tensor(self.kg.snapshots[0].num_rel, self.args.emb_dim).to(self.args.device).double())
            # weight for every entity
            self.linear_w = nn.Parameter(Tensor(self.kg.snapshots[0].num_ent,1).to(self.args.device).double())
            self.linear_w_list = nn.ParameterList()
            self.linear_w_list.append(self.linear_w)
            self.expand_linear_w = None

        '''loss function'''
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction="sum")#.to(self.args.device)  #

        self.init_parameters()

    def init_parameters(self):
        logging.info('Model Parameter Configuration:')
        for name, param in self.named_parameters():
            # named_parameters返回参数及参数名字，requires_grad是否带梯度
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
            if 'bias' in name:
                continue
            nn.init.xavier_uniform_(param)

    def reinit_param(self):
        '''
        Re-initialize all model parameters
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                xavier_normal_(p)

    def init_expanded_parameters(self):

        # 新方案：尝试继承前面已经训练好的向量嵌入
        # if self.expand_ent_embeddings is not None:
        #     ent_embeddings,_ = self.embedding()
        #     ent_embeddings = deepcopy(ent_embeddings.data)
        #     self.expand_ent_embeddings = nn.Parameter(ent_embeddings)

        # if self.expand_linear_w is not None:
        #     nn.init.xavier_uniform_(self.expand_linear_w)
        if self.args.Plan_weight == "True":
            if self.expand_ent_embeddings is not None:
                nn.init.xavier_uniform_(self.expand_ent_embeddings)

            if self.expand_linear_w is not None:
            # 配合expand函数中weight方案的
            # if self.args.Plan_weight == 'True':
            #     return
                nn.init.xavier_uniform_(self.expand_linear_w)

    # 实体参数横向扩展使用的
    def expand(self):
    
        if self.args.Plan_weight == 'True':
            self.expand_ent_embeddings = nn.Parameter(Tensor(self.kg.snapshots[self.args.snapshot].num_ent, self.args.emb_dim).to(self.args.device).double(), requires_grad=True)
            logging.info('Plan_weight 产生新权重')
            # 尝试将扩展的权重初始化为 0 的方案
            # self.expand_linear_w = nn.Parameter(torch.zeros(self.kg.snapshots[self.args.snapshot].num_ent, 1).to(self.args.device).double(), requires_grad=True)
            self.expand_linear_w = nn.Parameter(Tensor(self.kg.snapshots[self.args.snapshot].num_ent, 1).to(self.args.device).double(), requires_grad=True)

    # Plan_yuan使用的
    def expand_embedding_size(self):
        '''
        Initialize entity and relation embeddings for next snapshot
        '''
        ent_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_ent, self.args.emb_dim).to(
            self.args.device).double()
        rel_embeddings = nn.Embedding(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim).to(
            self.args.device).double()
        xavier_normal_(ent_embeddings.weight)
        xavier_normal_(rel_embeddings.weight)
        return deepcopy(ent_embeddings), deepcopy(rel_embeddings)
    
    # 实体纵向扩展
    def expand_ent_embedding_size(self):
        '''
        Initialize entity embeddings for next snapshot
        '''
        ent_embeddings = nn.Parameter(Tensor(self.kg.snapshots[self.args.snapshot + 1].num_ent, self.args.emb_dim)).to(self.args.device).double()
        # 这个weight就是实体及关系的向量嵌入
        xavier_normal_(ent_embeddings)
        return ent_embeddings.clone()
    
    # 关系正常扩展
    def expand_rel_embedding_size(self):
        '''
        Initialize relation embeddings for next snapshot
        '''
        rel_embeddings = nn.Parameter(Tensor(self.kg.snapshots[self.args.snapshot + 1].num_rel, self.args.emb_dim)).to(self.args.device).double()
        # 这个weight就是实体及关系的向量嵌入
        xavier_normal_(rel_embeddings)
        return rel_embeddings.clone()
    
    def open_parameters(self):
        if self.args.Plan_weight == 'True':
            for ent_embeddings in self.ent_embeddings_list:
                ent_embeddings.requires_grad = True
            
            for linear_w in self.linear_w_list:
                linear_w.requires_grad = True

            # self.rel_embeddings.requires_grad = True

    def isolate_parameters(self):
        if self.args.Plan_weight == 'True':
            for item in self.ent_embeddings_list:
                item.requires_grad = False
            for linear_w in self.linear_w_list:
                linear_w.requires_grad = False

            # 将关系也锁住
            # isolate_rel_embeddings = self.rel_embeddings[:self.kg.snapshots[self.args.snapshot - 1].num_rel]
            # isolate_rel_embeddings.requires_grad = False

    def combine(self):
        if self.args.Plan_weight == 'True':
            if self.expand_ent_embeddings is not None:
                self.ent_embeddings_list.append(self.expand_ent_embeddings)
                self.expand_ent_embeddings = None

            if self.expand_linear_w is not None:
                self.linear_w_list.append(self.expand_linear_w)
                self.expand_linear_w = None

    def switch_snapshot(self):
        '''
        After the training process of a snapshot, prepare for next snapshot
        '''
        pass

    def pre_snapshot(self):
        '''
        Preprocess before training on a snapshot
        '''
        pass

    def epoch_post_processing(self, size=None):
        '''
        Post process after a training iteration
        '''
        pass

    def snapshot_post_processing(self):
        '''
        Post process after training on a snapshot
        '''
        pass

    def store_old_parameters(self):
        '''
        Store the learned model after training on a snapshot
        '''
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer('old_data_{}'.format(name), value.clone().detach())

    def initialize_old_data(self):
        '''
        Initialize the storage of old parameters
        '''
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '_')
                self.register_buffer('old_data_{}'.format(n), p.data.clone())

    def embedding(self, stage=None):
        '''
        :param stage: Train / Valid / Test
        :return: entity and relation embeddings
        '''
        if self.args.Plan_yuan == 'True':
            return self.ent_embeddings.weight, self.rel_embeddings.weight
        elif self.args.Plan_weight == 'True':
            if self.args.snapshot == 0:
                ent_embeddings_out = self.ent_embeddings_list[0] * self.linear_w_list[0]
                return ent_embeddings_out, self.rel_embeddings
            
            ent_embeddings_out = None
            if self.expand_ent_embeddings is not None:
                num_ent = self.expand_ent_embeddings.shape[0]
            else:
                num_ent = self.ent_embeddings_list[-1].shape[0]

            for i in range(len(self.ent_embeddings_list)):
                if ent_embeddings_out is None:
                    ent_embeddings_out = self.ent_embeddings_list[i] * self.linear_w_list[i]
                    if ent_embeddings_out.shape[0] < num_ent:
                        ent_embeddings_out = torch.cat([ent_embeddings_out,torch.zeros(num_ent - ent_embeddings_out.shape[0], self.args.emb_dim).to(self.args.device)],dim=0)
                else:
                    ent_embeddings_temp = self.ent_embeddings_list[i] * self.linear_w_list[i]
                    if ent_embeddings_temp.shape[0] < num_ent:
                        ent_embeddings_temp = torch.cat([ent_embeddings_temp,torch.zeros(num_ent - ent_embeddings_temp.shape[0], self.args.emb_dim).to(self.args.device)],dim=0)
                    ent_embeddings_out += ent_embeddings_temp

            if self.expand_ent_embeddings is not None:
                ent_embeddings_temp = self.expand_ent_embeddings * self.expand_linear_w
                ent_embeddings_out += ent_embeddings_temp

            return ent_embeddings_out,self.rel_embeddings

            if self.args.rectify:
                for i in range(len(self.ent_embeddings_list)):
                    if ent_embeddings_out is None:
                        ent_embeddings_out = self.ent_embeddings_list[i] * self.linear_w_list[i]
                        if ent_embeddings_out.shape[0] < self.kg.snapshots[self.args.snapshot - 1].num_ent:
                            ent_embeddings_out = torch.cat([ent_embeddings_out,torch.zeros(self.kg.snapshots[self.args.snapshot - 1].num_ent - ent_embeddings_out.shape[0], self.args.emb_dim).to(self.args.device)],dim=0)
                    else:
                        ent_embeddings_temp = self.ent_embeddings_list[i] * self.linear_w_list[i]
                        if ent_embeddings_temp.shape[0] < self.kg.snapshots[self.args.snapshot - 1].num_ent:
                            ent_embeddings_temp = torch.cat([ent_embeddings_temp,torch.zeros(self.kg.snapshots[self.args.snapshot - 1].num_ent - ent_embeddings_temp.shape[0], self.args.emb_dim).to(self.args.device)],dim=0)
                        ent_embeddings_out += ent_embeddings_temp
            # for retrain 
            elif self.args.retrain:
                ent_embeddings_out = self.expand_ent_embeddings * self.expand_linear_w
                for i in range(len(self.ent_embeddings_list)):
                    ent_embeddings_temp = self.ent_embeddings_list[i] * self.linear_w_list[i]
                    if ent_embeddings_temp.shape[0] < ent_embeddings_out.shape[0]:
                        ent_embeddings_temp = torch.cat([ent_embeddings_temp,torch.zeros(ent_embeddings_out.shape[0] - ent_embeddings_temp.shape[0], self.args.emb_dim).to(self.args.device)],dim=0)
                    ent_embeddings_out += ent_embeddings_temp
            else: #for the test after retrain
                for i in range(len(self.ent_embeddings_list)):
                    if ent_embeddings_out is None:
                        ent_embeddings_out = self.ent_embeddings_list[i] * self.linear_w_list[i]
                        if ent_embeddings_out.shape[0] < self.kg.snapshots[self.args.snapshot].num_ent:
                            ent_embeddings_out = torch.cat([ent_embeddings_out,torch.zeros(self.kg.snapshots[self.args.snapshot].num_ent - ent_embeddings_out.shape[0], self.args.emb_dim).to(self.args.device)],dim=0)
                    else:
                        ent_embeddings_temp = self.ent_embeddings_list[i] * self.linear_w_list[i]
                        if ent_embeddings_temp.shape[0] < self.kg.snapshots[self.args.snapshot].num_ent:
                            ent_embeddings_temp = torch.cat([ent_embeddings_temp,torch.zeros(self.kg.snapshots[self.args.snapshot].num_ent - ent_embeddings_temp.shape[0], self.args.emb_dim).to(self.args.device)],dim=0)
                        ent_embeddings_out += ent_embeddings_temp

            return ent_embeddings_out, self.rel_embeddings
        
    def new_loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: loss of new facts
        '''
        return self.margin_loss(head, rel, tail, label)/head.size(0)



    def margin_loss(self, head, rel, tail, label=None):
        '''
        Pair Wise Margin loss: L1-norm(s + r - o)
        :param head:
        :param rel:
        :param tail:
        :param label:
        :return:
        '''
        ent_embeddings, rel_embeddings = self.embedding('Train')

        s = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, rel)
        o = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(s, r, o)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        loss = self.margin_loss_func(p_score, n_score, y)
        return loss

    def split_pn_score(self, score, label):
        '''
        Get the scores of positive and negative facts
        :param score: scores of all facts
        :param label: positive facts: 1, negative facts: -1
        :return:
        '''
        p_score = score[torch.where(label>0)]
        n_score = (score[torch.where(label<0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def score_fun(self, s, r, o):
        '''
        score function f(s, r, o) = L1-norm(s + r - o)
        :param h:
        :param r:
        :param t:
        :return:
        '''
        s = self.norm_ent(s)
        r = self.norm_rel(r)
        o = self.norm_ent(o)
        return torch.norm(s + r - o, 1, -1)

    def predict(self, sub, rel, stage='Valid'):
        '''
        Scores all candidate facts for evaluation
        :param head: subject entity id
        :param rel: relation id
        :param stage: object entity id
        :return: scores of all candidate facts
        '''

        '''get entity and relation embeddings'''
        if stage != 'Test':
            num_ent = self.kg.snapshots[self.args.snapshot].num_ent
        else:
            num_ent = self.kg.snapshots[self.args.snapshot_test].num_ent
        ent_embeddings, rel_embeddings = self.embedding(stage)
        s = torch.index_select(ent_embeddings, 0, sub)
        r = torch.index_select(rel_embeddings, 0, rel)
        # 这个地方还挺巧妙的
        o_all = ent_embeddings[:num_ent]
        s = self.norm_ent(s)
        r = self.norm_rel(r)
        o_all = self.norm_ent(o_all)

        '''s + r - o'''
        pred_o = s + r
        score = 9.0 - torch.norm(pred_o.unsqueeze(1) - o_all, p=1, dim=2)
        score = torch.sigmoid(score)

        return score

    def norm_rel(self, r):
        return nn.functional.normalize(r, 2, -1)

    def norm_ent(self, e):
        return nn.functional.normalize(e, 2, -1)
