import sys
sys.path.append('..')
import util
import torch

import math
import numpy as np
import dgl
from torch import nn
from torch.nn import functional as F



class Proto(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1, N=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.gconv = dgl.nn.pytorch.conv.GraphConv(768, 768)
        self.threshold = nn.Parameter(torch.tensor([0.95]).cuda(), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor([0.99]).cuda(), requires_grad=True)
        
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)
    
    def __get_embeddings__(self, embedding, tag, mask):
        embs, labels = [], []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        if tag is not None:
            tag = torch.cat(tag, 0)
            assert tag.size(0) == embedding.size(0)
            for label in range(torch.max(tag) + 1):
                curr_emb = embedding[tag == label]
                embs.append(curr_emb)
                labels.append(torch.ones(len(curr_emb)).mul(label).to(curr_emb.device))
        else:
            return embedding
        return torch.cat(embs), torch.cat(labels)#F.one_hot(torch.cat(labels).long())
    
    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag == label], 0))
        proto = torch.stack(proto)
        return proto

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        # Prototypical Networks
        logits = []
        proto = []
        embs = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            
            # Get support embeddings
            s_emb = support_emb[current_support_num:current_support_num+sent_support_num]
            s_label = support['label'][current_support_num: current_support_num+sent_support_num]
            s_mask = support['text_mask'][current_support_num: current_support_num+sent_support_num]
            s_emb_selected, s_label_selected = self.__get_embeddings__(
                s_emb,
                s_label,
                s_mask
            )
            
            # Get query embeddings
            q_emb = query_emb[current_query_num:current_query_num+sent_query_num]
            q_mask = query['text_mask'][current_query_num: current_query_num+sent_query_num]
            q_emb_selected = self.__get_embeddings__(
                q_emb,
                None,
                q_mask
            )
            
            # label propagation
            ## GCN 태워서 task에 맞게 embedding 변환 (transductive)
            embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > self.threshold, adj, 0)
            
            graph = dgl.graph(torch.nonzero(adj, as_tuple=True))
            graph = dgl.add_self_loop(graph)
            graph.ndata['h'] = embs
            graph = dgl.to_homogeneous(graph, ndata=['h'])
            
            ## 변환된 embedding으로 새롭게 그래프 구성
            embs = self.gconv(graph, embs)
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > self.threshold, adj, 0)
            
            ## 인접행렬 normalize
            D = adj.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + np.finfo(float).eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(embs))
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(embs), 1)
            S = D1 * adj * D2
            
            ## Label matrix 구성 (query set에 해당하는 label은 0으로 초기화)
            labels = torch.cat([s_label_selected, torch.zeros(len(q_emb_selected)).to(embs.device)]).long()
            labels = F.one_hot(labels).float()
            
            ## Label propagation 공식에 따라 logit 추정
            logit = torch.matmul(torch.inverse(torch.eye(len(labels)).to(embs.device) - self.alpha * S + np.finfo(float).eps), labels)
            logit = logit[len(s_label_selected):]
            
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return logits, pred

    
    
    
