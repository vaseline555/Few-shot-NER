import sys
sys.path.append('..')
import util
import torch

import math
import numpy as np
import dgl

from torch import nn
from torch.nn import functional as F
from sklearn.cluster import MeanShift

from collections import Counter


    
class TwinGNN(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout(0.2)
        self.dot = dot
        
        # embedding dimension
        self.embedding_dim = 768
        
        # graph configurations
        self.K = 5
        self.p = 0.2
        
        # graph layer
        self.task_gconv = dgl.nn.pytorch.conv.GraphConv(self.embedding_dim, self.embedding_dim) 
        
        self.span_gconv = dgl.nn.pytorch.conv.GraphConv(self.embedding_dim, self.embedding_dim)
        
        # projection layer
        self.proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(self.embedding_dim, 6)
        )
    
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        dist = self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)
        return dist
    
    def __get_proto__(self, emb, label):
        proto = []
        for l in torch.unique(label):
            proto.append(torch.mean(emb[label == l], 0))
        proto = torch.stack(proto)
        return proto

    def FKT(self, X, y, n_components):
        X1, X2 = X[y == 0], X[y != 0]
        m, m1, m2 = X.mean(0), X1.mean(0), X2.mean(0)
        Hb = torch.stack([math.sqrt(len(X1)) * (m1 - m), math.sqrt(len(X2)) * (m2 - m)], 1)
        Ht = torch.cat([X1, X2], 0).T - m.reshape(-1, 1)
        Q, R = torch.linalg.qr(Ht)
        S_tilde_t = R @ R.T
        Z = Q.T @ Hb
        S_tilde_b = Z @ Z.T
        lam, V = torch.linalg.eigh(torch.linalg.inv(S_tilde_t) @ S_tilde_b + torch.eye(len(S_tilde_t)).to(S_tilde_t.device).mul(1e-4))
        V = V[torch.argsort(lam, descending=True)]
        proj_F = Q @ V
        return proj_F[:, :n_components]
    
    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask']).detach() # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']).detach() # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)
        
        # Prototypical Networks
        logits = []
        proto = []
        embs = []
        ent_loss = 0
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            
            # Get support embeddings
            s_emb = support_emb[current_support_num:current_support_num+sent_support_num]
            s_label = support['label'][current_support_num:current_support_num+sent_support_num]
            s_mask = support['text_mask'][current_support_num:current_support_num+sent_support_num]
            
            # Get query embeddings
            q_emb = query_emb[current_query_num:current_query_num+sent_query_num]
            q_mask = query['text_mask'][current_query_num:current_query_num+sent_query_num]

            # Pre-process embeddings, masks, labels
            ## support set
            s_label = torch.cat(s_label)
            s_emb = s_emb[s_mask == 1][s_label != -1]
            s_label = s_label[s_label != -1]

            ## query set
            q_emb = q_emb[q_mask == 1]

            ## normalize
            s_emb = torch.nn.functional.normalize(s_emb, p=2, dim=1)
            q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1)

            # Concatenate embeddings for transductive setting
            embs = torch.cat([s_emb, q_emb], 0)

            # Task embedding
            sq_adj = embs.unsqueeze(1).sub(embs.unsqueeze(0)).pow(2).mean(2).mul(-1).div(2).exp()   # N*N*d -> N*N
            topk, indices = torch.topk(sq_adj, self.K)
            mask = torch.zeros_like(sq_adj)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).float()      # union, kNN graph
            sq_adj = sq_adj * mask.to(sq_adj.device)
            sq_graph = dgl.graph(torch.nonzero(sq_adj, as_tuple=True))
            span_embs = self.span_gconv(sq_graph, embs, edge_weight=sq_adj[torch.nonzero(sq_adj, as_tuple=True)])        
            s_emb = span_embs[:len(s_emb)]
            q_emb = span_embs[len(s_emb):]
            # 잘되면 FKT 추가 후 성능비교
            
            
            # Class embedding
            support_proto = self.__get_proto__(s_emb, s_label)
            s_src = torch.arange(len(torch.unique(s_label)) + len(s_label)).to(s_label.device)
            s_dst = torch.cat([torch.arange(len(torch.unique(s_label))).to(s_label.device), s_label], 0)
            
            q_src = torch.arange(len(q_emb)).to(s_src.device).add(len(s_src)).repeat(len(torch.unique(s_label)))
            q_dst = torch.unique(s_label).to(s_label.device).view(-1, 1).repeat(1, len(q_emb)).view(-1)

            ## convert into DGL data structure
            task_graph = dgl.graph((torch.cat([s_src, q_src]).tolist(), torch.cat([s_dst, q_dst]).tolist())).to(s_emb.device)
            task_graph = dgl.add_self_loop(task_graph)
            task_embs = self.task_gconv(task_graph, torch.cat([support_proto, embs], 0))
            task_proto = task_embs[:len(support_proto)]
            q_emb = task_embs[len(support_proto) + len(s_emb):]
                                                 
            # 4) Merge embeddings
            #logit = self.proj(F.relu(torch.cat([span_embs, task_embs], 1)))[len(s_emb):]
            logit = self.__batch_dist__(task_proto, q_emb)
              
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)        
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, None, proto, embs

    
    

    
