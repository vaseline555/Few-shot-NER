import sys
sys.path.append('..')
import util
import torch

import math
import numpy as np
import dgl

from torch import nn
from torch.nn import functional as F

import ot
from ot.gromov import gromov_wasserstein2

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
    
class SaCaProto(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout(0.2)
        self.dot = dot
        
        # embedding dimension
        self.embedding_dim = 768
        
        # for self attention
        self.ln_s = nn.LayerNorm(self.embedding_dim)
        
        self.to_ks = nn.Linear(self.embedding_dim, self.embedding_dim , bias=False)
        self.to_vs = nn.Linear(self.embedding_dim, self.embedding_dim , bias=False)
        self.to_qs = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.mhas = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=16, batch_first=True)
        
        # for cross attention
        self.ln_q = nn.LayerNorm(self.embedding_dim)
        
        self.to_kq = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.to_vq = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.to_qq = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.mhaq = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=16, batch_first=True)
        
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
        return torch.cat(embs), torch.cat(labels)
    
    def __get_proto__(self, emb, label):
        proto = []
        for l in torch.unique(label):
            proto.append(torch.mean(emb[label == l], 0))
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
        support_emb = self.word_encoder(support['word'], support['mask'])#.detach() # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask'])#.detach() # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)
        
        # Prototypical Networks
        logits = []
        proto = []
        embs = []
        ot_loss = 0
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
            s_emb_selected, s_label_selected = self.__get_embeddings__(
                s_emb,
                s_label,
                s_mask
            )
            
            # Get query embeddings
            q_emb = query_emb[current_query_num:current_query_num+sent_query_num]
            q_mask = query['text_mask'][current_query_num:current_query_num+sent_query_num]
            q_emb_selected = self.__get_embeddings__(
                q_emb,
                None,
                q_mask
            )
           
            s_emb_selected = torch.nn.functional.normalize(s_emb_selected, p=2, dim=1)
            q_emb_selected = torch.nn.functional.normalize(q_emb_selected, p=2, dim=1)
            
            
            
                                 
            # 1) Self-Attention & Cross-Attention
            # Get query, key, and value for self-attention
            Q, K, V = self.to_qs(s_emb_selected), self.to_ks(s_emb_selected), self.to_vs(s_emb_selected)
            s_emb_selected = self.ln_s(s_emb_selected + self.mhas(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze())
            
            # Get query, key (from S) and value (from Q) for cross-attention
            Q, K, V = self.to_qq(q_emb_selected), self.to_kq(s_emb_selected), self.to_vq(s_emb_selected)
            q_emb_selected = self.ln_q(q_emb_selected + self.mhaq(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze())
            
           
            
            
            # 2) Get adjacency
            embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj_dense = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            
            
            # 2) Optimize Gromov-Wasserstein loss
            adj_prior = torch.eye(len(torch.unique(s_label_selected))).to(embs.device)

            
            a0 = torch.rand(len(adj_dense)).to(embs.device).requires_grad_(True)
            a0 = a0 / a0.sum()
            a1 = torch.rand(len(adj_dense)).to(embs.device)

            ot_loss = gromov_wasserstein2(adj_prior, adj_dense, a0, a1)
            ot_loss.backward()
            
            with torch.no_grad():
                grad = a0.grad
                a0 -= grad * 1e-2   # step
                a0.grad.zero_()
                a0.data = ot.utils.proj_simplex(a0)
            logit = ot.gromov_wasserstein(adj_dense, adj_prior, ot.unif(len(embs)), a0)
            
            
            
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)        
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, None, proto, embs

    
    

    
