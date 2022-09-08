import sys
sys.path.append('..')
import util
import torch

import math
import numpy as np
import gpushift
import dgl

from torch import nn
from torch.nn import functional as F



class SaCaProto(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.sim = torch.nn.CosineSimilarity(dim=2)
        
        # embedding dimension
        self.embedding_dim = 512
        
        # for self attention
        self.ln_ks = nn.LayerNorm(768)
        self.ln_vs = nn.LayerNorm(768)
        self.ln_qs = nn.LayerNorm(768)
        
        self.to_ks = nn.Linear(768, 768 , bias=False)
        self.to_vs = nn.Linear(768, 768 , bias=False)
        self.to_qs = nn.Linear(768, 768, bias=False)
        
        self.mhas = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        
        # for cross attention
        self.ln_k = nn.LayerNorm(768)
        self.ln_v = nn.LayerNorm(768)
        self.ln_q = nn.LayerNorm(768)
        
        self.to_k = nn.Linear(768, 768, bias=False)
        self.to_v = nn.Linear(768, 768, bias=False)
        self.to_q = nn.Linear(768, 768, bias=False)
        
        self.mha = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        
        # projection
        self.proj = nn.Sequential(
            nn.Linear(768, self.embedding_dim),
            nn.Dropout(0.2)
        )
        
        # for label embedding
        self.label_embedding = nn.Embedding(6, self.embedding_dim, scale_grad_by_freq=True)
        
        # for embedding scaling
        self.scaler = nn.Sequential(
            nn.BatchNorm1d(self.embedding_dim, track_running_stats=False),
            nn.ReLU()
        )
        
        # k-NN graph
        self.K = 10
        
        # graph layer
        self.gconv1 = dgl.nn.pytorch.conv.GraphConv(self.embedding_dim, self.embedding_dim)
        self.gconv2 = dgl.nn.pytorch.conv.GraphConv(self.embedding_dim, 6)
        
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
        return torch.cat(embs), torch.cat(labels)#F.one_hot(torch.cat(labels).long())
    
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
            
            """
            # Version 1) self-attention of S & cross attention of Q -> ProtoNet
            # Get query, key, and value for self-attention
            Q, K, V = self.to_qs(self.ln_qs(s_emb_selected)), self.to_ks(self.ln_ks(s_emb_selected)), self.to_vs(self.ln_vs(s_emb_selected))
            s_emb_selected = self.mhas(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze()
            
            # Get query, key (from S) and value (from Q) for cross-attention
            Q, K, V = self.to_q(self.ln_q(q_emb_selected)), self.to_k(self.ln_k(s_emb_selected)), self.to_v(self.ln_v(s_emb_selected))
            q_emb_selected = self.mha(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze()
            
            s_emb_selected, q_emb_selected = self.proj(s_emb_selected), self.proj(q_emb_selected)
            
            # Transform into task-specific embeddings            
            s_label_embs = self.label_embedding(s_label_selected.long())
            q_label_embs = self.label_embedding.weight.mean(0).unsqueeze(0).repeat(len(q_emb_selected), 1)
            s_emb_selected = s_emb_selected + s_label_embs
            q_emb_selected = q_emb_selected + q_label_embs

            # Scale embedding
            s_emb_selected = self.scaler(s_emb_selected)
            q_emb_selected = self.scaler(q_emb_selected)
            
            # get prototype
            support_proto = self.__get_proto__(
                s_emb_selected, 
                s_label_selected
            )
             
            # calculate distance to each prototype
            logit = self.__batch_dist__(
                support_proto,
                q_emb_selected
            )
            """
            
            
            
            # Version 2) self-attention of S & cross attention -> Transductive GNN
            # Get query, key, and value for self-attention
            Q, K, V = self.to_qs(self.ln_qs(s_emb_selected)), self.to_ks(self.ln_ks(s_emb_selected)), self.to_vs(self.ln_vs(s_emb_selected))
            s_emb_selected = self.mhas(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze()
            
            # Get query, key (from S) and value (from Q) for cross-attention
            Q, K, V = self.to_q(self.ln_q(q_emb_selected)), self.to_k(self.ln_k(s_emb_selected)), self.to_v(self.ln_v(s_emb_selected))
            q_emb_selected = self.mha(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze()
            s_emb_selected, q_emb_selected = self.proj(s_emb_selected), self.proj(q_emb_selected)
            
            # Transform into task-specific embeddings            
            s_label_embs = F.dropout(self.label_embedding(s_label_selected.long()), p=0.2)
            q_label_embs = F.dropout(self.label_embedding.weight.mean(0).unsqueeze(0).repeat(len(q_emb_selected), 1), p=0.2)
            s_emb_selected = s_emb_selected + s_label_embs
            q_emb_selected = q_emb_selected + q_label_embs

            # Scale embedding
            s_emb_selected = self.scaler(s_emb_selected)
            q_emb_selected = self.scaler(q_emb_selected)
                        
            # Graph construction
            embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            
            """
            ## Euclidean adjcency matrix
            bandwidth = 1
            W = emb1.sub(emb2).pow(2).mean(2).mul(-1).div(2 * bandwidth**2).exp()   # N*N*d -> N*N
            """
            
            ## Dot product adjacency matrix
            W = (emb1 * emb2).sum(2)

            ## k-NN squashing
            topk, indices = torch.topk(W, self.K)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).float()      # union, kNN graph
            W = W * mask.to(W.device)

            ## normalize degree matrix
            D = W.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(embs))
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(embs), 1)
            adj = D1 * W * D2
            
            # construct graph
            graph = dgl.graph(torch.nonzero(adj, as_tuple=True))
            graph = dgl.add_self_loop(graph)
            
            graph.ndata['feat'] = embs
            graph = dgl.to_homogeneous(graph, ndata=['feat'])
       
            embs = F.relu(self.gconv1(graph, embs))
            logit = self.gconv2(graph, embs)
            
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)        
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, None, proto, embs

    
    

    
