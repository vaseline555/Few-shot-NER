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



class OAuProto(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.sim = torch.nn.CosineSimilarity(dim=2)
        
        # embedding dimension
        self.embedding_dim = 256
        
        # for self attention
        self.ln_s = nn.LayerNorm(768)
        
        self.to_ks = nn.Linear(768, 768 , bias=False)
        self.to_vs = nn.Linear(768, 768 , bias=False)
        self.to_qs = nn.Linear(768, 768, bias=False)
        
        self.mhas = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        
        # for cross attention
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
            Q, K, V = self.to_qs(s_emb_selected), self.to_ks(s_emb_selected), self.to_vs(s_emb_selected)
            s_emb_selected = self.ln_s(s_emb_selected + self.mhas(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze())
            
            # Get query, key (from S) and value (from Q) for cross-attention
            Q, K, V = self.to_q(q_emb_selected), self.to_k(s_emb_selected), self.to_v(s_emb_selected)
            q_emb_selected = self.ln_q(q_emb_selected + self.mha(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze())
            s_emb_selected, q_emb_selected = self.proj(F.relu(s_emb_selected)), self.proj(F.relu(q_emb_selected))
            
            # Transform into task-specific embeddings            
            s_label_embs = F.dropout(self.label_embedding(s_label_selected.long()), p=0.2)
            q_label_embs = F.dropout(self.label_embedding.weight.mean(0).unsqueeze(0).repeat(len(q_emb_selected), 1), p=0.2)
            s_emb_selected = s_emb_selected + s_label_embs
            q_emb_selected = q_emb_selected + q_label_embs

            # Scale embedding
            s_emb_selected = self.scaler(s_emb_selected)
            q_emb_selected = self.scaler(q_emb_selected)
            """
            
            # O-calss prototype augmentation
            mean_shift = gpushift.MeanShift(
                n_iter=10,  
                kernel='gaussian',
                bandwidth=0.5,
                use_keops=False
            )
            O_in_S = s_emb_selected[s_label_selected == 0]
            O_aug_proto = mean_shift(O_in_S.unsqueeze(0)).squeeze()
            
            # Get prototypes of entities
            support_proto = self.__get_proto__(
                s_emb_selected[s_label_selected != 0], 
                s_label_selected[s_label_selected != 0]
            )
            
            # calculate distance to each prototype
            logit_augmented = self.__batch_dist__(
                torch.cat([O_aug_proto, support_proto], 0),
                q_emb_selected
            )
            logit = logit_augmented[:, :len(O_aug_proto)].max(1)[0]
            logit = torch.cat([logit.view(-1, 1), logit_augmented[:, len(O_aug_proto):]], dim=1)
            
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)        
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, None, proto, embs

    
    

    
