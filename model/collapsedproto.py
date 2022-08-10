import sys
sys.path.append('..')
import util
import torch

import scipy
import math
import numpy as np
import ot

from torch import nn
from torch.nn import functional as F

class CollapsedPrOTo(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1, N=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.bary_map = None

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        #Q = self.bary_map.transform(Q)
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
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
            # Calculate prototype for each class
            support_proto = self.__get_proto__(
                support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num]
            )
                    
            # Generate random equiangular tight frames (ETFs)
            N, D = support_proto.shape
            U = torch.linalg.svd(support_proto.T)[-1]
            U = torch.nn.functional.pad(U, (0, 0, 0, D - N), mode='constant')
            M = (torch.eye(N)- (torch.ones(N, 1) @ torch.ones(1, N)).div(N)).mul(math.sqrt(N / (N - 1))).to(support_proto.device) @ U.T
            
            # calculate distance to each prototype
            logits.append(self.__batch_dist__(
                M,
                query_emb[current_query_num:current_query_num+sent_query_num], # [sentence_num (N-way), max_length, embedding_dim]
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            proto.append(M)
            #embs.append(query_emb[current_query_num:current_query_num+sent_query_num] @ T.T)

            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, None, proto, embs

    
    
    
