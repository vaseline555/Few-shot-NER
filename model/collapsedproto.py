import sys
sys.path.append('..')
import util
import torch

import math
import numpy as np
import ot
from collections import Counter

from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class CollapsedPrOTo(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, support_proto, transformed_proto, ot_mapping, T, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        kernel = ot.utils.kernel(Q, support_proto, method=ot_mapping.kernel, sigma=ot_mapping.sigma).to(support_proto.device)
        Q_transformed = kernel @ T
        return self.__dist__(transformed_proto.unsqueeze(0), Q.unsqueeze(1), 2)

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
                support['text_mask'][current_support_num: current_support_num+sent_support_num])
            
            # version 1) null projection for detecting 'O' class & equialngular tight frame conversion of other prototypes
            N, D = support_proto.shape
            u_list = []
            for i in range((N - 1) // 2):
                alpha = torch.randint(low=0, high=360, size=(1,))
                angle_to_radian = torch.tensor(math.pi / 180)
                s = torch.sin(alpha * angle_to_radian)
                c = torch.cos(alpha * angle_to_radian)
                rot = torch.tensor([[c, -s], [s, c]])
                u_list.append(rot)
            else:
                if (N - 1) % 2 == 1:
                    u_list.append(torch.tensor([1]))
                U = torch.block_diag(*u_list)
                U = torch.nn.functional.pad(U, (0, 0, 0, D - N + 1), mode='constant')
            M = (torch.eye(N - 1) - (torch.ones(N - 1, 1) @ torch.ones(1, N - 1)).div(N - 1)).mul(math.sqrt((N - 1) / (N - 2))) @ U.T
            target_proto = torch.cat([torch.zeros(1, D), M], dim=0).to(support_proto.device)
            
            """
            # version 2) null projection for 'O' class & equialngular tight frame conversion
            N, D = support_proto.shape
            u_list = []
            for i in range(N // 2):
                alpha = torch.randint(low=0, high=360, size=(1,))
                angle_to_radian = torch.tensor(math.pi / 180)
                s = torch.sin(alpha * angle_to_radian)
                c = torch.cos(alpha * angle_to_radian)
                rot = torch.tensor([[c, -s], [s, c]])
                u_list.append(rot)
            else:
                if N % 2 == 1:
                    u_list.append(torch.tensor([1]))
                U = torch.block_diag(*u_list)
                U = torch.nn.functional.pad(U, (0, 0, 0, D - N), mode='constant')
            M = (torch.eye(N) - (torch.ones(N, 1) @ torch.ones(1, N)).div(N)).mul((N / (N - 1))**0.5) @ U.T
            target_proto = M.to(support_proto.device)
            """
            # get Barycentric mapping using optimal transport
            bary_map = ot.da.MappingTransport(kernel="gaussian", mu=1, eta=1e-2, bias=False, norm=True)
            yt = np.ones(N); yt *= -1; yt[0] = 0
            ot_mapping = bary_map.fit(
                Xs=support_proto.detach().cpu().numpy(), 
                ys=np.arange(N), 
                Xt=target_proto.detach().cpu().numpy(), 
                yt=yt
            )
            
            kernel = ot.utils.kernel(support_proto, support_proto, method=ot_mapping.kernel, sigma=ot_mapping.sigma).to(support_proto.device)
            T = torch.tensor(ot_mapping.mapping_).to(support_proto.device)
            transformed_proto = kernel @ T

            # calculate distance to each prototype
            logits.append(self.__batch_dist__(
                support_proto,
                transformed_proto,
                ot_mapping,
                T,
                query_emb[current_query_num:current_query_num+sent_query_num], # [sentence_num (N-way), max_length, embedding_dim]
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            proto.append(transformed_proto)
            #embs.append(query_emb[current_query_num:current_query_num+sent_query_num] @ T.T)

            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, torch.tensor(ot_mapping.coupling_).float().to(logits.device).mean(), proto, embs

    
    
    
