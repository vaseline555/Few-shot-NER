import sys
sys.path.append('..')
import util
import torch

import scipy
import math
import numpy as np
import ot
import dgl
from torch import nn
from torch.nn import functional as F


def null_projection(A, rcond=None):
    ut, st, vht = torch.svd(A, some=False, compute_uv=True)
    vht = vht.T        
    Mt, Nt = ut.shape[0], vht.shape[1] 
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    numt= torch.sum(st > tolt, dtype=int)
    nullspace = vht[numt:,:].T.conj()
    # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
    return nullspace

class CollapsedPrOTo(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1, N=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.to_qkv = nn.Linear(768, 128 * 3)
        self.mha = nn.MultiheadAttention(embed_dim=128, num_heads=16, batch_first=True)
        self.scaler = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
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
            
            # Version 1) Basic label propagation
            """
            # 그냥 시도: zero division error
            # degree normalization 빼고 시도: zero division error
            # label 0에 해당하는 logit을 support set에 있는 0의 개수만큼 나눠주기: zero division error
            # `` & degree normalization 빼고: zero division error
            
            # nullproj: zero division error
            # nul_proj & degree normalization빼고: zero division error
            # nullproj & label 0에 해당하는 logit을 support set에 있는 0의 개수만큼 나눠주기: zero division error
            # nullproj & degree normalization빼고 & label 0에 해당하는 logit을 support set에 있는 0의 개수만큼 나눠주기: zero division error
            
            # gcn: zero division error
            # degree normalization 빼고 시도: zero division error
            # gcn & logit 나눠주기: zero division error
            # gcn & logit 나눠주기 & degree normalization 빼고: zero division error

            # 18!
            """
            """
            #null_proj = null_projection(s_emb_selected[s_label_selected == 0]).to(s_emb_selected.device)
            embs = torch.cat([s_emb_selected, q_emb_selected], 0) #@ null_proj
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > self.threshold, adj, 0)
            
            graph = dgl.graph(torch.nonzero(adj, as_tuple=True))
            graph = dgl.add_self_loop(graph)
            graph.ndata['h'] = embs
            graph = dgl.to_homogeneous(graph, ndata=['h'])
            embs = self.gconv(graph, embs)
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > self.threshold, adj, 0)
            
            D = adj.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + np.finfo(float).eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(embs))
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(embs), 1)
            S = D1 * adj * D2
            
            labels = torch.cat([s_label_selected, torch.zeros(len(q_emb_selected)).to(embs.device)]).long()
            labels = F.one_hot(labels).float()

            logit = torch.matmul(torch.inverse(torch.eye(len(labels)).to(embs.device) - self.alpha * S + np.finfo(float).eps
), labels)
            logit = logit[len(s_label_selected):]
            logit[:, 0] = logit[:, 0] / (s_label_selected == 0).sum()
            """
            """
            # Version 2) Basic + transform embedding - zeor division error (also holds when adj thresholding)
            null_proj = null_projection(s_emb_selected[s_label_selected == 0]).to(s_emb_selected.device)
            embs = torch.cat([s_emb_selected, q_emb_selected], 0) #@ null_proj
            #embs = F.pad(embs, (0, 768 - null_proj.shape[1]))
            qkv = self.to_qkv(embs)
            embs = self.mha(qkv[:, :768].unsqueeze(1), qkv[:, 768:-768].unsqueeze(1), qkv[:, -768:].unsqueeze(1))[0].squeeze()

            emb1 = torch.unsqueeze(embs, 1)#[:, :null_proj.shape[1]] # N*1*d
            emb2 = torch.unsqueeze(embs, 0)#[:, :null_proj.shape[1]] # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > torch.quantile(adj, 0.9), adj, 0)
            
            D = adj.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + np.finfo(float).eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(embs))
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(embs), 1)
            S = D1 * adj * D2
            
            labels = torch.cat([s_label_selected, torch.zeros(len(q_emb_selected)).to(embs.device)]).long()
            labels = F.one_hot(labels).float()

            logit = torch.matmul(torch.inverse(torch.eye(len(labels)).to(embs.device) - 0.5 * S + np.finfo(float).eps
), labels)
            logit = logit[len(s_label_selected):]
            """
            """
            # Version 3) Basic + scaling - zeor division error (also holds when adj thresholding)
            #null_proj = null_projection(s_emb_selected[s_label_selected == 0]).to(s_emb_selected.device)
            #embs = torch.cat([s_emb_selected, q_emb_selected], 0) #@ null_proj
            s_o_proto = s_emb_selected[s_label_selected == 0].mean(0).view(1, -1)
            embs = torch.cat([s_o_proto, s_emb_selected[s_label_selected != 0], q_emb_selected], 0)
            #embs = F.pad(embs, (0, 768 - null_proj.shape[1]))
            qkv = self.to_qkv(embs)
            embs = self.mha(qkv[:, :128].unsqueeze(1), qkv[:, 128:-128].unsqueeze(1), qkv[:, -128:].unsqueeze(1))[0].squeeze()
            scaling_consts = self.scaler(embs)
            #scaling_consts = F.pad(scaling_consts, (0, 5), 'constant', 1)
            embs = embs / scaling_consts
            
            emb1 = torch.unsqueeze(embs, 1)#[:, :null_proj.shape[1]] # N*1*d
            emb2 = torch.unsqueeze(embs, 0)#[:, :null_proj.shape[1]] # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > torch.quantile(adj, 0.9), adj, 0)
            #adj = ((emb1 - emb2)**2).mean(2).div(2).mul(-1).exp()   # N*N*d -> N*N
            
            D = adj.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + np.finfo(float).eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(embs))
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(embs), 1)
            S = D1 * adj * D2
            
            #labels = torch.cat([s_label_selected, torch.zeros(len(q_emb_selected)).to(embs.device)]).long()
            labels = torch.cat([torch.zeros(1).long().to(embs.device), s_label_selected[s_label_selected != 0], torch.zeros(len(q_emb_selected)).to(embs.device)]).long()
            labels = F.one_hot(labels).float()

            logit = torch.matmul(torch.inverse(torch.eye(len(labels)).to(embs.device) - self.alpha * S + np.finfo(float).eps
), labels)
            logit[:, 0] = logit[:, 0] / 20
            logit = logit[1 + len(s_emb_selected[s_label_selected != 0]):]#logit[len(s_label_selected):]
            """
            """
            # Version 4) Not using all O - using O representative instead - zero division error (also holds when adj thresholding)
            null_proj = null_projection(s_emb_selected[s_label_selected == 0]).to(s_emb_selected.device)
            s_o_proto = s_emb_selected[s_label_selected == 0].mean(0).view(1, -1)
            embs = torch.cat([s_o_proto, s_emb_selected[s_label_selected != 0], q_emb_selected], 0)
            embs = torch.cat([s_emb_selected, q_emb_selected], 0) @ null_proj
            embs = F.pad(embs, (0, 768 - [:, :null_proj.shape[1]]))
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > torch.quantile(adj, 0.9), adj, 0)

            D = adj.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / D)
            S = torch.diag(D_sqrt_inv) @ adj @ torch.diag(D_sqrt_inv)
            
            labels = torch.cat([torch.zeros(1).to(embs.device), s_label_selected[s_label_selected != 0], torch.zeros(len(q_emb_selected)).to(embs.device)]).long()
            labels = F.one_hot(labels).float()

            logit = torch.matmul(torch.inverse(torch.eye(len(labels)).to(embs.device) - 0.5 * S + np.finfo(float).eps
), labels)
            logit = logit[1 + len(s_emb_selected[s_label_selected != 0]):]
            """
            """
            # Version 5) Not using all O - using O representative instead + transform embedding
            null_proj = null_projection(s_emb_selected[s_label_selected == 0]).to(s_emb_selected.device)
            s_o_proto = s_emb_selected[s_label_selected == 0].mean(0).view(1, -1)
            embs = torch.cat([s_o_proto, s_emb_selected[s_label_selected != 0], q_emb_selected], 0)
            embs = torch.cat([s_emb_selected, q_emb_selected], 0) @ null_proj
            qkv = self.to_qkv(embs)
            embs = self.mha(qkv[:, :768].unsqueeze(1), qkv[:, 768:-768].unsqueeze(1), qkv[:, -768:].unsqueeze(1))[0].squeeze()
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj = F.cosine_similarity(emb1, emb2, dim=2).add(1.).div(2.)
            adj = torch.where(adj > torch.quantile(adj, 0.9), adj, 0)

            D = adj.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + np.finfo(float).eps))
            S = torch.diag(D_sqrt_inv) @ adj @ torch.diag(D_sqrt_inv)
            
            labels = torch.cat([torch.zeros(1).to(embs.device), s_label_selected[s_label_selected != 0], torch.zeros(len(q_emb_selected)).to(embs.device)]).long()
            labels = F.one_hot(labels).float()

            logit = torch.matmul(torch.inverse(torch.eye(len(labels)).to(embs.device) - 0.5 * S + np.finfo(float).eps
), labels)
            logit = logit[1 + len(s_emb_selected[s_label_selected != 0]):]
            """
            
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, None, proto, embs

    
    
    
