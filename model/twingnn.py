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
from sklearn.cluster import MeanShift
from sklearn.neighbors import NeighborhoodComponentsAnalysis, RadiusNeighborsClassifier, KNeighborsClassifier

from collections import Counter


    
class TwinGNN(util.framework.FewShotNERModel):
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
                                                        
                                                              
            
            
            
            # 2) Class graph construction
            ## construct edges
            s_src = torch.arange(len(s_label_selected)).to(s_label_selected.device).add(2) # 2 for 0 and 1
            s_dst = s_label_selected.clone()
            s_dst[s_dst != 0] = 1
            
            #nca = NeighborhoodComponentsAnalysis().fit(s_emb_selected.detach().cpu().numpy(), s_dst.detach().cpu().numpy())
            nn = KNeighborsClassifier(
                algorithm='brute',
                weights='distance',
                n_neighbors=self.K,
                metric='mahalanobis',
                metric_params={'VI': np.cov(s_emb_selected.detach().cpu().numpy().T)},
                n_jobs=-1
            ).fit(s_emb_selected.detach().cpu().numpy(), s_label_selected.detach().cpu().numpy())

            #proj = self.FKT(s_emb_selected, s_dst, self.embedding_dim).detach().cpu().numpy()
            q_src = torch.arange(len(q_emb_selected)).to(s_label_selected.device).add(len(s_label_selected)).add(2)
            q_dst = torch.tensor(nn.predict(q_emb_selected.detach().cpu().numpy())).to(q_src.device)
            q_dst[q_dst != 0] = 1
            print(Counter(q_dst.detach().cpu().numpy()))
            
            ## convert into DGL data structure 
            graph = dgl.graph((torch.cat([torch.tensor([0, 1]).to(s_src.device), s_src, q_src]).tolist(), torch.cat([torch.tensor([0, 1]).to(s_src.device), s_dst, q_dst]).tolist())).to(s_emb.device)
            graph = dgl.add_self_loop(graph)

            graph.ndata['feat'] = torch.cat(
                [
                    torch.cat(
                        [
                            s_emb_selected[s_label_selected == 0],
                            q_emb_selected[q_dst == 0]
                        ], 0
                    ).mean(0, keepdim=True), 
                    torch.cat(
                        [
                            s_emb_selected[s_label_selected == 1],
                            q_emb_selected[q_dst == 1]
                        ], 0
                    ).mean(0, keepdim=True), 
                    s_emb_selected, 
                    q_emb_selected
                ], 0
            )
            graph = dgl.to_homogeneous(graph, ndata=['feat'])
            span_embs = self.drop(self.span_gconv(graph, torch.cat(
                [
                    torch.cat(
                        [
                            s_emb_selected[s_label_selected == 0],
                            q_emb_selected[q_dst == 0]
                        ], 0
                    ).mean(0, keepdim=True), 
                    torch.cat(
                        [
                            s_emb_selected[s_label_selected == 1],
                            q_emb_selected[q_dst == 1]
                        ], 0
                    ).mean(0, keepdim=True), 
                    s_emb_selected, 
                    q_emb_selected
                ], 0
            )))[2:]
            
            
            
            
            
            # 3) Task graph construction
            ms = MeanShift(bandwidth=1, n_jobs=-1).fit(torch.cat([s_emb_selected, q_emb_selected], 0).detach().cpu().numpy())
            clustered_labels = ms.fit_predict(torch.cat([s_emb_selected, q_emb_selected], 0).detach().cpu().numpy())
            
            s_src = torch.arange(len(s_label_selected)).to(s_label_selected.device).add(len(np.unique(clustered_labels)))
            s_dst = torch.tensor(clustered_labels)[:len(s_emb_selected)].to(s_label_selected.device)
            
            q_src = torch.arange(len(q_emb_selected)).to(s_label_selected.device).add(len(s_label_selected)).add(len(np.unique(clustered_labels)))
            q_dst = torch.tensor(clustered_labels)[len(s_emb_selected):].to(s_label_selected.device)
            
            ## convert into DGL data structure
            graph = dgl.graph((torch.cat([torch.arange(len(np.unique(clustered_labels))).to(s_src.device), s_src, q_src]).tolist(), torch.cat([torch.arange(len(np.unique(clustered_labels))).to(s_src.device),s_dst, q_dst]).tolist())).to(s_emb.device)
            graph = dgl.add_self_loop(graph)
            
            
            graph.ndata['feat'] = torch.cat([torch.tensor(ms.cluster_centers_).to(s_emb.device), s_emb_selected, q_emb_selected], 0)
            graph = dgl.to_homogeneous(graph, ndata=['feat'])
            task_embs = self.drop(self.task_gconv(graph, torch.cat([torch.tensor(ms.cluster_centers_).to(s_emb.device), s_emb_selected, q_emb_selected], 0)))[len(np.unique(clustered_labels)):]

            
                                                 
            # 4) Merge embeddings
            #logit = self.proj(F.relu(torch.cat([span_embs, task_embs], 1)))[len(s_emb_selected):]
            embs = span_embs * task_embs
            support_proto = self.__get_proto__(embs[:len(s_label_selected)], s_label_selected)
            logit = self.__batch_dist__(support_proto, embs[len(s_label_selected):])
              
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)        
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, None, proto, embs

    
    

    
