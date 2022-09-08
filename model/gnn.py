import sys
sys.path.append('..')
import util
import torch

import scipy
import math
import numpy as np
import ot
import dgl
import gpushift

from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression as LR


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

class GraphProto(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.sim = torch.nn.CosineSimilarity(dim=2)
        self.threshold = nn.Parameter(torch.tensor([0.5]))
        self.offset = nn.Parameter(torch.tensor([1e-4]))
        self.gconv = dgl.nn.pytorch.conv.GraphConv(768, 256)
        self.gconv2 = dgl.nn.pytorch.conv.GraphConv(256, 64)
        
        
        self.embedding_dim = 128
        self.label_embedding = nn.Embedding(6, self.embedding_dim, scale_grad_by_freq=True)
        self.embedder = nn.Sequential(
            #nn.Linear(768, 128),
            nn.BatchNorm1d(self.embedding_dim, track_running_stats=False),
            nn.ReLU()
        )
        
        self.ln_k = nn.LayerNorm(self.embedding_dim)
        self.ln_v = nn.LayerNorm(self.embedding_dim)
        self.ln_q = nn.LayerNorm(self.embedding_dim)
        
        self.to_k = nn.Linear(768, self.embedding_dim , bias=False)
        self.to_v = nn.Linear(768, self.embedding_dim , bias=False)
        self.to_q = nn.Linear(768, self.embedding_dim, bias=False)
        
        self.span_detector = nn.Sequential(
            nn.BatchNorm1d(self.embedding_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.mha = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=16, batch_first=True)
        
        
        
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
    
    def __get_ot_loss__(self, emb, label, etf):
        eigs = []
        for l in torch.unique(label):
            emb_centered = (emb[label == l] - emb[label == l].mean(0)) / len(emb[label == l])
            emb_eig = torch.linalg.eigh(emb_centered.T @ emb_centered)[-1][:, 0]
            eigs.append(emb_eig)
        else:
            eigs = torch.stack(eigs)
        C = ot.dist(eigs, etf)
        ot_emd = ot.emd(torch.stack([sum(label == l) / len(label) for l in torch.unique(label)]), torch.ones(len(etf)).div(len(etf)).to(emb.device), C)
        return (torch.tensor(ot_emd).to(emb.device) * C).sum()
    
    def transform_and_normalize(self, vecs, kernel, bias):
        """
            Applying transformation then standardize
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias) @ kernel
        return self.normalize(vecs)

    def normalize(self, vecs):
        """
            Standardization
        """
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def compute_kernel_bias(self, vecs):
        """
        Calculate Kernal & Bias for the final transformation - y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(0, keepdims=True)
        cov = torch.cov(vecs.T).to(vecs.device)
        u, s, vh = torch.linalg.svd(cov.add(torch.eye(len(cov)).to(cov.device)))
        W = u @ torch.diag(s**0.5) @ u.T
        W = torch.linalg.inv(W.T)
        return W, -mu

    def batch_cov(self, points):
        B, N, D = points.size()
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return bcov  # (B, D, D)

    def FKT(self, X, y, n_components):
        X1, X2 = X[y == 0], X[y != 0]
        m, m1, m2 = X.mean(0), X1.mean(0), X2.mean(0)
        Hb = torch.stack([math.sqrt(len(X1)) * (m1 - m), math.sqrt(len(X2)) * (m2 - m)], 1)
        Ht = torch.cat([X1, X2], 0).T - m.reshape(-1, 1)
        Q, R = torch.linalg.qr(Ht)
        S_tilde_t = R @ R.T
        Z = Q.T @ Hb
        S_tilde_b = Z @ Z.T
        lam, V = torch.linalg.eigh(torch.linalg.inv(S_tilde_t) @ S_tilde_b)
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
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
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
            
            # Reduce BERT embedding dimension by ZCA whitening
            embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            kernel, bias = self.compute_kernel_bias(embs)
            kernel = kernel[:, :self.embedding_dim]
            embs = self.transform_and_normalize(embs, kernel=kernel, bias=bias)
            s_emb_selected = embs[:len(s_label_selected)]
            q_emb_selected = embs[len(s_label_selected):]
            
            # Transform into task-specific embeddings            
            s_label_embs = self.label_embedding(s_label_selected.long())
            q_label_embs = self.label_embedding.weight.mean(0).unsqueeze(0).repeat(len(q_emb_selected), 1)
            s_emb_selected = s_emb_selected + s_label_embs
            q_emb_selected = q_emb_selected + q_label_embs

            # Scale embedding
            s_emb_selected = self.embedder(s_emb_selected)
            q_emb_selected = self.embedder(q_emb_selected)
            
            # Get query, key (from S) and value (from Q) for cross-attention
            #Q, K, V = self.to_q(q_emb_selected), self.to_k(s_emb_selected), self.to_v(s_emb_selected)
            #q_emb_selected = self.mha(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))[0].squeeze()
            
            """
            # Fit support span
            S_span_pred = self.span_detector(s_emb_selected)
            span_loss = span_loss + nn.BCEWithLogitsLoss()(S_span_pred.squeeze(), (s_label_selected != 0).float())
            
            # Get query span
            q_prob = self.span_detector(q_emb_selected).sigmoid()
            q_prob = q_prob / q_prob.max()
            Q_span_pred = (q_prob > S_span_pred.sigmoid().mean()).long().squeeze()
            
            # Get prototype for possible entities
            support_proto = self.__get_proto__(
                s_emb_selected[s_label_selected != 0], 
                s_label_selected[s_label_selected != 0]
            )
            
            # calculate distance to each prototype
            logit = self.__batch_dist__(
                support_proto,
                q_emb_selected
            )
            o_offset = logit[Q_span_pred == 1].max().add(0.1).item()
            logit = torch.cat([torch.ones(len(logit)).to(logit.device).mul(logit[Q_span_pred == 1].min()).unsqueeze(1), logit], 1)
            logit[Q_span_pred == 0, 0] = o_offset
            
            """
            """
            embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            
            # transductive learning
            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            
            ## Euclidean adjcency matrix
            #bandwidth = 1
            #W = emb1.sub(emb2).pow(2).mean(2).mul(-1).div(2 * bandwidth**2).exp()   # N*N*d -> N*N
            
            ## Cosine similarity adjacency matrix
            W = self.sim(emb1, emb2).add(1.).div(2.)
            
            K = 2
            topk, indices = torch.topk(W, K)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).float()      # union, kNN graph
            W = W * mask.to(W.device)

            ## normalize
            D = W.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(embs))
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(embs), 1)
            adj = D1 * W * D2
            
            
            # construct graph
            graph = dgl.graph(torch.nonzero(adj, as_tuple=True))
            graph = dgl.add_self_loop(graph)
            graph.ndata['h'] = embs
            graph = dgl.to_homogeneous(graph, ndata=['h'])
       
            embs = F.relu(self.gconv(graph, embs))
            embs = self.gconv2(graph, embs)
            
            # get prototype
            support_proto = self.__get_proto__(
                embs[:len(s_label_selected)], 
                s_label_selected
            )
             
            # calculate distance to each prototype
            logit = self.__batch_dist__(
                support_proto,
                embs[len(s_label_selected):]
            )
            
            """
            # 0 - B) Construct transductive graph
            mean_shift = gpushift.MeanShift(
                n_iter=50,  
                kernel='gaussian',
                bandwidth=0.1,
                use_keops=False
            )
            s_o = s_emb_selected[s_label_selected == 0]
            o_clustered = mean_shift(s_o.unsqueeze(0)).squeeze()
            num_o = len(o_clustered)
            ent_loss += HLoss()(o_clustered)
            """
            embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            s_label_embs = self.label_embedding(s_label_selected.long())
            q_label_embs = self.label_embedding.weight.mean(0).unsqueeze(0).repeat(len(q_emb_selected), 1)
            q_label_embs = q_label_embs.add(torch.randn_like(q_label_embs))
            labels = torch.cat([s_label_embs, q_label_embs], 0)
            embs = embs + labels

            emb1 = torch.unsqueeze(embs, 1) # N*1*d
            emb2 = torch.unsqueeze(embs, 0) # 1*N*d
            adj = self.sim(emb1, emb2).add(1.).div(2.)
            adj = torch.where(adj > self.threshold, adj, 0)
            
            graph = dgl.graph(torch.nonzero(adj, as_tuple=True))
            graph = dgl.add_self_loop(graph)
            graph.ndata['h'] = embs
            graph = dgl.to_homogeneous(graph, ndata=['h'])
       
            embs = self.gconv(graph, self.drop(embs))
            embs = self.gconv2(graph, self.drop(embs))
            
            # version 1) Fukunaga-Koontz transformation
            o_label = s_label_selected.clone()
            o_label[o_label != 0] = 1
            embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            fkt_proj = self.FKT(embs[:len(s_emb_selected)], (~o_label.bool()).long(), 256)
            embs = embs @ fkt_proj
            """
        
            # 0 - C) prototypical networks
            support_proto = self.__get_proto__(
                s_emb_selected[s_label_selected != 0], 
                s_label_selected[s_label_selected != 0]
            )
            
            # calculate distance to each prototype
            logit = self.__batch_dist__(
                torch.cat([o_clustered, support_proto], 0),
                q_emb_selected
            )
            collapsed_logit = logit[:, :num_o].max(1)[0]
            logit = torch.cat([collapsed_logit.view(-1, 1), logit[:, num_o:]], dim=1)
            
            """
            # version 2) sieve out 'O' first
            o_label = s_label_selected.clone()
            o_label[o_label != 0] = 1
            clf = LR(solver='lbfgs', max_iter=1000).fit(embs[:len(s_label_selected)].detach().cpu().numpy(), o_label.detach().cpu().numpy())
            o_pred = clf.predict(embs[len(s_label_selected):].detach().cpu().numpy())
            
            support_proto = self.__get_proto__(
                embs[:len(s_label_selected)][s_label_selected != 0], 
                s_label_selected[s_label_selected != 0]
            )
            
            # calculate distance to each prototype
            logit = self.__batch_dist__(
                support_proto,
                embs[len(s_label_selected):]
            )
            
            logit[o_pred == 0] = torch.zeros(logit.shape[1]).to(logit.device)
            logit_o = torch.zeros(len(logit)).to(logit.device).add(logit.min() + self.offset)
            logit_o[o_pred == 0] = logit.max()
            logit = torch.cat([logit_o.view(-1, 1), logit], 1)
            
            """
            
            """
            # version 3) ETF (NC)
            support_proto = self.__get_proto__(
                embs[:len(s_label_selected)], 
                s_label_selected
            )
             
            # calculate distance to each prototype
            logit = self.__batch_dist__(
                support_proto,
                embs[len(s_label_selected):]
            )
            """
            """
            with torch.no_grad():
                # calculate ot loss
                N, D = support_proto.shape
                U = torch.linalg.svd(support_proto.T)[-1]
                U = torch.nn.functional.pad(U, (0, 0, 0, D - N), mode='constant')
                etfs = (torch.eye(N)- (torch.ones(N, 1) @ torch.ones(1, N)).div(N)).mul(math.sqrt(N / (N - 1))).to(support_proto.device) @ U.T
            
            ot_loss += self.__get_ot_loss__(embs, torch.cat([s_label_selected, torch.max(logit, 1)[-1]], 0), etfs)
            """
            """
            # version 4) Using O prototype
            s_o_proto = embs[:len(s_label_selected)][s_label_selected == 0].mean(0).view(1, -1)
            s_o_proto_cov = self.batch_cov(embs[:len(s_label_selected)][s_label_selected == 0].unsqueeze(1))
            embs = torch.cat([s_o_proto, embs[:len(s_label_selected)][s_label_selected != 0], embs[len(s_label_selected):]], 0)
            
            labels = torch.cat([torch.zeros(1).long().to(embs.device), s_label_selected[s_label_selected != 0]]).long()
            
            support_proto = self.__get_proto__(
                embs[:sum(s_label_selected != 0) + 1], 
                labels
            )
            logit = self.__batch_dist__(
                support_proto,
                embs[sum(s_label_selected != 0) + 1:],
                s_o_proto_cov
            )
            
            """
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, ent_loss, proto, embs

    
    

    
