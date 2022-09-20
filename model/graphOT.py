import sys
sys.path.append('..')
import util
import torch

import math
import numpy as np
import dgl
import ot

from torch import nn
from torch.nn import functional as F

from collections import Counter


    
class GraphOT(util.framework.FewShotNERModel):
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout(0.2)
        self.dot = dot
        
        # embedding dimension
        self.embedding_dim = 768
        
        # squashing k-NN graph
        self.K = 5

        # self attention uisng meassge passing
        self.mpnn_s = dgl.nn.pytorch.conv.GraphConv(self.embedding_dim, self.embedding_dim)
        self.mpnn_q = dgl.nn.pytorch.conv.GraphConv(self.embedding_dim, self.embedding_dim)

        # scross attention using message passing
        self.mpnn_sq = dgl.nn.pytorch.conv.GraphConv(self.embedding_dim, self.embedding_dim)
    
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

    def solve_GW(self, S, Q, p_S, p_Q, n_iter=5):
        p_S.requires_grad_(True)
        Q.requires_grad_(True)

        for _ in range(n_iter):
            loss = ot.gromov_wasserstein2(S, Q, p_S, p_Q)
            loss.backward()

            with torch.no_grad():
                grad = p_S.grad
                p_S -= grad * 1e-2   # step
                p_S.grad.zero_()
                p_S.data = ot.utils.proj_simplex(p_S)

                grad = Q.grad
                Q -= grad * 1e-2   # step
                Q.grad.zero_()
                Q.data = torch.clamp(Q, 0, 1)
        return Q

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


            # Self attention - Support set
            s_adj = s_emb.unsqueeze(1).sub(s_emb.unsqueeze(0)).pow(2).mean(2).mul(-1).div(2).exp()   # N*N*d -> N*N
            topk, indices = torch.topk(s_adj, self.K)
            mask = torch.zeros_like(s_adj)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).float()      # union, kNN graph
            s_adj = s_adj * mask.to(s_adj.device)
            s_graph = dgl.graph(torch.nonzero(s_adj, as_tuple=True))
            s_emb = self.mpnn_s(s_graph, s_emb, edge_weight=s_adj[torch.nonzero(s_adj, as_tuple=True)])

            # Self attention - Query set
            q_adj = q_emb.unsqueeze(1).sub(q_emb.unsqueeze(0)).pow(2).mean(2).mul(-1).div(2).exp()   # N*N*d -> N*N
            topk, indices = torch.topk(q_adj, self.K)
            mask = torch.zeros_like(q_adj)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).float()      # union, kNN graph
            q_adj = q_adj * mask.to(q_adj.device)
            q_graph = dgl.graph(torch.nonzero(q_adj, as_tuple=True))
            q_emb = self.mpnn_q(q_graph, q_emb, edge_weight=q_adj[torch.nonzero(q_adj, as_tuple=True)])

            # Cross attention - support and query set
            embs = torch.cat([s_emb, q_emb], 0)
            sq_adj = embs.unsqueeze(1).sub(embs.unsqueeze(0)).pow(2).mean(2).mul(-1).div(2).exp()   # N*N*d -> N*N
            topk, indices = torch.topk(sq_adj, self.K)
            mask = torch.zeros_like(sq_adj)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).float()      # union, kNN graph
            sq_adj = sq_adj * mask.to(sq_adj.device)
            sq_graph = dgl.graph(torch.nonzero(sq_adj, as_tuple=True))
            embs = self.mpnn_sq(sq_graph, embs, edge_weight=sq_adj[torch.nonzero(sq_adj, as_tuple=True)])
            s_emb, q_emb = embs[:len(s_emb)], embs[len(s_emb):]

            # Optimal transport for span prediction
            transductive_emd = ot.da.SinkhornTransport(reg_e=0.1, max_iter=10000)
            transductive_emd.fit(Xs=q_emb, Xt=s_emb, yt=s_label)
            q_emb = transductive_emd.transform(q_emb)         
            ent_loss += transductive_emd.cost_.mean()

            # Prototypical Networks
            support_proto = self.__get_proto__(s_emb, s_label)
            logit = self.__batch_dist__(support_proto, q_emb)


            """
            # Logit construction
            logit = torch.empty(len(q_emb), len(torch.unique(s_label))).to(q_emb.device)
            if sum(q_span_pred == 1) > 0:
                entity_proto = self.__get_proto__(s_emb[s_label != 0], s_label[s_label != 0])
                q_entity_logit = self.__batch_dist__(entity_proto, q_emb[q_span_pred != 0])
                logit[q_span_pred == 1] = torch.cat([q_entity_logit.min(1)[0].view(-1, 1), q_entity_logit], 1)
            if sum(q_span_pred == 0) > 0:
                o_proto = self.__get_proto__(s_emb, (s_label != 0).long())
                o_logit = self.__batch_dist__(o_proto, q_emb[q_span_pred == 0])[:, 0].view(-1, 1)
                logit[q_span_pred == 0] = torch.cat([o_logit, torch.ones_like(o_logit).mul(o_logit.min() - 1).repeat(1, s_label.max().long().item())], 1)
            """

            """
            # Graph matching for span detection
            q_adj = F.cosine_similarity(q_emb_selected.unsqueeze(1), q_emb_selected.unsqueeze(0), dim=2).mul(-1).add(1)

                        
            s_prior = torch.eye(len(torch.unique(s_label_selected))).to(q_adj.device)
            p_S = torch.rand(len(torch.unique(s_label_selected))).to(s_emb_selected.device)
            for l in torch.unique(s_label_selected).long():
                p_S[l] = (s_label_selected == l).sum()
            #S_adj = torch.eye(len(torch.unique(s_label_selected))).to(s_emb_selected.device)
            #p_S = torch.rand(len(torch.unique(s_label_selected))).to(s_emb_selected.device)
            #for l in torch.unique(s_label_selected).long():
            #    p_S[l] = (s_label_selected == l).sum()
            #p_S = torch.rand(len(s_emb_selected)).to(s_emb_selected.device)
            p_S /= p_S.sum()

            p_Q = torch.rand(len(q_emb_selected)).to(q_emb_selected.device)
            p_Q /= p_Q.sum()
            
            q_adj_estimated = self.solve_GW(s_prior.detach().clone(), q_adj.detach().clone(), p_S.detach().clone(), p_Q.detach().clone())
            q_graph = dgl.graph(torch.nonzero(q_adj_estimated, as_tuple=True))
            q_emb_selected = self.mpnn_q(q_graph, q_emb_selected, edge_weight=q_adj[torch.nonzero(q_adj_estimated, as_tuple=True)])

            
            support_proto = self.__get_proto__(s_emb_selected, s_label_selected)
            logit = self.__batch_dist__(support_proto, q_emb_selected)
            print(logit.max(1)[1])
            
            logit = torch.empty(len(q_emb_selected), len(torch.unique(s_label_selected))).to(q_emb_selected.device)
            if sum(q_span_pred == 1) > 0:
                entity_proto = self.__get_proto__(s_emb_selected[s_label_selected != 0], s_label_selected[s_label_selected != 0])
                q_entity_logit = self.__batch_dist__(entity_proto, q_emb_selected[q_span_pred != 0])
                logit[q_span_pred == 1] = torch.cat([q_entity_logit.min(1)[0].view(-1, 1), q_entity_logit], 1)
            if sum(q_span_pred == 0) > 0:
                o_logit = self.__batch_dist__(o_proto, q_emb_selected[q_span_pred == 0])[:, 0].view(-1, 1)
                logit[q_span_pred == 0] = torch.cat([o_logit, torch.ones_like(o_logit).mul(o_logit.min() - 1).repeat(1, s_label_selected.max().long().item())], 1)
            """

            """
            # Span prediction
            ## Coarsely predict span by optimal transport domain adaptation
            transductive_emd = ot.da.SinkhornTransport(reg_e=10)
            transductive_emd.fit(Xs=q_emb_selected, Xt=s_emb_selected, yt=(s_label_selected != 0).long())
            q_span_pred = transductive_emd.inverse_transform_labels((s_label_selected != 0).long()).max(1)[1].long()
            print(q_emb_selected.mean().round(decimals=4).item(), q_emb_selected.std().round(decimals=4).item(),"전")
            q_emb_selected = transductive_emd.transform(q_emb_selected)            
            print(q_emb_selected.mean().round(decimals=4).item(), q_emb_selected.std().round(decimals=4).item(),"후\n")
            """

            """
            logit = torch.empty(len(q_emb_selected), len(torch.unique(s_label_selected))).to(q_emb_selected.device)
            s_label_span = (s_label_selected != 0).long()

            #import pdb;pdb.set_trace()
            proto_O = s_emb_selected[s_label_span == 0].mean(0, keepdim=True)
            logit_O = self.__batch_dist__(proto_O, q_emb_selected[q_span_pred == 0])
            logit_O = logit_O.repeat(1, len(torch.unique(s_label_selected)))
            logit_O[:, 1:] = logit_O[:, 1:] - 1

            proto_entity = self.__get_proto__(s_emb_selected[s_label_span != 0], s_label_selected[s_label_span != 0])
            logit_entity = self.__batch_dist__(proto_entity, q_emb_selected[q_span_pred != 0])
            logit_entity = torch.cat([logit_entity.min(1, keepdim=True)[0].sub(1), logit_entity], 1)
            
            logit[q_span_pred == 0] = logit_O
            logit[q_span_pred == 1] = logit_entity
            """
            """
            ## Finely predict span by label propagation
            ### Construct a graph
            entity_embs = torch.cat([s_emb_selected, q_emb_selected], 0)
            entity_sim = F.cosine_similarity(entity_embs.unsqueeze(1), entity_embs.unsqueeze(0), dim=2).add(1).div(2)
            entity_sim = (entity_sim > torch.quantile(entity_sim, q=0.95)).float()

            
            ## Label propgation for correction
            graph = dgl.graph(torch.nonzero(entity_sim, as_tuple=True))
            graph = dgl.add_self_loop(graph)

            lp = dgl.nn.pytorch.utils.LabelPropagation(k=50, alpha=0.9, reset=True)
            spans_old = torch.cat([(s_label_selected != 0).float(), q_span_pred]).long()
            masks = torch.cat([torch.ones_like(s_label_selected), torch.zeros_like(q_span_pred)]).bool()
            spans_new = lp(graph, spans_old, masks).argmax(1)[masks == 0]
            q_span_pred[q_span_pred == 1] = spans_new
            
            D = entity_sim.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + np.finfo(float).eps))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, len(entity_embs))
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(len(entity_embs), 1)
            S = D1 * entity_sim * D2

            labels = torch.cat([(s_label_selected != 0).float(), q_span_pred]).long()
            labels = F.one_hot(labels).float()
            
            ## Label propagation 공식에 따라 logit 추정
            logit = torch.matmul(torch.inverse(torch.eye(len(labels)).to(entity_embs.device) - 0.9 * S + np.finfo(float).eps), labels)
            q_span_pred = logit[len(s_label_selected):].max(1)[1]
            """


            # Prototypical Networks
            #logit = torch.empty(len(q_emb_selected), len(torch.unique(s_label_selected))).to(q_emb_selected.device)

            #support_proto = self.__get_proto__(s_emb_selected, s_label_selected)
            #logit = self.__batch_dist__(support_proto, q_emb_selected)
            """
            logit_O = self.__batch_dist__(support_proto, q_emb_selected)
            logit[q_span_pred == 0] = logit_O[q_span_pred == 0]

            ## Get outside prototype
            support_proto = self.__get_proto__(s_emb_selected, s_label_selected)
            logit_entity = self.__batch_dist__(support_proto, transductive_emd.transform(q_emb_selected))
            logit[q_span_pred != 0] = logit_entity[q_span_pred != 0]
            """

            """
            # get logit
            o_proto = self.__get_proto__(s_emb_selected, (s_label_selected != 0).float())

            # Graph matching for span detection
            S_adj = torch.eye(2).to(s_emb_selected.device)
            p_S = torch.rand(len(S_adj)).to(s_emb_selected.device)wj
            p_S /= p_S.sum()

            Q_adj = F.cosine_similarity(q_emb_selected.unsqueeze(1), q_emb_selected.unsqueeze(0), dim=2).add(1).div(2)
            p_Q = torch.ones(len(q_emb_selected)).to(q_emb_selected.device).div(len(q_emb_selected))
            p_Q /= p_Q.sum()
            print("Before span", self.__batch_dist__(o_proto, q_emb_selected).max(1)[1])
            
            Q_adj_estimated = self.solve_GW(S_adj.detach().clone(), Q_adj.detach().clone(), p_S.detach().clone(), p_Q.detach().clone())
            graph = dgl.graph(torch.nonzero(Q_adj_estimated, as_tuple=True))
            graph = dgl.add_self_loop(graph)
            graph.ndata['feat'] = q_emb_selected
            graph = dgl.to_homogeneous(graph, ndata=['feat'])
            

            q_emb_selected = self.drop(F.relu(self.gconv1(graph, q_emb_selected)))
            q_span_pred = self.__batch_dist__(o_proto, q_emb_selected).max(1)[1]
            print("After span", self.__batch_dist__(o_proto, q_emb_selected).max(1)[1])

            entity_proto = self.__get_proto__(s_emb_selected[s_label_selected != 0], s_label_selected[s_label_selected != 0])
            q_entity_logit = self.__batch_dist__(entity_proto, q_emb_selected[q_span_pred != 0])

            logit = torch.empty(len(q_emb_selected), len(torch.unique(s_label_selected))).to(q_emb_selected.device)
            logit[q_span_pred == 1] = torch.cat([q_entity_logit.min(1)[0].view(-1, 1), q_entity_logit], 1)
            o_logit = self.__batch_dist__(o_proto, q_emb_selected[q_span_pred == 0])[:, 0].view(-1, 1)
            logit[q_span_pred == 0] = torch.cat([o_logit, torch.ones_like(o_logit).mul(o_logit.min() - 1).repeat(1, s_label_selected.max().long().item())], 1)
            """
            
            logits.append(logit)
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)        
        _, pred = torch.max(logits, 1)
        #proto = torch.stack(proto, 0) # save epoisode-wise prototypes
        return logits, pred, ent_loss, proto, embs

    
    

    
