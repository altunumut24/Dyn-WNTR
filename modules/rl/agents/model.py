import itertools
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchviz import make_dot
#from s2v_wdn_dqn.utils import visualize_pytorch_graph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class QNetwork(nn.Module):
    def __init__(self, embedding_layers, n_node_features, n_edge_features, embed_dim=256, bias=False, normalize=False, use_new_edge_q_layer=False):
        super().__init__()

        self.embedding_layers = embedding_layers
        self.embed_dim = embed_dim
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        self.node_features_embedding_layer = NodeFeaturesEmbeddingLayer(embed_dim, n_node_features, bias)
        self.edge_features_embedding_layer = EdgeFeaturesEmbeddingLayer(embed_dim, n_edge_features, bias)
        self.embedding_layer = EmbeddingLayer(n_node_features=n_node_features, embed_dim=embed_dim, bias=bias)
        self.use_new_edge_q_layer = use_new_edge_q_layer

        if self.use_new_edge_q_layer:
            self.q_layer = EdgeQLayer(embed_dim=embed_dim, bias=bias, normalize=normalize)
        else:
            self.q_layer = QLayer(embed_dim=embed_dim, bias=bias, normalize=normalize)

    def forward(self, state, edge_features, edges_ij):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if self.n_edge_features > 0 and edge_features.dim() == 2:
            edge_features = edge_features.unsqueeze(0)

        #B, N = state.size(0), state.size(1)
        #E = edges_ij.size(0)

        # adj = state[:, :, self.n_node_features:(self.n_node_features + n_vertices)]
        # edge_features = state[:, :, (self.n_node_features + n_vertices):]

        # calculate node embeddings        
        node_features_embeddings = self.node_features_embedding_layer(state)
        embeddings = node_features_embeddings

        for _ in range(self.embedding_layers):
            edge_features_embeddings = self.edge_features_embedding_layer(embeddings, edges_ij, edge_features)
            embeddings = self.embedding_layer(embeddings, edges_ij, node_features_embeddings, edge_features_embeddings)

        # calculate \hat{Q} based on embeddings and given vertices
        if self.use_new_edge_q_layer:
            q_hat = self.q_layer(embeddings, edges_ij)
        else:
            q_hat = self.q_layer(embeddings)
        return q_hat


class NodeFeaturesEmbeddingLayer(nn.Module):
    """
    Calculate the theta1 component
    """
    def __init__(self, embed_dim, n_node_features, bias=False):
        super().__init__()
        self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)

    def forward(self, node_features):
        # node_features.shape = (batch_size, n_vertices, n_node_features)
        # ret.shape = (batch_size, n_vertices, embed_dim)
        ret = self.theta1(node_features)
        return ret

class EdgeFeaturesEmbeddingLayer(nn.Module):
    def __init__(self, embed_dim, n_edge_features, bias=False):
        super().__init__()
        self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias)

    def forward(self, embeddings, edges_ij, edge_features):
        """
        embeddings:    (B, N, D)
        edges_ij:      (E, 2) long tensor (canonical undirected edges)
        edge_features: (B, E, F_edge)
        returns:       (B, N, D) edge-driven contribution
        """
        B, N, D = embeddings.shape
        E = edges_ij.size(0)
        u = edges_ij[:, 0]
        v = edges_ij[:, 1]

        # Project edge features then scatter-sum to endpoints
        x4 = F.leaky_relu(self.theta4(edge_features))      # (B, E, D)

        msg = embeddings.new_zeros(B, N, D)
        msg.index_add_(1, u, x4)                           # add to node u
        msg.index_add_(1, v, x4)                           # add to node v (undirected)

        return self.theta3(msg)                            # (B, N, D)

class EmbeddingLayer(nn.Module):
    def __init__(self, n_node_features, embed_dim, bias=False):
        super().__init__()
        self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)
        self.theta2 = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, prev_embeddings, edges_ij, node_features_embeddings, edge_features_embeddings):
        """
        prev_embeddings:         (B, N, D)
        edges_ij:                (E, 2)
        node_features_embeddings:(B, N, D)  [your theta1(node_feats)]
        edge_features_embeddings:(B, N, D)  [from EdgeFeaturesEmbeddingLayer]
        """
        B, N, D = prev_embeddings.shape
        u = edges_ij[:, 0]
        v = edges_ij[:, 1]

        # neighbor sum of node embeddings (two-way since undirected)
        nbr = prev_embeddings.new_zeros(B, N, D)
        nbr.index_add_(1, u, prev_embeddings[:, v, :])     # j -> i
        nbr.index_add_(1, v, prev_embeddings[:, u, :])     # i -> j
        x2 = self.theta2(nbr)

        ret = F.leaky_relu(node_features_embeddings + x2 + edge_features_embeddings)
        return ret

class EdgeQLayer(nn.Module):
    def __init__(self, embed_dim, bias=False, normalize=False):
        super().__init__()
        self.theta5 = nn.Linear(3*embed_dim, 1, bias=bias)  # [global, i, j]
        self.theta6 = nn.Linear(embed_dim, embed_dim, bias=bias)  # global proj
        self.theta7 = nn.Linear(embed_dim, embed_dim, bias=bias)  # node proj
        self.noop   = nn.Linear(embed_dim, 1, bias=bias)
        self.normalize = normalize

    def forward(self, embeddings, edges_ij):
        # embeddings: (B,N,D); edges_ij: (E,2) long
        B, N, D = embeddings.shape
        u = edges_ij[:,0]
        v = edges_ij[:,1]

        g = embeddings.sum(1)                        # (B,D)
        if self.normalize: g = g / N
        g_proj = self.theta6(g)                      # (B,D)

        node_proj = self.theta7(embeddings)          # (B,N,D)
        i_proj = node_proj[:, u, :]                  # (B,E,D)
        j_proj = node_proj[:, v, :]                  # (B,E,D)
        g_tile = g_proj.unsqueeze(1).expand(-1, i_proj.size(1), -1)  # (B,E,D)

        feats = torch.cat([g_tile, i_proj, j_proj], dim=-1)          # (B,E,3D)
        feats = torch.nn.functional.leaky_relu(feats)
        edge_q = self.theta5(feats).squeeze(-1)                       # (B,E)
        noop_q = self.noop(g)                                         # (B,1)
        return torch.cat([edge_q, noop_q], dim=1)                     # (B,E+1)

class QLayer(nn.Module):
    """
    Given node embeddings, calculate Q_hat for all vertices
    """
    def __init__(self, embed_dim, bias=False, normalize=False):
        super().__init__()
        self.theta5 = nn.Linear(3*embed_dim, 1, bias=bias)
        self.theta6 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta7 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.noop = nn.Linear(embed_dim, 1, bias=bias)  # No-op layer for compatibility
        self.normalize = normalize

    def forward(self, embeddings):
        # embeddings: (batch_size, N, D)
        B, N, D = embeddings.shape

        # 1) Global summary
        g = embeddings.sum(dim=1)                  # (B, D)
        if self.normalize:
            g = g / N
        g_proj = self.theta6(g)                    # (B, D)

        # 2) Pairwise i→j embeddings
        #    i_embed[b,i,j] = embeddings[b,i,:]
        i_embed = embeddings.unsqueeze(2).expand(-1, N, N, -1)  # (B,N,N,D)
        #    j_embed[b,i,j] = embeddings[b,j,:]
        j_embed = embeddings.unsqueeze(1).expand(-1, N, N, -1)  # (B,N,N,D)

        # 3) Project them
        i_proj = self.theta7(i_embed)              # (B,N,N,D)
        j_proj = self.theta7(j_embed)              # (B,N,N,D)

        # 4) Tile global proj over all (i,j)
        g_tile = g_proj.view(B,1,1,D).expand(-1, N, N, -1)  # (B,N,N,D)

        # 5) Concatenate [g_tile, i_proj, j_proj] → (B,N,N,3D)
        features = torch.cat([g_tile, i_proj, j_proj], dim=-1)
        features = nn.LeakyReLU()(features)

        # 6) Compute edge-Q and flatten to (B, N*N)
        edge_q = self.theta5(features).squeeze(-1).view(B, -1)  # (B, N, N) -> (B, N*N) 
        # 7) Compute no-op Q → (B,1)
        noop_q = self.noop(g)                      # (B, 1)

        # 8) Final vector → (B, N*N + 1)
        return torch.cat([edge_q, noop_q], dim=1)


    def old_forward(self, embeddings):
        # embeddings.shape = (batch_size, n_vertices, embed_dim)
        # sum_embeddings.shape = (batch_size, embed_dim)
        # x6.shape = (batch_size, embed_dim)
        sum_embeddings = embeddings.sum(dim=1)
        if self.normalize:
            sum_embeddings = sum_embeddings / embeddings.shape[1]
        x6 = self.theta6(sum_embeddings)
        
        # repeat graph embedding for all vertices
        # x6.shape = (batch_size, embed_dim)
        # embeddings.shape[1] = n_vertices
        # x6_repeated.shape = (batch_size, n_vertices, embed_dim)
        x6_repeated = x6.unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        
        # embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x7.shape = (batch_size, n_vertices, embed_dim)
        x7 = self.theta7(embeddings)
        
        # x6.shape = x7.shape = (batch_size, n_vertices, embed_dim)
        # features.shape = (batch_size, n_vertices, 2*embed_dim)
        # x5.shape = (batch_size, n_vertices, 1)
        # features = F.relu(torch.cat([x6_repeated, x7], dim=-1))
        features = nn.LeakyReLU()(torch.cat([x6_repeated, x7], dim=-1))
        x5 = self.theta5(features)
        
        # out.shape = (batch_size, n_vertices)
        out = x5.squeeze(-1)

        # print(f"{out.shape=}")
        
        return out        




class OldEmbeddingLayer(nn.Module):
    """
    Calculate embeddings for all vertices
    """
    def __init__(self, embed_dim, bias=False, normalize=False):
        super().__init__()
        # self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)
        self.theta2 = nn.Linear(embed_dim, embed_dim, bias=bias, )
        # self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias) if n_edge_features > 0 else None
        self.normalize = normalize
        
    def forward(self, prev_embeddings, adj, node_features_embeddings, edge_features_embeddings):
        # adj.shape = (batch_size, n_vertices, n_vertices)
        # prev_embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x2.shape = (batch_size, n_vertices, embed_dim)
        x2 = self.theta2(torch.matmul(adj, prev_embeddings))

        x1 = node_features_embeddings
        x3 = edge_features_embeddings

        if x3 is not None:
            ret = nn.LeakyReLU()(x1 + x2 + x3)
        else:
            ret = nn.LeakyReLU()(x1 + x2)

        return ret

        # node_features.shape = (batch_size, n_vertices, n_node_features)
        # x1.shape = (batch_size, n_vertices, embed_dim)
        # x1 = self.theta1(node_features)

        # n_edge_features = edge_features.shape[2]
        # if n_edge_features > 0:
        #     # edge_features.shape = (batch_size, n_vertices, n_vertices, n_edge_features)
        #     # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        #     if edge_features.dim() == 3:
        #         edge_features = edge_features.unsqueeze(-1)
        #     # x4 = F.relu(self.theta4(edge_features))
        #     x4 = nn.LeakyReLU()(self.theta4(edge_features))
        #
        #     # adj.shape = (batch_size, n_vertices, n_vertices)
        #     # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        #     # sum_neighbor_edge_embeddings.shape = (batch_size, n_vertices, embed_dim)
        #     # x3.shape = (batch_size, n_vertices, embed_dim)
        #     sum_neighbor_edge_embeddings = (adj.unsqueeze(-1) * x4).sum(dim=2)
        #     if self.normalize:
        #         norm = adj.sum(dim=2).unsqueeze(-1)
        #         norm[norm == 0] = 1
        #         sum_neighbor_edge_embeddings = sum_neighbor_edge_embeddings / norm
        #
        #     x3 = self.theta3(sum_neighbor_edge_embeddings)
        #
        #     ret = nn.LeakyReLU()(x1 + x2 + x3)
        # else:
        #     ret = nn.LeakyReLU()(x1 + x2)

        # ret.shape = (batch_size, n_vertices, embed_dim)
        # ret = F.relu(x1 + x2 [+ x3])
        # return ret

class OldEdgeFeaturesEmbeddingLayer(nn.Module):
    """
    Calculate the theta3/theta4 component
    """
    def __init__(self, embed_dim, n_edge_features, bias=False, normalize=False):
        super().__init__()
        self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias)
        self.normalize = normalize

    def forward(self, edge_features, adj):
        # edge_features.shape = (batch_size, n_vertices, n_vertices, n_edge_features)
        # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        if edge_features.dim() == 3:
            logging.warning("Wrong number of dimensions")

        # x4 = F.relu(self.theta4(edge_features))
        x4 = nn.LeakyReLU()(self.theta4(edge_features))

        # adj.shape = (batch_size, n_vertices, n_vertices)
        # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        # sum_neighbor_edge_embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x3.shape = (batch_size, n_vertices, embed_dim)
        sum_neighbor_edge_embeddings = (adj.unsqueeze(-1) * x4).sum(dim=2)

        if self.normalize:
            norm = adj.sum(dim=2).unsqueeze(-1)
            norm[norm == 0] = 1
            sum_neighbor_edge_embeddings = sum_neighbor_edge_embeddings / norm

        ret = self.theta3(sum_neighbor_edge_embeddings)

        return ret

