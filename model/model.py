import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from module.input import SetEncoder
from module.perceiver import Perceiver as _Perceiver

from module.node import GNN_node_Virtualnode


class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Regressor, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim=in_dim)  # Pytorch Module class w/ learnable parameters
        self.bond_encoder = BondEncoder(emb_dim=in_dim)  # Pytorch Module class w/ learnable parameters
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.regress = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = self.atom_encoder(h)
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.regress(hg)


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, n_hidden=2, **kwargs):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(in_feats, h_feats, num_heads=kwargs['num_heads'], feat_drop=kwargs['feat_drop']))
        for _ in range(n_hidden):
            self.layers.append(dglnn.GATConv(h_feats * kwargs['num_heads'], h_feats, num_heads=kwargs['num_heads'],
                                             feat_drop=kwargs['feat_drop']))
        self.layers.append(dglnn.GraphConv(h_feats * kwargs['num_heads'], num_classes))

    def forward(self, g, in_feat):
        for i, layer in enumerate(self.layers):
            if i == 0:
                h = layer(g, in_feat)
                h = F.relu(h)
            elif i == len(self.layers) - 1:
                h = layer(g, h)
                return h
            else:
                h = layer(g, h)
                h = F.relu(h)


class Perceiver(nn.Module):
    def __init__(
            self, depth, emb_dim,
            self_per_cross, num_latents, latent_dim,
            attn_dropout, ff_dropout,
    ):
        super().__init__()
        self.encoder = SetEncoder(
            emb_dim=emb_dim
        )
        self.perceiver = _Perceiver(
            depth=depth,
            input_channels=emb_dim,
            self_per_cross_attn=self_per_cross,
            num_latents=num_latents,
            latent_dim=latent_dim,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    def forward(self, graph, atom_input, bond_input):
        x, mask = self.encoder(graph, atom_input, bond_input)
        x = self.perceiver(x, mask)
        return x.squeeze()


class GNN(torch.nn.Module):
    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300,
                 gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0, JK="last", graph_pooling="sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                                 gnn_type=gnn_type)
        else:
            # self.gnn_node = GNN_node(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
            #                          gnn_type=gnn_type)
            raise NotImplementedError

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)
