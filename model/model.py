import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from module.input import SetEncoder
from module.perceiver import Perceiver as _Perceiver


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
