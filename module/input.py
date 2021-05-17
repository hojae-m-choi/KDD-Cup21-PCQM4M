from dgl.ops.segment import segment_reduce
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder

import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.atom_encoder = AtomEncoder(
            emb_dim=emb_dim,
        )
        self.bond_encoder = BondEncoder(
            emb_dim=emb_dim,
        )
        self.out_features = emb_dim

    def forward(self, graph, atom_input, bond_input):
        batch_num_edges = graph.batch_num_edges() // 2
        batch_size = graph.batch_size
        max_num_edges = int(torch.max(batch_num_edges).item())
        shape = (batch_size, max_num_edges, self.out_features)

        segments = torch.zeros((batch_size, max_num_edges))
        for i, n in enumerate(batch_num_edges):
            segments[i, :int(n)] = 1
        mask = segments.bool()
        mask = mask.cuda()
        segments = segments.reshape(-1).int()
        segments = segments.cuda()

        atom_h = self.atom_encoder(atom_input)

        # pool bond-connected atoms
        u, v = graph.edges()
        u = u[::2]
        v = v[::2]
        atom_h = atom_h[u] + atom_h[v]  # TODO; better pooling?

        bond_input = bond_input[::2]
        bond_h = self.bond_encoder(bond_input)

        # pool atom-bond
        h = atom_h + bond_h     # TODO; better pooling?

        # make input block with zero padding
        h = segment_reduce(segments, h, 'sum')
        h = h.reshape(*shape)

        return h, mask
