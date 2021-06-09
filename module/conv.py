import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

from module.gvp import GVPConv, _merge
import math

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "add"):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.atom_encoder = AtomEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr, no_emb = False, isLinegraph = True):
        if no_emb:
            edge_embedding = edge_attr
        else:
            if isLinegraph:
                edge_embedding = self.atom_encoder(edge_attr)
            else:    
                edge_embedding = self.bond_encoder(edge_attr)
        
        #print(f'shape of edge_embedding: {edge_embedding.shape}')
        #print(f'shape of x: {x.shape}')
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out
    
    
### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layers (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation

### GVP+GIN convolution along the graph structure    
class GVPGINConv(torch.nn.Module):
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=1, module_list=None, gvp_aggr="mean", gin_aggr = "add" ):
        
        super(GVPGINConv, self).__init__()
        
        ## GVP
        self.gvp_conv = GVPConv(in_dims, 
                                (out_dims[0] - in_dims[1]*3, in_dims[1]), 
                                edge_dims,
                                n_layers, module_list, gvp_aggr)
        
        ## GIN
        self.gin_conv = GINConv(emb_dim = out_dims[0], aggr = gin_aggr)
        

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        
        ## transformation for edge_attr
        edge_embedding = self.gin_conv.bond_encoder(edge_attr)
        
        ## GVP
        GVP_out = self.gvp_conv(x, edge_index, edge_embedding)
        x = _merge(*GVP_out)
        
        ## GIN
        out = super(GVPConv, self).forward(x, edge_index, edge_attr, no_emb = True)

        return out

### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        
        if gnn_type == 'gvp+gin':
#             self.convs.append( GVPConv( (emb_dim, 1), (emb_dim, 1), (emb_dim, 0), n_layers = 1) ) 
            self.convs.append( GVPConv( (emb_dim, 1), (emb_dim, 0), (emb_dim, 1), n_layers = 2) )
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            
            for layer in range(num_layers-1):
                self.convs.append(GINConv(emb_dim))
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            
=======
=======
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
=======
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
                
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
        else:
            for layer in range(num_layers):
                if gnn_type == 'gin':
                    self.convs.append(GINConv(emb_dim))
                elif gnn_type == 'gcn':
                    self.convs.append(GCNConv(emb_dim))
                else:
                    ValueError('Undefined GNN type called {}'.format(gnn_type))

                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

#         h_list = [self.atom_encoder(x)]
        h_list = [self.bond_encoder(x)]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            if 'gvp' in self.gnn_type:
                if layer == 0:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                    x_pos3D = torch.cat ( [batched_data.x_pos, 
                                           torch.zeros( (batched_data.x_pos.shape[0], 1) ).cuda() ] ,
                                         dim =1).reshape( (batched_data.x_pos.shape[0], 1, 3) )
                    edge_pos3D = torch.cat( [batched_data.edge_pos,
                                             torch.zeros((batched_data.edge_pos.shape[0], 1)).cuda() ],
                                           dim = 1).reshape((batched_data.edge_pos.shape[0], 1, 3) )
                    # edge_embedding = self.bond_encoder(edge_attr)
                    edge_embedding = self.atom_encoder(edge_attr)
                    h = self.gvplayer( (h_list[layer], x_pos3D,), edge_index, (edge_embedding, edge_pos3D)  )  
                    h = self.convs[layer]( h, edge_index, (edge_embedding, edge_pos3D)  )
                    h = self.layer_norm(h)
                    h = _merge(h[0], h[1])
=======
=======
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
=======
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
                    x_pos3D = torch.cat ( [batched_data.x_pos, torch.zeros( (batched_data.x_pos.shape[0], 1) ).cuda() ] ,
                                         1).reshape( (batched_data.x_pos.shape[0], 1, 3) )
                    #print( f'\nx_pos3D shape: {x_pos3D.shape}')
                    edge_pos3D = torch.zeros( (batched_data.edge_attr.shape[0], 1, 3) ).cuda()
                    #print( f'edge_attr shape: {edge_attr.shape}')
                    #print( f'edge_pos3D shape: {edge_pos3D.shape}')
                    edge_embedding = self.bond_encoder(edge_attr)
                    h = self.convs[layer]( (h_list[layer], x_pos3D,), edge_index, (edge_embedding, edge_pos3D)  )
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
=======
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
=======
>>>>>>> parent of f6d75b0... mod adding gvpconvlayer(layernorm, dropout), instead of gvpconv
                else:
                    h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]
        
        return node_representation



    
### Virtual GVP+GIN GNN to generate node embedding
class GVPGIN_node_Virtualnode(torch.nn.Module):
    
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin' ):
        '''
            in_dims tuple(int, int): n_scalar_features, n_vector_features of nodes
            edge_dims tuple(int, int): n_scalar_features, n_vector_features of edges
            out_dim (int): node embedding dimensionality + 3*in_dims[1]. == dimensionality of output.
        '''
        
        super(GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gvp+gin':
                self.convs.append( _merge( GVPConv( (emb_dim, 1), (emb_dim,1), emb_dim, n_layers = 1) ) )
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gvp+gcn':
                self.convs.append( _merge( GVPConv( (emb_dim, 1), (emb_dim,1), emb_dim, n_layers = 1) ) )
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

        
    

if __name__ == "__main__":
    pass