from typing import List, Optional, Callable, Union, Any, Tuple
from collections.abc import Sequence
import multiprocessing
import os.path as osp
import shutil
from itertools import repeat, product
import copy

import dgl
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.lsc import PygPCQM4MDataset as PygPCQM4MDataset
from ogb.lsc import DglPCQM4MDataset as _PCQM4MDataset
from ogb.utils.features import (
    atom_to_feature_vector,
    bond_to_feature_vector,
)

from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from pqdm.processes import pqdm
from tqdm import tqdm

from utils.transforms import LineGraph

import pickle

from torch_sparse import coalesce, transpose
IndexType = Union[slice, Tensor, np.ndarray, Sequence]


def _smiles2graph(smiles_string, gap):
    """
    This function returns same graph dict, but
    added xyz position of atoms calculated from rdkit.
    :param smiles_string:
    :return:
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atom positions
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    mol = Chem.RemoveHs(mol)
    conformer = mol.GetConformers()[0]
    node_positions = np.array(conformer.GetPositions())

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['node_pos'] = node_positions[:, :2]
    graph['num_nodes'] = len(x)

    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    assert (len(graph['node_pos']) == graph['num_nodes'])

    dgl_graph = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])
    dgl_graph.edata['feat'] = torch.from_numpy(graph['edge_feat']).to(torch.int64)
    dgl_graph.ndata['feat'] = torch.from_numpy(graph['node_feat']).to(torch.int64)
    dgl_graph.ndata['pos'] = torch.from_numpy(graph['node_pos']).float()

    return dgl_graph, gap

def smiles2graphWith2Dposition(smiles_string, gap):
    """
    This function returns same graph dict, but
    added xyz position of atoms calculated from rdkit.
    :param smiles_string:
    :return:
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atom positions
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    mol = Chem.RemoveHs(mol)
    conformer = mol.GetConformers()[0]
    node_positions = np.array(conformer.GetPositions())

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['node_pos'] = node_positions[:, :2]
    graph['num_nodes'] = len(x)

    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    assert (len(graph['node_pos']) == graph['num_nodes'])

    return graph, gap

def smiles2graphWith3Dposition(smiles_string):
    """
    This function returns same graph dict, but
    added xyz position of atoms calculated from rdkit.
    :param smiles_string:
    :return:
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atom positions
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    mol = Chem.RemoveHs(mol)
    conformer = mol.GetConformers()[0]
    node_positions = np.array(conformer.GetPositions())

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['node_pos'] = node_positions[:, :2]
    graph['num_nodes'] = len(x)

    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    assert (len(graph['node_pos']) == graph['num_nodes'])

    return graph


class DglPCQM4MDatasetForDebug(_PCQM4MDataset):
    """
    Added node 3D positions and replace tqdm to pqdm.
    """
    def __init__(self, root):
        super().__init__(root)

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):
            # if pre-processed file already exists
            self.graphs, label_dict = load_graphs(pre_processed_file_path, list(range(102400)))
            self.labels = label_dict['labels']

    def get_idx_split(self):
        cur = 102400 - 2048
        split_dict = {
            'train': torch.from_numpy(np.array(list(range(cur)))),
            'valid': torch.from_numpy(np.array(list(range(cur, cur+1024)))),
            'test': torch.from_numpy(np.array(list(range(cur+1024, cur+2048))))
        }
        return split_dict


class DglPCQM4MDatasetWithPosition(_PCQM4MDataset):
    def __init__(self, root, smiles2graph=_smiles2graph):
        super().__init__(root, smiles2graph)

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed_with_position')

        if osp.exists(pre_processed_file_path):
            # if pre-processed file already exists
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict['labels']
        else:
            # if pre-processed file does not exist
            if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
                # if the raw file does not exist, then download it.
                self.download()

            data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
            smiles_list = data_df['smiles']
            homolumogap_list = data_df['homolumogap']

            print('Converting SMILES strings into graphs...')
            self.graphs = []
            self.labels = []
            for smiles_string, gap in tqdm(zip(smiles_list, homolumogap_list), total=len(homolumogap_list)):
                dgl_graph, gap = self.smiles2graph(smiles_string, gap)

                self.graphs.append(dgl_graph)
                self.labels.append(gap)

            self.labels = torch.tensor(self.labels, dtype=torch.float32)

            # double-check prediction target
            split_dict = self.get_idx_split()
            assert (all([not torch.isnan(self.labels[i]) for i in split_dict['train']]))
            assert (all([not torch.isnan(self.labels[i]) for i in split_dict['valid']]))
            assert (all([torch.isnan(self.labels[i]) for i in split_dict['test']]))

            print('Saving...')
            save_graphs(pre_processed_file_path, self.graphs, labels={'labels': self.labels})


class DglPCQM4MDatasetWithPositionForDebug(DglPCQM4MDatasetWithPosition):
    def __init__(self, root, smiles2graph=_smiles2graph):
        super().__init__(root, smiles2graph)

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed_with_position')

        if osp.exists(pre_processed_file_path):
            # if pre-processed file already exists
            self.graphs, label_dict = load_graphs(pre_processed_file_path, list(range(102400)))
            self.labels = label_dict['labels']

    def get_idx_split(self):
        cur = 102400 - 2048
        split_dict = {
            'train': torch.from_numpy(np.array(list(range(cur)))),
            'valid': torch.from_numpy(np.array(list(range(cur, cur + 1024)))),
            'test': torch.from_numpy(np.array(list(range(cur + 1024, cur + 2048))))
        }
        return split_dict

    
class PygPCQM4MDatasetForDebug(PygPCQM4MDataset):
    """
    Added node 3D positions and replace tqdm to pqdm.
    """
    def __init__(self, root):
        super().__init__(root)
    
    def get_idx_split(self):
        cur = 102400 - 2048
        split_dict = {
            'train': torch.from_numpy(np.array(list(range(cur)))),
            'valid': torch.from_numpy(np.array(list(range(cur, cur+1024)))),
            'test': torch.from_numpy(np.array(list(range(cur+1024, cur+2048))))
        }
        return split_dict

    
class PygPCQM4MDatasetWithPosition(PygPCQM4MDataset):
    def __init__(self, root, smiles2graph = smiles2graphWith2Dposition):
        super().__init__(root, smiles2graph)
        
    @property
    def processed_file_names(self):
        return 'geometric_data_processed_with_position.pt'
    
    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list)), ascii=True):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph, gap = self.smiles2graph(smiles, homolumogap)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            edge_attrs = []
            if len(data.edge_index[0]) > 0:
                edge_attrs.append(torch.from_numpy(graph['edge_feat']).to(torch.int64))
                edge_attrs.append(torch.from_numpy(graph['node_feat'][data.edge_index[0]]))
                edge_attrs.append(torch.from_numpy(graph['node_feat'][data.edge_index[1]]))
            else:
                edge_attrs.append(torch.zeros((1, 3)))
                edge_attrs.append(torch.zeros((1, 9)))
                edge_attrs.append(torch.zeros((1, 9)))
            edge_attr = torch.cat(edge_attrs, -1)
            
            data.edge_attr = edge_attr
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            
            data.x_pos = torch.from_numpy(graph['node_pos']).to(torch.float32)
            x_pos2D = data.x_pos.reshape( (data.x_pos.size(0), 1, 2) )
            src_idx = data.edge_index[0,:].transpose(0,-1)
            dst_idx = data.edge_index[1,:].transpose(0,-1)
            edge_pos2D = torch.cat( [ x_pos2D[src_idx, :, : ], x_pos2D[dst_idx, :, : ] ], dim = 1)
            data.edge_pos = torch.mean(edge_pos2D, dim = -2, keepdim = False )
            
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

        
class PygPCQM4MDatasetWithPositionLineGraph(PygPCQM4MDatasetWithPosition):
    def __init__(self, root, smiles2graph = smiles2graphWith2Dposition):
        super().__init__(root, smiles2graph)
        
        self.lined_data, self.lined_slices = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load( self.processed_paths[1])
        self.indices_map = pickle.load( open(self.processed_paths[2], 'rb') )
        self.offset = sum(self.indices_map['isLinegraph'])
        self._indices = None
        
    @property
    def processed_file_names(self):
        return ('geometric_data_processed_with_position_linegraph.pt',
                'geometric_data_processed_with_position_onenode.pt',
                'geometric_data_processed_with_position_linegraph_mapper.pt',
               )
    
    def len(self):
        return len(self.indices_map['slice_idx'])
    
#     def indices(self) -> Sequence:
#         if self._indices is None:
#             return range(self.len())
#         else:
#             return [self.indices_map['slice_idx'][i] if self.indices_map['isLinegraph'][i] else self.indices_map['slice_idx'][i] + self.offset for i in self._indices]
    
#     def index_select(self, idx: IndexType) -> 'Dataset':
#         indices = self.indices()

#         if isinstance(idx, slice):
#             indices = indices[idx]

#         elif isinstance(idx, Tensor) and idx.dtype == torch.long:
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
#             idx = idx.flatten().nonzero(as_tuple=False)
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
#             idx = idx.flatten().nonzero()[0]
#             return self.index_select(idx.flatten().tolist())

#         elif isinstance(idx, Sequence) and not isinstance(idx, str):
#             indices = [indices[i] for i in idx]

#         else:
#             raise IndexError(
#                 f"Only integers, slices (':'), list, tuples, torch.tensor and "
#                 f"np.ndarray of dtype long or bool are valid indices (got "
#                 f"'{type(idx).__name__}')")

#         dataset = copy.copy(self)
#         dataset._indices = indices
        
#         return dataset
    
    def get(self, idx: int) -> Data:
        
        if hasattr(self, '_data_list'):
            if self._data_list is None:
                self._data_list = self.len() * [None]
            else:
                data = self._data_list[idx]
                if data is not None:
                    return copy.copy(data)
        
        isLinegraph = self.indices_map['isLinegraph'][idx]    
#         if idx < self.offset : 
        if isLinegraph:
            self_data = self.lined_data
            self_slices = self.lined_slices
#             local_idx = idx
        else:
            self_data = self.data
            self_slices = self.slices
#             local_idx = idx - self.offset
        local_idx = self.indices_map['slice_idx'][idx]
    
        data = self_data.__class__()
        if hasattr(self_data, '__num_nodes__'):
            try:
                data.num_nodes = self_data.__num_nodes__[local_idx]
            except:
                print(len(self_data.__num_nodes__), local_idx, idx)
                raise IndexError
                    
        for key in self_data.keys:
            item, slices = self_data[key], self_slices[key]
            start, end = slices[local_idx].item(), slices[local_idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                cat_dim = self_data.__cat_dim__(key, item)
                if cat_dim is None:
                    cat_dim = 0
                s[cat_dim] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        if hasattr(self, '_data_list'):
            self._data_list[idx] = copy.copy(data)

        return data
    
    
    def process(self):
#         self.data, self.slices = torch.load(self.root + '/processed/'+'geometric_data_processed_with_position.pt')
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        linegraph_data_list = []
        graph_data_list = []
        indices_map = {'isLinegraph':[], 'slice_idx': []}
        for i in tqdm(range(len(smiles_list)), ascii=True):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
#             edge_attrs = []
            if len(data.edge_index[0]) > 0:
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
#                 edge_attrs.append(torch.from_numpy(graph['edge_feat']).to(torch.int64))
#                 edge_attrs.append(torch.from_numpy(graph['node_feat'][data.edge_index[0]]))
#                 edge_attrs.append(torch.from_numpy(graph['node_feat'][data.edge_index[1]]))
            else:
                data.edge_attr = torch.zeros((1, 3))
#                 edge_attrs.append(torch.zeros((1, 3)))
#                 edge_attrs.append(torch.zeros((1, 9)))
#                 edge_attrs.append(torch.zeros((1, 9)))
#             edge_attr = torch.cat(edge_attrs, -1)
            
#             data.edge_index, data.edge_attr = coalesce(data.edge_index, edge_attr, data.num_nodes, data.num_nodes)
            
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            
            data.x_pos = torch.from_numpy(graph['node_pos']).to(torch.float32)
            x_pos2D = data.x_pos.reshape( (data.x_pos.size(0), 1, 2) )
            src_idx = data.edge_index[0,:].transpose(0,-1)
            dst_idx = data.edge_index[1,:].transpose(0,-1)
            edge_pos2D = torch.cat( [ x_pos2D[src_idx, :, : ], x_pos2D[dst_idx, :, : ] ], dim = 1)
            data.edge_pos = torch.mean(edge_pos2D, dim = -2, keepdim = False )
            
            data.y = torch.Tensor([homolumogap])
            
            ####
            
            if data.__num_nodes__ == 1:
                print(f"{i}th molecule has Num nodes == 1,")
                print(smiles_list[i])
                print('append original graph')
                data.isLinegraph = False
                indices_map['isLinegraph'].append(False)
                indices_map['slice_idx'].append( len(graph_data_list) )
                graph_data_list.append(data)
                continue

            if data.edge_index.size(1) != data.edge_attr.size(0):
                print(f"{i}th: edge_index.size(1) != x.edge_attr.size(0), pass")
                data.isLinegraph = False
                indices_map['isLinegraph'].append(False)
                indices_map['slice_idx'].append( len(graph_data_list) )
                graph_data_list.append(data)
                continue
            
            data.x = torch.cat([data.x, data.x_pos], dim = 1)
            data.edge_attr = torch.cat([data.edge_attr, data.edge_pos], dim = 1)
#             num_edges = data.edge_index.size(1)
#             if data.is_directed():
#                 edge_index, edge_attr = coalesce(data.edge_index, data.edge_attr, data.num_nodes,
#                                      data.num_nodes)
                
#                 edge_index_t, edge_attr_t = transpose(edge_index, edge_attr, data.num_nodes,
#                                               data.num_nodes, coalesced=True)
#                 index_symmetric = torch.all(edge_index == edge_index_t)
#                 if not index_symmetric:
#                     print(edge_index)
#                     print(edge_index_t)
#                 attr_symmetric = torch.all(data.edge_attr == edge_attr_t)
#                 if not attr_symmetric:
#                     print(edge_attr)
#                     print(edge_attr_t)
                    
#                 print(data.num_nodes)
#                 print(data.edge_attr.shape)
#                 print(data.edge_index)
#                 print(f"{i}th isDirected")
            data = LineGraph()(data)
            
#             if data.x.size(0) != num_edges//2:
#                 print(f"{i}th: data.x.size(0)({data.x.size(0)}) != num_edges//2({num_edges//2})")
#                 print(smiles_list[i])
#                 print("something changed by linegraph transform, pass")
#                 data.isLinegraph = True
#                 indices_map['isLinegraph'].append(True)
#                 indices_map['slice_idx'].append( len(linegraph_data_list) )
#                 linegraph_data_list.append(data)
#                 continue
                
            data.__num_nodes__ = int(data.num_nodes)
            data.x_pos = data.x[:,-2:]
            data.x = data.x[:,:-2].to(torch.int64)
            data.edge_pos = data.edge_attr[:,-2:]
            data.edge_attr = data.edge_attr[:,:-2].to(torch.int64)
            
            data.isLinegraph = True
            indices_map['isLinegraph'].append(True)
            indices_map['slice_idx'].append( len(linegraph_data_list) )
            
            linegraph_data_list.append(data)
            
            
            
        # double-check prediction target
#         split_dict = self.get_idx_split()
#         assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
#         assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
#         assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test']]))

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
        if len(linegraph_data_list) > 0 :
            lined_data, lined_slices = self.collate(linegraph_data_list)
        else:
            lined_slices = {'x':torch.Tensor([])}
        if len(graph_data_list) > 0 :
            data, slices = self.collate(graph_data_list)
        else:
            slices = {'x':torch.Tensor([])}

        print('Saving...')
        torch.save((lined_data, lined_slices), self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[1])
        pickle.dump(indices_map, open(self.processed_paths[2], 'wb'))

        
class PygPCQM4MDatasetWithPositionForDebug(PygPCQM4MDatasetWithPosition):
    def __init__(self, root, smiles2graph=_smiles2graph):
        super().__init__(root, smiles2graph)
    
    def get_idx_split(self):
        cur = 102400 - 2048
        split_dict = {
            'train': torch.from_numpy(np.array(list(range(cur)))),
            'valid': torch.from_numpy(np.array(list(range(cur, cur+1024)))),
            'test': torch.from_numpy(np.array(list(range(cur+1024, cur+2048))))
        }
        return split_dict

    
if __name__ == "__main__":
    import os
    root = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
    root = os.path.abspath(root)
    dataset = PygPCQM4MDatasetWithPosition(root)
