import multiprocessing
import os.path as osp
import shutil

import dgl
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.lsc import PygPCQM4MDataset
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
from pqdm.processes import pqdm
from tqdm import tqdm


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
    graph['node_pos'] = node_positions[:, :3]
    graph['num_nodes'] = len(x)

    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    assert (len(graph['node_pos']) == graph['num_nodes'])

    return graph, gap


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
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph, gap = self.smiles2graph(smiles, homolumogap)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x_pos = torch.from_numpy(graph['node_pos']).to(torch.float32)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
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
    dataset = DglPCQM4MDatasetWithPosition(root)
