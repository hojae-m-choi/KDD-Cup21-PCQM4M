import argparse
import os
import random
import logging
from collections import OrderedDict
from datetime import datetime
import time

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import DataLoader
from dgl.dataloading import GraphDataLoader, AsyncTransferer
from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator

from data.factory import create_dataset_pyg
from data.dataset import _smiles2graph
from model.model import Perceiver, GNN


_logger = logging.getLogger(__name__)


def resume_checkpoint(model, checkpoint_path, optimizer=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # for faster training, but not deterministic


def test(model, loader, transferer):
    model.eval()
    y_pred = []

    pbar = tqdm(loader, ascii=True)
    for step, graph in enumerate(pbar):
        atom_input = graph.ndata['feat']
        atom_input_gpu = transferer.async_copy(atom_input, torch.device('cuda:0'))
        bond_input = graph.edata['feat']
        bond_input_gpu = transferer.async_copy(bond_input, torch.device('cuda:0'))

        with torch.no_grad():
            pred = model(graph, atom_input_gpu.wait(), bond_input_gpu.wait())

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred

def test_pyg(model, loader):
    model.eval()
    y_pred = []

    pbar = tqdm(loader, ascii=True)
    for step, data in enumerate(pbar):
        data = data.cuda()
        
        with torch.no_grad():
            # Forward
            pred = model(data)
            
        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


class OnTheFlyPCQMDataset(object):
    def __init__(self, smiles_list, smiles2graph=_smiles2graph):
        super(OnTheFlyPCQMDataset, self).__init__()
        self.smiles_list = smiles_list
        self.smiles2graph = smiles2graph

    def __getitem__(self, idx):
        '''Get datapoint with index'''
        smiles, _ = self.smiles_list[idx]
        graph, _ = self.smiles2graph(smiles, None)

        return graph

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.smiles_list)


def main(args):
    shared_params = {
        'num_layers': args.depth,
        'emb_dim': args.emb_dim,
        'graph_pooling': args.graph_pooling,
        'drop_ratio': args.drop_ratio
    }
    
    model = GNN(gnn_type = args.gnn_type, virtual_node = True, **shared_params)
    
    model.cuda()

    seed_everything(args.seed)

    start_epoch = resume_checkpoint(
        model,
        args.checkpoint,
        optimizer=None,
    )

    # dataset
    print(f"Start loading dataset...")
    start = time.time()
    train_dataset, valid_dataset, test_dataset = create_dataset_pyg(args)
    print(f"Dataset is loaded, took {time.time()-start:.2f}s")
    print(len(test_dataset))
    
    if args.platform == 'pyg':
        loader = DataLoader(test_dataset, batch_size=args.batch_size)
        print('Predicting on test data...')
        y_pred = test_pyg(model, loader)
    else:
        loader = GraphDataLoader(test_dataset, batch_size=args.batch_size)
        transferer = AsyncTransferer(torch.device('cuda:0'))
        print('Predicting on test data...')
        y_pred = test(model, loader, transferer)
    
    evaluator = PCQM4MEvaluator()

    
    print('Saving test submission file...')
    print(y_pred.shape)
    evaluator.save_test_submission({'y_pred': y_pred.squeeze()}, args.save_test_dir)


if __name__ == "__main__":
    # TODO; make clean
    default_data_folder = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    default_data_folder = os.path.abspath(default_data_folder)
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('save_test_dir', type=str)
    parser.add_argument('--data', type=str, default=default_data_folder)
    parser.add_argument('-b', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--add-position', action='store_true', default=False)

    # model
    parser.add_argument('--gnn-type', type=str, default='gin')
    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--self-per-cross', type=int, default=1)
    parser.add_argument('--num-latents', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--drop-ratio', type=float, default=0.2)

    # misc
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--platform', type=str, default='pyg')

    args = parser.parse_args()
    main(args)
