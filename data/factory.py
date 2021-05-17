from ogb.lsc import DglPCQM4MDataset, PygPCQM4MDataset
from data.dataset import (
    DglPCQM4MDatasetForDebug,
    DglPCQM4MDatasetWithPosition,
    DglPCQM4MDatasetWithPositionForDebug
)

from data.dataset import (
    PygPCQM4MDatasetForDebug,
    PygPCQM4MDatasetWithPosition,
    PygPCQM4MDatasetWithPositionForDebug
)

def create_dataset_pyg(args):
    if args.debug:
        if args.add_position:
            dataset_cls = PygPCQM4MDatasetWithPositionForDebug
        else:
            dataset_cls = PygPCQM4MDatasetForDebug
    else:
        if args.add_position:
            dataset_cls = PygPCQM4MDatasetWithPosition
        else:
            dataset_cls = PygPCQM4MDataset

    dataset = dataset_cls(root=args.data)
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]
    return train_dataset, valid_dataset, test_dataset



def create_dataset(args):
    if args.debug:
        if args.add_position:
            dataset_cls = DglPCQM4MDatasetWithPositionForDebug
        else:
            dataset_cls = DglPCQM4MDatasetForDebug
    else:
        if args.add_position:
            dataset_cls = DglPCQM4MDatasetWithPosition
        else:
            dataset_cls = DglPCQM4MDataset

    dataset = dataset_cls(root=args.data)
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]
    return train_dataset, valid_dataset, test_dataset
