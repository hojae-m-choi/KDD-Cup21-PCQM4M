from ogb.lsc import DglPCQM4MDataset
from data.dataset import (
    DglPCQM4MDatasetForDebug,
    DglPCQM4MDatasetWithPosition,
    DglPCQM4MDatasetWithPositionForDebug
)


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
