def create_dataset(args):
    if args.debug:
        from data.dataset import DglPCQM4MDataset
    else:
        from ogb.lsc import DglPCQM4MDataset

    dataset = DglPCQM4MDataset(root=args.data)
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]
    return train_dataset, valid_dataset, test_dataset
