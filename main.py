import argparse
import os
import random
import logging
from collections import OrderedDict
from datetime import datetime
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from dgl.dataloading import GraphDataLoader, AsyncTransferer
from torch_geometric.data import DataListLoader

from data.factory import create_dataset, create_dataset_pyg
from engine.train import train_one_epoch
from engine.valid import validate
from model.model import Perceiver, GNN
from utils.checkpoint_saver import CheckpointSaver
from utils.summary import update_summary


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


def main(args):
    # model = Perceiver(
    #     depth=args.depth,
    #     emb_dim=args.emb_dim,
    #     self_per_cross=args.self_per_cross,
    #     num_latents=args.num_latents,
    #     latent_dim=args.latent_dim,
    #     attn_dropout=args.attn_dropout,
    #     ff_dropout=args.ff_dropout,
    # )
    model = GNN(graph_pooling='attention')
    model.cuda()

    seed_everything(args.seed)

    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=args.decay)  # TODO; LR
    if args.sched:
        if args.sched == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
        elif args.sched == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        start_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
        )
        scheduler.__dict__.update({'last_epoch': start_epoch-1})
        print(f"Restore scheduler last epoch: {scheduler.last_epoch}")

    # dataset
    print(f"Start loading dataset...")
    start = time.time()
    train_dataset, valid_dataset, test_dataset = create_dataset(args)
    print(f"Dataset is loaded, took {time.time()-start:.2f}s")

    train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataListLoader(valid_dataset, batch_size=args.batch_size)

    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        # args.model,
    ])
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        args=args,
        checkpoint_dir=output_dir,
        recovery_dir=output_dir,
        decreasing=True,
    )

    best_metric = None
    best_epoch = None
    for epoch in range(start_epoch, args.epochs):
        train_metric = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
        )

        eval_metric = validate(
            epoch=epoch,
            model=model,
            loader=valid_loader,
        )

        update_summary(
            epoch, train_metric, eval_metric, os.path.join(output_dir, 'summary.csv'),
            write_header=best_metric is None
        )
        # save proper checkpoint with eval metric
        save_metric = eval_metric['loss']
        best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

        # step
        scheduler.step()


if __name__ == "__main__":
    # TODO; make clean
    default_data_folder = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    default_data_folder = os.path.abspath(default_data_folder)
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--data', type=str, default=default_data_folder)
    parser.add_argument('-b', dest='batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--no-resume-opt', action='store_true', default=False)
    parser.add_argument('--add-position', action='store_true', default=False)
    parser.add_argument('--toLinegraph', action='store_true', default=False)
    parser.add_argument('--sched', type=str, default='step')
    parser.add_argument('--decay', type=float, default=0.0)

    # model
    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--self-per-cross', type=int, default=1)
    parser.add_argument('--num-latents', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--attn-dropout', type=float, default=0.2)
    parser.add_argument('--ff-dropout', type=float, default=0.2)

    # misc
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--platform', type=str, default='pyg')

    args = parser.parse_args()
    main(args)
