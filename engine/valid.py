import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.transforms import LineGraph
from tqdm import tqdm

from utils.metrics import AverageMeter


def validate(
        epoch,
        model,
        loader,
):
    model.cuda()
    model.eval()
    last_batch = len(loader) - 1

    loss_m = AverageMeter()
    pbar = tqdm(loader, ascii=True)
    for i, batch in enumerate(pbar):
        num_edges = [x.edge_index.shape[-1] for x in batch]
        new_batch = []
        for j, x in enumerate(batch):
            if x.__num_nodes__ == 1:
                continue
            if x.edge_index.shape[-1] != x.edge_attr.shape[0]:
                continue
            x = LineGraph()(x)
            if len(x.x) != num_edges[j]:
                continue
            new_batch.append(x)

        batch = Batch.from_data_list(new_batch)
        batch = batch.to('cuda')
        with torch.no_grad():
            # Forward
            outputs = model(batch)
            # Compute loss
            loss = F.l1_loss(outputs.squeeze(), batch.y.cuda())

        loss_m.update(loss.item(), batch.num_graphs)

        if i % 10 == 0 or i == last_batch:
            pbar.set_description(f"Epoch {epoch:02d}, Loss {loss_m.avg:.5f}")

    return {
        'loss': loss_m.avg
    }

def validate_pyg(
        epoch,
        model,
        loader,
):
    model.cuda()
    model.eval()
    last_batch = len(loader) - 1

    loss_m = AverageMeter()
    pbar = tqdm(loader, ascii=True)
    for i, data in enumerate(pbar):
        data = data.cuda()
        labels = data.y.reshape((-1, 1))

        with torch.no_grad():
            # Forward
            outputs = model(data)
            # Compute loss
            loss = F.l1_loss(outputs, labels.cuda())

        loss_m.update(loss.item(), data.batch.shape[0])

        if i % 10 == 0 or i == last_batch:
            pbar.set_description(f"Epoch {epoch:02d}, Loss {loss_m.avg:.5f}")

    return {
        'loss': loss_m.avg
    }
