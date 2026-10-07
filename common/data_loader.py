import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .utils import StandardScaler

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _vrange(starts, stops):
    stops = np.asarray(stops)
    l = stops - starts
    assert l.min() == l.max(), "ranges must be equal length"
    return (np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())).reshape(-1, l[0])


def load_adj(dataset):
    return np.load(os.path.join(DATA_ROOT, dataset, "adj.npy")).astype(np.float32)


def get_dataloaders(dataset, batch_size=64, tod=False, dow=False,
                    in_steps=12, out_steps=12, num_workers=0, logger=None):
    data_dir = os.path.join(DATA_ROOT, dataset)
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))
    splits = {}
    for split in ["train", "val", "test"]:
        idx = index[split]
        x_idx = _vrange(idx[:, 0], idx[:, 1])
        y_idx = _vrange(idx[:, 1], idx[:, 2])
        x = data[x_idx]
        y = data[y_idx][..., :1]
        splits[split] = [x, y]

    scaler = StandardScaler(mean=splits["train"][0][..., 0].mean(),
                            std=splits["train"][0][..., 0].std())
    for split in splits:
        splits[split][0][..., 0] = scaler.transform(splits[split][0][..., 0])

    loaders = {}
    for split in ["train", "val", "test"]:
        x, y = splits[split]
        ds = TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=num_workers, drop_last=False)
        if logger:
            logger.info(f"{split}: x{tuple(x.shape)} y{tuple(y.shape)}")

    return loaders["train"], loaders["val"], loaders["test"], scaler
