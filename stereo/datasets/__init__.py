import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset_template import custom_collate
from .rigidendo_dataset import RigidEndoDataset
from .scared_dataset import SCAREDDataset
from .serv_dataset import SERVDataset

DATASETS = {
    'RigidEndoDataset': RigidEndoDataset,
    'SCAREDDataset': SCAREDDataset,
    'SERVDataset': SERVDataset,
}


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(data_cfg, batch_size, is_dist, workers, pin_memory, mode,
                     prefetch_factor=1):
    dataset = torch.utils.data.ConcatDataset([
        DATASETS[info.DATASET](info, data_cfg, mode)
        for info in data_cfg.DATA_INFOS
    ])
    shuffle = mode == 'training'
    seed = data_cfg.get('SEED', 42)
    rank = torch.distributed.get_rank() if is_dist else 0
    generator = torch.Generator().manual_seed(seed + rank)
    if is_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    else:
        sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    collate = partial(
        custom_collate, concat_dataset=dataset,
        batch_uniform=shuffle and data_cfg.DATA_TRANSFORM.get('BATCH_UNIFORM', False),
        random_type=data_cfg.DATA_TRANSFORM.get('RANDOM_TYPE'),
        h_range=data_cfg.DATA_TRANSFORM.get('H_RANGE'),
        w_range=data_cfg.DATA_TRANSFORM.get('W_RANGE'),
    )
    kwargs = dict(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
        collate_fn=collate, pin_memory=pin_memory, drop_last=False,
        worker_init_fn=seed_worker, generator=generator,
    )
    if workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
    return dataset, DataLoader(**kwargs), sampler
