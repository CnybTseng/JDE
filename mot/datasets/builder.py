import torch
import numpy as np
from torch.utils.data import DataLoader
from mot.utils import Registry, build_from_config

DATASETS = Registry('datasets')
COLLATES = Registry('collates')

def build_dataset(config):
    return build_from_config(DATASETS, config)

def build_collate(config):
    return build_from_config(COLLATES, config)

def build_dataloader(dataset, config):
    collate_class = build_collate(config.TRANSFORM.COLLATE)
    args, kwargs = [], {}
    # if collate_class.multiscale:
    init_size = np.array([config.MODEL.ARGS.INPUT.HEIGHT,
        config.MODEL.ARGS.INPUT.WIDTH], dtype=np.int32)
    shared_size = torch.from_numpy(init_size).share_memory_()
    kwargs['shared_size'] = shared_size
    collate_fn = collate_class(*args, **kwargs)
    dataloader = DataLoader(dataset,
        batch_size=config.SOLVER.BATCH_SIZE,
        shuffle=True, num_workers=config.SYSTEM.NUM_WORKERS,
        collate_fn=collate_fn, pin_memory=True, drop_last=True)
    return dataloader