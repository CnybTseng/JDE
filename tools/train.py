import os
import sys
import torch
import argparse
sys.path.append(os.getcwd())
from mot.utils import config, mkdirs
from mot.models import build_tracker
from mot.datasets import build_dataset
from mot.apis import train_tracker

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training multiple object tracker.')
    parser.add_argument('--config', type=str, default='',
        help='training configuration file path')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
        help='modify configuration in command line')
    return parser.parse_args()

def main():
    # Parse configurations.
    args = parse_args()
    if os.path.isfile(args.config):
        config.merge_from_file(args.config)
    config.merge_from_list(args.opts)
    
    torch.backends.cudnn.benchmark = True
    mkdirs(config.SYSTEM.TASK_DIR)
    
    # Build dataset
    dataset = build_dataset(config.DATASET)
    
    # Build model.
    num_ide = int(dataset.max_id + 1)
    config.MODEL.ARGS.HEAD.ARGS[1]['num_ide'] = num_ide
    model = build_tracker(config.MODEL)
   
    # Train tracker now.
    train_tracker(model, dataset, config)
    
if __name__ == '__main__':
    main()