import os
import sys
import torch
import argparse
from torch import nn
sys.path.append(os.getcwd())
from mot.utils import config
from mot.models import build_tracker

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training multiple object tracker.')
    parser.add_argument('--config', type=str, default='',
        help='training configuration file path')
    parser.add_argument('--save-name', '-sn', type=str,
        default='./model.arch',
        help='saving name for generated architecture configuration')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
        help='modify configuration in command line')
    return parser.parse_args()

def main():
    args = parse_args()
    if os.path.isfile(args.config):
        config.merge_from_file(args.config)
    config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    
    model = build_tracker(config.MODEL)
    classes = (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ReLU,
        nn.MaxPool2d)
    with open(args.save_name, 'w') as fd:
        for name, module in model.named_modules():
            if isinstance(module, classes):
                fd.write('{}\n'.format(name))
    
if __name__ == '__main__':
    main()