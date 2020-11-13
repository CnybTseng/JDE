import os
import argparse
from mot.utils import config
from mot.models import build_tracker

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training multiple object tracker.')
    parser.add_argument('--config', type=str, default='',
        help='training configuration file path')
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
    print(model)
    
if __name__ == '__main__':
    main()