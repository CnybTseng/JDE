import os
import sys
import random
import argparse
import os.path as osp
sys.path.append(os.getcwd())
from mot.utils import mkdirs

def parse_args():
    parser= argparse.ArgumentParser(
        description='Generate INT8 quantization samples from given dataset.')
    parser.add_argument('--root', '-r', type=str,
        help='dataset data root directory.')
    parser.add_argument('--data', '-d', type=str,
        help='dataset configuration filepath.')
    parser.add_argument('-k', type=int, default=1000,
        help='The maximum number of generated samples.')
    parser.add_argument('--save-path', '-sp', type=str,
        default='./samples',
        help='Samples saving path.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    all_samples = []
    with open(args.data, 'r') as fd1:
        files = fd1.readlines()
        files = [f.strip() for f in files]
        files = list(filter(lambda f: len(f) > 0, files))
        for file in files:
            with open(file, 'r') as fd2:
                samples = fd2.readlines()
                samples = [s.strip() for s in samples]
                samples = list(filter(lambda s: len(s) > 0, samples))
                samples = [osp.join(args.root, s) for s in samples]
            all_samples += samples
    k = min(len(all_samples), args.k)
    selections = random.sample(all_samples, k)
    mkdirs(args.save_path)
    for s in selections:
        print('copy {} to {}'.format(s, args.save_path))
        os.system('cp {} {}'.format(s, args.save_path))