import os
import sys
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', '-r', type=str,
    help='path to the dataset root directory')
parser.add_argument('--path', '-p', type=str,
    help='path to the dataset in root directory')
parser.add_argument('--step', '-s', type=int, default=10,
    help='sampling step, default 10')
parser.add_argument('--name', '-n', type=str,
    help='generated dataset name')
parser.add_argument('--save-path', '-sp', type=str,
    help='path to the generated dataset')
args = parser.parse_args()

ims = list(sorted(glob.glob(os.path.join(args.root, args.path, 'images', '*.jpg'))))

with open(os.path.join(args.save_path, args.name + '.train'), 'a') as file:
    for i in range(0, len(ims), args.step):
        name = os.path.basename(ims[i])
        name = os.path.join(args.path, 'images', name)
        file.write("{}\n".format(name))
    file.close()
        