import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', '-r', type=str,
    help='path to the dataset root directory')
parser.add_argument('-path', type=str,
    help='path to the dataset in root directory')
parser.add_argument('--train-ratio', '-tr', dest='tr', type=float, default=0.8,
    help='the ratio of training samples')
parser.add_argument('--name', type=str,
    help='dataset name')
parser.add_argument('--save-path', type=str,
    help='path to the generated dataset')
args = parser.parse_args()

ims = list(sorted(glob.glob(os.path.join(args.root, args.path, 'images', '*.jpg'))))

num_im = len(ims)
rand_index = np.random.permutation(num_im)
num_train = int(args.tr * num_im)
num_notrain = num_im - num_train

with open(os.path.join(args.save_path, args.name + '.train'), 'a') as file:
    for i in range(num_train):
        name = os.path.basename(ims[rand_index[i]])
        name = os.path.join(args.path, 'images', name)
        file.write("{}\n".format(name))
    file.close()

if num_notrain < 1:
    sys.exit()

with open(os.path.join(args.save_path, args.name + '.notrain'), 'a') as file:
    for i in range(num_train, num_im):
        name = os.path.basename(ims[rand_index[i]])
        name = os.path.join(args.path, 'images', name)
        file.write("{}\n".format(name))
    file.close()