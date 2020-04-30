import os
import sys
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to training samples')
parser.add_argument('--train-ratio', '-tr', dest='tr', default=1, type=float, help='the ratio of training samples')
args = parser.parse_args()

image_filenames = []
for p in args.path.split(','):
    image_filenames += list(sorted(glob.glob(os.path.join(p, '*.jpg'))))
    image_filenames += list(sorted(glob.glob(os.path.join(p, '*.JPG'))))
    image_filenames += list(sorted(glob.glob(os.path.join(p, '*.jpeg'))))
    image_filenames += list(sorted(glob.glob(os.path.join(p, '*.png'))))

num_samples = len(image_filenames)
rand_index = np.random.permutation(num_samples)
num_train = int(args.tr * num_samples)
num_test = num_samples - num_train

with open('train.txt', 'w') as file:
    for i in range(num_train):
        root, ext = os.path.splitext(image_filenames[rand_index[i]])
        root = root.replace('img1', 'labels_with_ids')
        file.write(f"{image_filenames[rand_index[i]]} {root}.txt\n")
    file.close()

if num_test < 1:
    sys.exit()

with open('test.txt', 'w') as file:
    for i in range(num_train, num_samples):
        root, ext = os.path.splitext(image_filenames[rand_index[i]])
        root = root.replace('img1', 'labels_with_ids')
        file.write(f"{image_filenames[rand_index[i]]} {root}.txt\n")
    file.close()