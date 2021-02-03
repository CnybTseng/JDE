import os
import cv2
import sys
import argparse
import numpy as np

sys.path.append('.')
from xxx import LoadImagesAndLabels 

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,
    help='path to the dataset root directory')
parser.add_argument('--path', type=str,
    help='path to the dataset')
args = parser.parse_args()

im_paths = open(args.path, 'r').readlines()
im_paths = [path.strip() for path in im_paths]
im_paths = list(filter(lambda x: len(x) > 0, im_paths))
im_paths = [os.path.join(args.root, path) for path in im_paths]

loader = LoadImagesAndLabels(augment=False, transforms=None)
writer = None
for n, im_path in enumerate(im_paths):
    lb_path = im_path.replace('images', 'labels_with_ids')
    lb_path = lb_path.replace('.png', '.txt')
    lb_path = lb_path.replace('.jpg', '.txt')
    image, labels, _, _ = loader.get_data(im_path, lb_path)
    image = image.numpy()
    image = np.ascontiguousarray(image.transpose(1, 2, 0)).astype(np.uint8)
    for c, i, x, y, w, h in labels:
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        np.random.seed(int(i))
        color = np.random.randint(0, 256, size=(3,)).tolist()
        rect = np.array([x - w / 2, y - h / 2, w, h])
        image = cv2.rectangle(image, rect, color)
    if writer is None:
        size = (image.shape[1], image.shape[0])
        writer = cv2.VideoWriter('check.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, size)
    writer.write(image)
    print('\r{}/{}: {}'.format(n, len(im_paths), im_path), end='', flush=True)