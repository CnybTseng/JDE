import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--video1', '-v1', type=str,
    help='the first video file')
parser.add_argument('--video2', '-v2', type=str,
    help='the second video file')
parser.add_argument('--cache-dir', '-cd', type=str,
    help='cache directory')
args = parser.parse_args()

cap1 = cv2.VideoCapture(args.video1)
cap2 = cv2.VideoCapture(args.video2)
i = 0

while True:
    ret1, im1 = cap1.read()
    if not ret1:
        break
    ret2, im2 = cap2.read()
    if not ret2:
        break
    if im1.shape != im2.shape:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
    im = np.concatenate([im1, im2], axis=1)
    cv2.imwrite(os.path.join(args.cache_dir, '%06d.jpg' % i), im)
    i += 1

os.system('ffmpeg -i {} {}.mp4 -y'.format(os.path.join(args.cache_dir, '%06d.jpg'),
    os.path.join(args.cache_dir, 'cat')))
    