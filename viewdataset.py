import os
import cv2
import numpy as np


im_label_paths = open('dataset/caltech/train.txt').read().split()
im_paths = im_label_paths[0::2]
label_paths = im_label_paths[1::2]

for i, (ip, lp) in enumerate(zip(im_paths, label_paths)):
    im = cv2.imread(ip)
    if im is None:
        print("cv2.imread({}) fail".format(im))
        continue
    
    labels = np.loadtxt(lp).reshape(-1, 6)
    for c, i, x, y, w, h in labels:
        x, w = int(x * im.shape[1]), int(w * im.shape[1])
        y, h = int(y * im.shape[0]), int(h * im.shape[0])
        t, l = y - h // 2, x - w // 2
        b, r = y + h // 2, x + w // 2
        cv2.rectangle(im, (l, t), (r, b), (0, 255, 255))
    
    cv2.imwrite(os.path.join('labels', '%04d.jpg' % i), im)