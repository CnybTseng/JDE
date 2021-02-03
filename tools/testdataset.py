import os
import cv2
import sys
import torch
import argparse
import numpy as np
sys.path.append(os.getcwd())
from mot.utils import config
from mot.datasets import build_dataset, build_dataloader

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
    
    # Build dataset
    dataset = build_dataset(config.DATASET)
    
    # Build dataloader.
    dataloader = build_dataloader(dataset, config)
    print(len(dataset))
    for epoch in range(1):
        print('****************************************************')
        i = 0
        for index, batch in enumerate(dataloader):
            print('{} {}'.format(index, batch[0].shape))
            label = batch[1]
            im_id = label[:, 0].long()
            for j, tensor in enumerate(batch[0]):
                im = tensor.permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
                labeli = label[im_id == j]
                for lab in labeli:
                    x = int(im.shape[1] * lab[3])
                    y = int(im.shape[0] * lab[4])
                    w = int(im.shape[1] * lab[5])
                    h = int(im.shape[0] * lab[6])
                    l = x - w // 2
                    t = y - h // 2
                    r = x + w // 2
                    b = y + h // 2
                    im = cv2.rectangle(im, (l, t), (r, b), (0, 255, 255))
                cv2.imwrite('mosaic/%06d.jpg' % i, im)
                i += 1
            
if __name__ == '__main__':
    main()