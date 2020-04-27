import sys
import torch
import random
import timeit
import argparse
import torch.nn as nn
import numpy as np

import yolov3

def parse_args():
    parser = argparse.ArgumentParser(
        description='test YOLOv3Loss layer')
    parser.add_argument('--insize', type=str, default='320x576',
        help='network input size, default=320x576')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    h, w = [int(s) for s in args.insize.split('x')]
    if '320x576' in args.insize:
        anchors = ((6,16),   (8,23),    (11,32),   (16,45),
                   (21,64),  (30,90),   (43,128),  (60,180),
                   (85,255), (120,360), (170,420), (340,320))
    elif '480x864' in args.insize:
        anchors = ((6,19),    (9,27),    (13,38),   (18,54),
                   (25,76),   (36,107),  (51,152),  (71,215),
                   (102,305), (143,429), (203,508), (407,508))
    elif '608x1088' in args.insize:
        anchors = ((8,24),    (11,34),   (16,48),   (23,68),
                   (32,96),   (45,135),  (64,192),  (90,271),
                   (128,384), (180,540), (256,640), (512,640))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = nn.Linear(512, 1000)
    criterion = yolov3.YOLOv3Loss(1, anchors, 1000, classifier).to(device)   
    
    count = 0
    duration = 0
    gw, gh = w // 32, h // 32
    while 1:
        xs = [torch.rand((8, 24 + 512, gh * 2 ** i, gw * 2 ** i)).to(device) for i in range(3)]
        
        targets = []
        for b in range(8):
            if np.random.rand() < 0.5:
                continue
            
            n = np.random.randint(1, 10, size=(1,)).item()
            target = torch.rand(n, 7).to(device)
            target[:, 0] = b
            target[:, 2] = torch.FloatTensor(random.sample(range(100), n)).to(device)
            targets.append(target)

        if len(targets) == 0:
            continue
        targets = torch.cat(targets, dim=0).to(device)
        print(targets.size())
        start = timeit.default_timer()
        loss, metrics = criterion(xs, targets, (h, w))
        end = timeit.default_timer()
        duration += (end - start)
        count += 1
        print(f"average execution time is {duration / count} seconds.")
        for metric in metrics:
            for key, val in metric.items():
                print(f"{key}:%.5f " % val, end='')
            print('')