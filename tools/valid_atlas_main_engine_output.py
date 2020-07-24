import cv2
import sys
import torch
import numpy as np

sys.path.append('.')

import kalman
import yolov3
import darknet
import tracker

if __name__ == '__main__':
    anchors = ((6,16),   (8,23),    (11,32),   (16,45),
               (21,64),  (30,90),   (43,128),  (60,180),
               (85,255), (120,360), (170,420), (340,320))
    decoder = yolov3.YOLOv3Decoder((320,576), 1, anchors).cuda()
    
    im = cv2.imread('./platforms/atlas200dk/run/out/test/data/000002.jpg')
    
    baselines = []
    baselines += [torch.from_numpy(np.fromfile('./platforms/atlas200dk/run/out/test/data/000002-10x18.bin',
        dtype=np.float32)).view(1, -1, 10, 18).cuda()]
    baselines += [torch.from_numpy(np.fromfile('./platforms/atlas200dk/run/out/test/data/000002-20x36.bin',
        dtype=np.float32)).view(1, -1, 20, 36).cuda()]
    baselines += [torch.from_numpy(np.fromfile('./platforms/atlas200dk/run/out/test/data/000002-40x72.bin',
        dtype=np.float32)).view(1, -1, 40, 72).cuda()]
    
    print('baselines:')
    for baseline in baselines:
        print(baseline.size())
    
    outputs = []
    outputs += [torch.from_numpy(np.fromfile('./platforms/atlas200dk/run/out/test/data/385920.bin',
        dtype=np.float32)).view(1, 10, 18, -1).permute(0, 3, 1, 2).cuda()]
    outputs += [torch.from_numpy(np.fromfile('./platforms/atlas200dk/run/out/test/data/1543680.bin',
        dtype=np.float32)).view(1, 20, 36, -1).permute(0, 3, 1, 2).cuda()]
    outputs += [torch.from_numpy(np.fromfile('./platforms/atlas200dk/run/out/test/data/6174720.bin',
        dtype=np.float32)).view(1, 40, 72, -1).permute(0, 3, 1, 2).cuda()]
    
    print('outputs:')
    for output in outputs:
        print(output.size())
    
    for baseline, output in zip(baselines, outputs):
        delta = torch.abs(baseline - output)
        print(f'min(diff) is {delta.min()}, max(diff) is {delta.max()}, mean(diff) is {delta.mean()}')
    
    outputs = decoder(outputs)
    
    outputs = tracker.nonmax_suppression(outputs, 0.5, 0.4)[0]
    outputs[:, :4] = tracker.ltrb_net2img(outputs[:, :4], (320,576), im.shape[:2])
    result = tracker.overlap(outputs, im)
    cv2.imwrite('./platforms/atlas200dk/run/out/test/data/atlas.png', result)