import os
import sys
import torch
import argparse
import collections
import numpy as np

sys.path.append('.')

import darknet
import shufflenetv2
import yolov3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
        help='path to the model')
    parser.add_argument('--input', type=str,
        help='path to the input tensor')
    parser.add_argument('--output', type=str,
        help='path to the tensorrt inference output tensor')
    args = parser.parse_args()
    
    anchors = ((6,16),   (8,23),    (11,32),   (16,45),
               (21,64),  (30,90),   (43,128),  (60,180),
               (85,255), (120,360), (170,420), (340,320))
    
    model = shufflenetv2.ShuffleNetV2(anchors, model_size='0.5x')
    
    state_dict = model.state_dict()
    train_state_dict = torch.load(args.model)
    train_state_dict = {k:v for k,v in train_state_dict.items() if k in state_dict.keys()}
    train_state_dict = collections.OrderedDict(train_state_dict)
    state_dict.update(train_state_dict)
    model.load_state_dict(state_dict)
    
    model.eval()
    
    x = torch.from_numpy(np.fromfile(args.input, np.float32)).view(1, 3, 320, 576)
    y = torch.from_numpy(np.fromfile(args.output, np.float32)).view(1, 152, 10, 18)
    print('{}'.format(x[0,0,0,:10]))
    print('{}'.format(y[0,0,:,:]))
    
    with torch.no_grad():
        ys = model(x)
    for yi in ys:
        print('{}'.format(yi.size()))
    
    print('{}'.format(ys[0][0,0,:,:]))
    delta = torch.abs(y - ys[0])
    print('output error:{} {} {}'.format(delta.min(), delta.max(), delta.mean()))