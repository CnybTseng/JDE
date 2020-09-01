# -*- coding: utf-8 -*-
# file: toonnx.py
# brief: Joint Detection and Embedding
# author: Zeng Zhiwei
# date: 2020/5/8

import os
import torch
import argparse
import collections
import torch.onnx as onnx
import onnxruntime as ort
import numpy as np
import darknet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch-model', '-pm', type=str,
        help='path to the pytorch model')
    parser.add_argument('--onnx-model', '-om', type=str,
        default='./generated.onnx', help='path to the onnx model'
        ', the default value is ./generated.onnx')
    parser.add_argument('--insize', type=int, nargs='+',
        default=[320,576], help='network input size (height, width)'
        ', the default value are 320 576')
    parser.add_argument('--full-model', action='store_true',
        help='load full pytorch model, not only state dict')
    return parser.parse_args()
    
def toonnx(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.full_model:
        model = darknet.DarkNet(np.random.randint(0, 100, (12, 2))).to(device)
        model_state_dict = model.state_dict()
        state_dict = torch.load(args.pytorch_model, map_location=device)
        state_dict = {k:v for k,v in state_dict.items() if k in model_state_dict}
        state_dict = collections.OrderedDict(state_dict)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
    else:
        print('Warning: this function has not been tested yet!')
        model = torch.load(args.pytorch_model)
    
    dummy_input = torch.rand(1, 3, args.insize[0], args.insize[1], device=device)
    onnx.export(model, dummy_input, args.onnx_model, verbose=True, input_names=['data'])
    
    session = ort.InferenceSession(args.onnx_model)
    outputs = session.run(None, {'data':dummy_input.cpu().numpy()})
    for i, output in enumerate(outputs):
        print('branch {} output size is {}'.format(i, output.shape))

if __name__ == '__main__':
    args = parse_args()
    toonnx(args)