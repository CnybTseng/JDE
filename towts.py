import torch
import struct
import argparse
import collections
import numpy as np

import darknet
import shufflenetv2

if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(
        description='export PyTorch weidhts to a .wts file')
    parser.add_argument('--pytorch-model', '-pm', type=str,
        help='path to the PyTroch model')
    parser.add_argument('--wts', type=str,
        help='path to the gernerated .wts file')
    parser.add_argument('--backbone', '-bb', type=str, default='shufflenetv2',
        help='backbone architecture, default is shufflenetv2'
        'other option is darknet')
    parser.add_argument('--thin', type=str, default='0.5x',
        help='shufflenetv2 backbone thin, default is 0.5x, other options'
        'are 1.0x, 1.5x, and 2.0x')
    args = parser.parse_args()
    
    # construct JDE model
    anchors = np.random.randint(low=0, high=200, size=(12,2))
    if args.backbone == 'shufflenetv2':
        model = shufflenetv2.ShuffleNetV2(anchors, model_size=args.thin)
    elif args.backbone == 'darknet':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # load weights if PyTorch model was given
    if args.pytorch_model:
        state_dict = model.state_dict()
        train_state_dict = torch.load(args.pytorch_model)
        # remove identifier classifier from train_state_dict
        train_state_dict = {k:v for k,v in train_state_dict.items() if k in state_dict.keys()}
        train_state_dict = collections.OrderedDict(train_state_dict)
        state_dict.update(train_state_dict)
        model.load_state_dict(state_dict)
    
    model.eval()
    
    # write layer name and corresponding weidhts to .wts file
    file = open(args.wts, 'w')
    file.write('{}\n'.format(len(model.state_dict().keys())))
    for k,v in model.state_dict().items():
        print('export {}, size is {}'.format(k, v.shape))
        ws = v.reshape(-1).cpu().numpy()
        file.write('{} {}'.format(k, len(ws)))
        for w in ws:
            file.write(' ')
            file.write(struct.pack('>f', float(w)).hex())
        file.write('\n')