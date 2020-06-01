import os
import sys
import torch
import argparse
import numpy as np

import darknet

def parse_args():
    parser = argparse.ArgumentParser(description='Export weights of trained model from D. Bolya')
    parser.add_argument('--trained-model', type=str, default='', help='path to the trained model')
    parser.add_argument('--exported-model', type=str, default='', help='path to the exported model')
    parser.add_argument('--export-backbone-only', '-ebo', dest='ebo', help='export backbone only')
    args = parser.parse_args()
    return args

def main(args):
    net = darknet.DarkNet(np.random.randint(0, 100, (12, 2)), num_ids=165)
    
    state_dict = torch.load(args.trained_model, map_location=torch.device('cpu'))
    modules = state_dict['model']

    weights = []
    for key, value in modules.items():
        if not 'num_batches_tracked' in key and not 'yolo' in key:
            weights.append(value)
            print(f"{key} {value.size()}")
    
    i = 0
    for name, module in net.named_modules():
        if args.ebo and name is 'pair1':
            break
        if isinstance(module, torch.nn.Conv2d):
            print(f"write to {name}.weight")
            assert module.weight.size() == weights[i].size()
            module.weight.data = weights[i].data.clone()
            i += 1
            if module.bias is not None:
                print(f"write to {name}.bias")
                assert module.bias.size() == weights[i].size()
                module.bias.data = weights[i].data.clone()
                i += 1
        elif isinstance(module, torch.nn.BatchNorm2d):
            print(f"write to {name}")
            assert module.weight.size() == weights[i].size()
            module.weight.data = weights[i].data.clone()
            i += 1
            assert module.bias.size() == weights[i].size()
            module.bias.data = weights[i].data.clone()
            i += 1
            assert module.running_mean.size() == weights[i].size()
            module.running_mean.data = weights[i].data.clone()
            i += 1
            assert module.running_var.size() == weights[i].size()
            module.running_var.data = weights[i].data.clone()
            i += 1
        elif isinstance(module, torch.nn.Linear):
            print(f"write to {name}")
            assert module.weight.size() == weights[i].size()
            module.weight.data = weights[i].data.clone()
            i += 1
            assert module.bias.size() == weights[i].size()
            module.bias.data = weights[i].data.clone()
    
    torch.save(net.state_dict(), args.exported_model)

if __name__ == '__main__':
    args = parse_args()
    main(args)