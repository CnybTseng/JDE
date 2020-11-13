# -*- coding: utf-8 -*-
# file: darknet2pytorch.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/11/7

import os
import sys
import torch
import argparse
import numpy as np

sys.path.append('.')

import darknet
import dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch-model', '-pm', type=str, dest='pm', help='pytorch-format model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset root path')
    parser.add_argument('--num-classes', type=int, default=1, help='number of classes')
    parser.add_argument('--darknet-model', '-dm', type=str, dest='dm', default='darknet.weights', help='darknet-format model file')
    parser.add_argument('--load-backbone-only', '-lbo', dest='lbo', help='only load the backbone', action='store_true')
    args = parser.parse_args()
    
    dataset = dataset.HotchpotchDataset(args.dataset, './data/train.txt')
    num_ids = int(dataset.max_id + 1)
    print(num_ids)
    
    model = darknet.DarkNet(np.random.randint(0, 100, (12, 2)), num_classes=args.num_classes, num_ids=num_ids)
    
    with open(args.dm, 'rb') as file:
        major = np.fromfile(file, dtype=np.int32, count=1)
        minor = np.fromfile(file, dtype=np.int32, count=1)
        revision = np.fromfile(file, dtype=np.int32, count=1)
        seen = np.fromfile(file, dtype=np.int64, count=1)
        print(f'darknet model version:{major.data[0]}.{minor.data[0]}.{revision.data[0]}')
        
        last_conv = None
        last_name = None
        for name, module in model.named_modules():
            if args.lbo and name is 'pair1':
                break
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                last_name = name
                if name is 'conv3':
                    if last_conv.bias is not None:
                        print(f'write to {last_name}.bias')
                        bias = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=last_conv.bias.numel()))
                        assert bias.numel() == last_conv.bias.numel()
                        last_conv.bias.data.copy_(bias.view_as(last_conv.bias))
                    print(f'write to {last_name}')
                    weight = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=last_conv.weight.numel()))
                    assert weight.numel() == last_conv.weight.numel()
                    last_conv.weight.data.copy_(weight.view_as(last_conv.weight))
            elif isinstance(module, torch.nn.BatchNorm2d):
                print(f'write to {name}')
                bias = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=module.bias.numel()))
                assert bias.numel() == module.bias.numel()
                weight = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=module.weight.numel()))
                assert weight.numel() == module.weight.numel()
                running_mean = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=module.running_mean.numel()))
                assert running_mean.numel() == module.running_mean.numel()
                running_var = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=module.running_var.numel()))
                assert running_var.numel() == module.running_var.numel()
                module.bias.data.copy_(bias.view_as(module.bias))
                module.weight.data.copy_(weight.view_as(module.weight))
                module.running_mean.data.copy_(running_mean.view_as(module.running_mean))
                module.running_var.data.copy_(running_var.view_as(module.running_var))
                if last_conv is not None:
                    print(f'write to {last_name}')
                    weight = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=last_conv.weight.numel()))
                    assert weight.numel() == last_conv.weight.numel()
                    last_conv.weight.data.copy_(weight.view_as(last_conv.weight))
                    last_conv = None
                else:
                    print(f"the module before {name} isn't Conv2d, that's impossible!!!")
                    break
            else:
                if last_conv is not None:
                    if last_conv.bias is not None:
                        print(f'write to {last_name}.bias')
                        bias = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=last_conv.bias.numel()))
                        assert bias.numel() == last_conv.bias.numel()
                        last_conv.bias.data.copy_(bias.view_as(last_conv.bias))
                    print(f'write to {last_name}')
                    weight = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=last_conv.weight.numel()))
                    assert weight.numel() == last_conv.weight.numel()
                    last_conv.weight.data.copy_(weight.view_as(last_conv.weight))
                last_conv = None
        torch.save(model.state_dict(), f'{args.pm}')
        file.close()