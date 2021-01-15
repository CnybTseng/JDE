import os
import sys
import glob
import torch
import argparse
import collections
from torch import nn
from collections import defaultdict
sys.path.append('.')
import shufflenetv2

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', '-mp', type=str,
    help='path to models')
args = parser.parse_args()

avr_model = shufflenetv2.ShuffleNetV2(model_size='1.0x')
for name, module in avr_model.named_modules():
    if isinstance(module, nn.Conv2d):
        module.weight.data[...] = 0
        if module.bias is not None:
            module.bias.data[...] = 0
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data[...] = 0
        module.bias.data[...] = 0
        module.running_mean.data[...] = 0
        module.running_var.data[...] = 0

paths = sorted(glob.glob(os.path.join(args.model_path, '*.pth')))
paths = [path for path in paths if 'trainer-ckpt' not in path]
print(paths)
params = defaultdict(list)
for path in paths:
    model = shufflenetv2.ShuffleNetV2(model_size='1.0x')
    model_dict = model.state_dict()
    trained_model_dict = torch.load(path, map_location='cpu')
    trained_model_dict = {k : v for (k, v) in trained_model_dict.items() if k in model_dict}
    trained_model_dict = collections.OrderedDict(trained_model_dict)
    model_dict.update(trained_model_dict)
    model.load_state_dict(model_dict)
    with torch.no_grad():
        for (name, module), (_, avr_module) in zip(model.named_modules(), avr_model.named_modules()):
            if isinstance(module, nn.Conv2d):
                avr_module.weight.data += module.weight.data
                if module.bias is not None:
                    avr_module.bias.data += module.bias.data 
            elif isinstance(module, nn.BatchNorm2d):
                avr_module.weight.data += module.weight.data
                avr_module.bias.data += module.bias.data
                avr_module.running_mean.data += module.running_mean.data
                avr_module.running_var.data += module.running_var.data
    print('\rfusing {} done.'.format(path), end='', flush=True)

print('')
with torch.no_grad():
    for name, module in avr_model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data /= len(paths)
            if module.bias is not None:
                module.bias.data /= len(paths)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data /= len(paths)
            module.bias.data /= len(paths)
            module.running_mean.data /= len(paths)
            module.running_var.data /= len(paths)

torch.save(avr_model.state_dict(), os.path.join(args.model_path, 'avr_model.pth'))