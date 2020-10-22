import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from count_hooks import *

import os
import sys
import numpy as np
sys.path.append('.')
import darknet
import argparse
import collections
import shufflenetv2v2 as shufflenetv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample
}


def profile(model, inputs, custom_ops=None, verbose=True):
    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logger.warning("Either .total_ops or .total_params is already defined in %s."
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")

    return total_ops, total_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-size', type=str, default='320,576', help='network input size')
    parser.add_argument('--model', type=str, default='', help='model file')
    parser.add_argument('--dataset', type=str, default='', help='dataset path')
    parser.add_argument('--num-classes', type=int, default=3, help='number of classes')
    parser.add_argument('--pruned-model', '-pm', action='store_true')
    parser.add_argument('--backbone', type=str, default='darknet53', help='backbone architecture[darknet53(default),shufflenetv2]')
    parser.add_argument('--thin', type=str, default='2.0x',
        help='shufflenetv2 thin, default is 2.0x, candidates are 0.5x, 1.0x, 1.5x')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_size = [int(insz) for insz in args.in_size.split(',')]
    
    if not args.pruned_model:
        dummy_anchors = np.random.randint(0, 100, (12, 2))
        if args.backbone == 'darknet53':
            model = darknet.DarkNet(dummy_anchors).to(device)
        elif args.backbone == 'shufflenetv2':
            model = shufflenetv2.ShuffleNetV2(dummy_anchors, model_size=args.thin).to(device)
        else:
            print('unknown backbone architecture!')
            sys.exit(0)
        model_dict = model.state_dict()
        trained_model_dict = torch.load(args.model, map_location='cpu')
        trained_model_dict = {k : v for (k, v) in trained_model_dict.items() if k in model_dict}
        trained_model_dict = collections.OrderedDict(trained_model_dict)
        model_dict.update(trained_model_dict)
        model.load_state_dict(model_dict)
    else:
        model = torch.load(args.model, map_location=device)
    
    model.eval()
    input = torch.randn(1, 3, in_size[0], in_size[1]).to(device)
    flops, params = profile(model, inputs=(input, ))
    print(f'flops={flops}, params={params}')