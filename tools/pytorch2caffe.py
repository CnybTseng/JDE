# -*- coding: utf-8 -*-
# file: pytorch2caffe.py
# brief: Convert PyTroch-format model to caffe-format model through google protobuf.
# author: Zeng Zhiwei
# date: 2020/7/20

import os
import sys
import torch
sys.path.append('.')
import darknet
import argparse
import numpy as np
import collections
import caffe_pb2 as pb

def write_prototxt(net, filename, in_size):
    eltwiseOps = ['PROD', 'SUM', 'MAX']
    with open(filename, 'w') as file:
        file.write(f"name: \"{net.name}\"\n")
        file.write("input: \"data\"\n")
        file.write("input_shape {\n")
        file.write(" "*4+"dim: 1\n")
        file.write(" "*4+"dim: 3\n")
        file.write(" "*4+f"dim: {in_size[0]}\n")
        file.write(" "*4+f"dim: {in_size[1]}\n")
        file.write('}\n')
        for layer in net.layer:
            file.write("layer {\n")
            for bottom in layer.bottom:
                file.write(" "*4+f"bottom: \"{bottom}\"\n")
            file.write(" "*4+f"top: \"{layer.top[0]}\"\n")
            file.write(" "*4+f"name: \"{layer.name}\"\n")
            file.write(" "*4+f"type: \"{layer.type}\"\n")
            if layer.type == 'Convolution':
                file.write(" "*4+"convolution_param {\n")
                file.write(" "*8+f"num_output: {layer.convolution_param.num_output}\n")
                file.write(" "*8+f"kernel_size: {layer.convolution_param.kernel_size[0]}\n")
                file.write(" "*8+f"pad: {layer.convolution_param.pad[0]}\n")
                file.write(" "*8+f"stride: {layer.convolution_param.stride[0]}\n")
                file.write(" "*8+f"bias_term: {layer.convolution_param.bias_term}\n".lower())
                file.write(' '*4+'}\n')
            elif layer.type == 'BatchNorm':
                file.write(" "*4+"batch_norm_param {\n")
                file.write(" "*8+f"use_global_stats: {layer.batch_norm_param.use_global_stats}\n".lower())
                file.write(' '*4+'}\n')
            elif layer.type == 'Scale':
                file.write(" "*4+"scale_param {\n")
                file.write(" "*8+f"bias_term: {layer.scale_param.bias_term}\n".lower())
                file.write(' '*4+'}\n')
            elif layer.type == 'ReLU':
                file.write(" "*4+"relu_param {\n")
                file.write(" "*8+f"negative_slope: {layer.relu_param.negative_slope:.1f}\n")
                file.write(' '*4+'}\n')
            elif layer.type == 'Eltwise':
                file.write(" "*4+"eltwise_param {\n")
                file.write(" "*8+f"operation: {eltwiseOps[layer.eltwise_param.operation]}\n")
                file.write(' '*4+'}\n')
            elif layer.type == 'Concat':
                pass
            elif layer.type == 'Upsample':
                file.write(" "*4+"upsample_param {\n")
                file.write(" "*8+f"scale: {layer.upsample_param.scale}\n")
                file.write(' '*4+'}\n')
            else:
                print(f"unknown layer type:{layer.type}!")
            file.write("}\n")
        file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch-model', '-pm', dest='pmodel', help='PyTorch-format model path')
    parser.add_argument('--in-size', type=str, default='320,576', help='network input size')
    parser.add_argument('--num-classes', type=int, default=1, help='number of classes')
    parser.add_argument('--caffe-model', '-cm', dest='cmodel', help='Caffe-format model path')
    parser.add_argument('--pruned-model', action='store_true')
    args = parser.parse_args()
    print(args)
    
    in_size = [int(size) for size in args.in_size.split(',')]
    if os.path.isfile(args.cmodel):
        file = open(args.cmodel, 'rb')
        net = pb.NetParameter()
        net.ParseFromString(file.read())
        file.close()
        prototxt = args.cmodel.replace('caffemodel', 'prototxt')
        write_prototxt(net, prototxt, in_size)
        sys.exit(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.pruned_model:
        dummy_anchors = np.random.randint(0, 100, (12, 2))
        model = darknet.DarkNet(dummy_anchors)        
        # load state dict except the classifier layer
        model_dict = model.state_dict()
        trained_model_dict = torch.load(args.pmodel, map_location='cpu')
        trained_model_dict = {k : v for (k, v) in trained_model_dict.items() if k in model_dict}
        trained_model_dict = collections.OrderedDict(trained_model_dict)
        model_dict.update(trained_model_dict)
        model.load_state_dict(model_dict)
    else:
        model = torch.load(args.pmodel, map_location=device)
    
    # we have to process route layer specially!    
    route_bottoms = [
        ['cbrl7_relu'],   ['upsample1', 'stage4_7_ew'],
        ['pair2_3_relu'], ['upsample2', 'stage3_7_ew'],
        ['cbrl7_relu'],   ['conv1', 'conv4'],
        ['pair2_3_relu'], ['conv2', 'conv5'],
        ['pair3_3_relu'], ['conv3', 'conv6']]
    
    net = pb.NetParameter()
    net.name = 'JDE'
    
    bottom = 'data'
    skip_bottom = ''        # for Residual(Eltwise)
    residual = False        # processing Residual block or not
    for _name, module in model.named_modules():
        name = _name.replace('.', '_')
        if isinstance(module, torch.nn.Conv2d):         # Convolution
            print(f'export {name}')
            layer = net.layer.add()
            layer.name = name
            layer.type = 'Convolution'
            layer.bottom.append(bottom)
            layer.top.append(name)
            layer.convolution_param.num_output = module.out_channels
            layer.convolution_param.bias_term = module.bias is not None
            layer.convolution_param.pad.append(module.kernel_size[0]==3)
            layer.convolution_param.kernel_size.append(module.kernel_size[0])
            layer.convolution_param.stride.append(module.stride[0])
            if residual and 'conv1' in name:
                skip_bottom = bottom                    # first bottom of Eltwise layer
            bottom = name
            
            layer.blobs.append(pb.BlobProto())
            layer.blobs[-1].shape.dim.append(module.weight.size(0))
            layer.blobs[-1].shape.dim.append(module.weight.size(1))
            layer.blobs[-1].shape.dim.append(module.weight.size(2))
            layer.blobs[-1].shape.dim.append(module.weight.size(3))
            for data in module.weight.detach().cpu().numpy().flatten():
                layer.blobs[-1].data.append(data)
            if module.bias is not None:
                layer.blobs.append(pb.BlobProto())
                layer.blobs[-1].shape.dim.append(module.bias.size(0))
                for data in module.bias.detach().cpu().numpy().flatten():
                    layer.blobs[-1].data.append(data)
        elif isinstance(module, torch.nn.BatchNorm2d):  # BatchNorm and Scale
            print(f'export {name}')
            bn_layer = net.layer.add()
            bn_layer.name = name
            bn_layer.type = 'BatchNorm'
            bn_layer.bottom.append(bottom)
            bn_layer.top.append(name)
            bn_layer.batch_norm_param.use_global_stats = 1
            
            bn_layer.blobs.append(pb.BlobProto())       # mean
            bn_layer.blobs[-1].shape.dim.append(module.running_mean.size(0))
            for data in module.running_mean.detach().cpu().numpy():
                bn_layer.blobs[-1].data.append(data)
            bn_layer.blobs.append(pb.BlobProto())       # variance
            bn_layer.blobs[-1].shape.dim.append(module.running_var.size(0))
            for data in module.running_var.detach().cpu().numpy():
                bn_layer.blobs[-1].data.append(data)
            bn_layer.blobs.append(pb.BlobProto())       # moving average factor
            bn_layer.blobs[-1].shape.dim.append(1)
            bn_layer.blobs[-1].data.append(1)
            
            name_ = name.replace('norm', 'scale')
            scale_layer = net.layer.add()
            scale_layer.name = name_
            scale_layer.type = 'Scale'
            scale_layer.bottom.append(name)
            scale_layer.top.append(name_)
            scale_layer.scale_param.bias_term = True
            bottom = name_
            
            scale_layer.blobs.append(pb.BlobProto())
            scale_layer.blobs[-1].shape.dim.append(module.weight.size(0))
            for data in module.weight.detach().cpu().numpy():
                scale_layer.blobs[-1].data.append(data)
            scale_layer.blobs.append(pb.BlobProto())
            scale_layer.blobs[-1].shape.dim.append(module.bias.size(0))
            for data in module.bias.detach().cpu().numpy():
                scale_layer.blobs[-1].data.append(data)
        elif isinstance(module, darknet.Residual):      # Eltwise
            residual = True                             # beginning of Residual block
        elif isinstance(module, torch.nn.LeakyReLU):    # ReLU with(out) Eltwise
            print(f'export {name}')
            relu_layer = net.layer.add()
            relu_layer.name = name
            relu_layer.type = 'ReLU'
            relu_layer.bottom.append(bottom)
            relu_layer.top.append(name)
            relu_layer.relu_param.negative_slope = module.negative_slope
            bottom = name
            if residual and 'relu2' in name:
                if skip_bottom == '':
                    print('skip_bottom of Eltwise is empty!')
                    sys.exit(0)
                name_ = name.replace('relu2', 'ew')
                ew_layer = net.layer.add()
                ew_layer.name = name_
                ew_layer.type = 'Eltwise'
                ew_layer.bottom.append(skip_bottom)
                ew_layer.bottom.append(bottom)
                ew_layer.top.append(name_)
                ew_layer.eltwise_param.operation = 1    # 1:SUM
                residual = False                        # end of Residual block
                skip_bottom = ''
                bottom = name_                
        elif isinstance(module, darknet.Upsample):      # Upsample
            print(f'export {name}')
            layer = net.layer.add()
            layer.name = name
            layer.type = 'Upsample'
            layer.bottom.append(bottom)
            layer.top.append(name)
            layer.upsample_param.scale = module.scale_factor
            bottom = name
        elif isinstance(module, darknet.Route):         # Concat
            print(f'export {name}')
            route_id = int(name[5:]) - 1
            if route_id in [1, 3, 5, 7, 9]:             # Concat
                layer = net.layer.add()
                layer.name = name
                layer.type = 'Concat'
                print(f'->{bottom} {route_bottoms[route_id]}')
                layer.bottom.append(route_bottoms[route_id][0])
                layer.bottom.append(route_bottoms[route_id][1])
                layer.top.append(name)
                bottom = name
            else:                                       # only one bottom
                bottom = route_bottoms[route_id][0]

    with open(args.cmodel, 'wb') as file:
        file.write(net.SerializeToString())
        file.close()
    
    prototxt = args.cmodel.replace('caffemodel', 'prototxt')
    write_prototxt(net, prototxt, in_size)