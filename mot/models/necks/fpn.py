from torch import nn
import torch.nn.functional as F
from mot.models import Merge
from mot.models.builder import NECKS

@NECKS.register_module()
class FPN(nn.Module):
    '''Feature Pyramid Networks for Object Detection.
    
    Param
    -----
    in_channels     : Number of input channels per scale.
    out_channels    : Number of output channels for each scale. Default: 128.
    num_outputs     : Number of output scales.
    start_level     : The start index of backbone output level to build the FPN.
                      Default: 0.
    end_level       : The end index of backbone output level (exclusive) to build the FPN.
                      Default: -1, which means the last level.
    extra_conv_input: The input of extra convolutional layer. It can be set as
                      'input', 'lateral', or 'output'. The 'input' means the last
                      backbone output, 'lateral' means the output of the last lateral
                      convolution, and 'output' means the output of the last FPN
                      convolution.
    merge_method    : Lateral and FPN feature map merging method. It can be set as
                      'cat' or 'add', 'cat' means concatenate along channel dimension,
                      and 'add' means element-wise sum. Default: 'cat'.
    '''
    def __init__(self,
        in_channels=[48, 96, 192],
        out_channels=128,
        num_outputs=3,
        start_level=0,
        end_level=-1,
        extra_conv_input='input',
        merge_method='cat'):
        super(FPN, self).__init__()
        
        if not isinstance(in_channels, list):
            raise TypeError('in_channels must be a list, but got {}'
                ''.format(type(in_channels)))
        
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
            assert(num_outputs >= self.end_level - start_level)
        else:
            self.end_level = end_level
            assert(end_level <= len(in_channels))
            assert(num_outputs == end_level - start_level)
        assert(extra_conv_input in ['input', 'lateral', 'output'])
        self.extra_conv_input = extra_conv_input
        assert(merge_method in ['cat', 'add'])
        self.merge_method = merge_method
        self.merge = Merge(merge_method)

        self.lateral, self.fpn = nn.ModuleList(), nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # Lateral convolution.
            if (i == self.end_level - 1) or (self.merge_method == 'add'):
                self.lateral.append(nn.Sequential(
                    nn.Conv2d(in_channels[i+self.start_level], out_channels,
                        kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)))
            else:
                self.lateral.append(nn.Sequential())
            
            # Top-down FPN convolution.
            # Only the upsampled and merged feature need to be convolved
            # to reduce the aliasing effect of upsampling.
            if (i < self.end_level - 1):
                if merge_method == 'cat':
                    in_channels_i = in_channels[i] + out_channels
                else:
                    in_channels_i = out_channels
                self.fpn.append(nn.Sequential(
                    nn.Conv2d(in_channels_i, out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)))
            else:
                self.fpn.append(nn.Sequential())
        
        extra_levels = num_outputs - (self.end_level - self.start_level)
        for i in range(extra_levels):
            if i == 0 and self.extra_conv_input == 'input':
                in_channels = self.in_channels[self.end_level - 1]
            else:
                in_channels = out_channels
            self.fpn.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        
        self._init_weights()
    
    def _init_weights(self):
        # A bad implementation for now!
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, input):
        """FPN forward"""
        assert(len(input) == len(self.in_channels))
        # Lateral path.
        output = []
        for i, module in enumerate(self.lateral):
            output.append(module(input[i + self.start_level]))
        
        # Top-down path.
        use_backbone_level = len(self.lateral)
        for i in range(use_backbone_level - 1, 0, -1):
            size = output[i - 1].shape[2:]
            output[i - 1] = self.merge([output[i - 1],
                F.interpolate(output[i], size)])
            if self.merge_method == 'cat':
                output[i - 1] = self.fpn[i - 1](output[i - 1])
        if self.merge_method == 'cat':
            return tuple(output)
        
        fpn_output = [None] * len(output)
        for i in range(use_backbone_level):
            fpn_output[i] = self.fpn[i](output[i])
        if self.num_outputs == len(fpn_output):
            return tuple(fpn_output)
        
        # Extra output, down-top path.
        if self.extra_conv_input == 'input':
            extra_input = input[self.end_level - 1]
        elif self.extra_conv_input == 'lateral':
            extra_input = output[-1]
        elif self.extra_conv_input == 'output':
            extra_input = fpn_output[-1]
        fpn_output.append(self.fpn[use_backbone_level](extra_input))
        for i in range(use_backbone_level + 1, self.num_outputs):
            fpn_output.append(self.fpn[i](fpn_output[-1]))       
        return tuple(fpn_output)