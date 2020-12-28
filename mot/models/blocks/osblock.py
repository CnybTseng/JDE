import torch
from torch import nn

class LiteConv3x3(nn.Module):
    """Lite 3x3 convolution"""
    def __init__(self):
        super(LiteConv3x3, self).__init__()
    
    def forward(self, input):
        return input

class AG(nn.Module):
    """Aggregation gate"""
    def __init__(self):
        super(AG, self).__init__()
    
    def forward(self, input):
        return input

class OSBlock(nn.Module):
    """Omni-scale block"""
    def __init__(self):
        super(OSBlock, self).__init__()
    
    def forward(self, input):
        return input

if __name__ == '__main__':
    print('test OSBlock')