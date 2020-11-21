import torch
from torch import nn
from mot.models.builder import BLOCKS

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, tensors):
        return sum(tensors)

class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()
    
    def forward(self, tensors):
        return torch.cat(tensors, dim=1)

@BLOCKS.register_module()
class Merge(nn.Module):
    '''Merge multiple input tensors.
    
    Param
    -----
    method: Merge method. It can be set as 'cat' or 'add'.
            'cat' means concatenate along channel dimension,
            'add' means element-wise sum. Default: 'cat'.
    '''
    def __init__(self, method='cat'):
        super(Merge, self).__init__()
        assert(method in ['cat', 'add'])
        self.merge = Sum() if method == 'add' else Cat()
    
    def forward(self, tensors):
        """Merge forward"""
        return self.merge(tensors)