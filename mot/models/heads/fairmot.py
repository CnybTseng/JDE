import torch
from torch import nn
from mot.models.builder import HEADS

@HEADS.register_module()
class FairMOTHead(nn.Module):
    '''FairMOTHead
    '''
    def __init__(self):
        super(FairMOTHead, self).__init__()
    
    def forward(self, input):
        """FairMOTHead forward"""
        return input