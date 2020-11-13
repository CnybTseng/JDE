from torch import nn
from mot.models.builder import NECKS

@NECKS.register_module()
class FPN(nn.Module):
    def __init__(self, *args):
        super(FPN, self).__init__()
    
    def forward(self, input):
        pass