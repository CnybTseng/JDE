from torch import nn
from mot.models.builder import HEADS

@HEADS.register_module()
class JDELoss(nn.Module):
    def __init__(self, *args):
        super(JDELoss, self).__init__()
    
    def forward(self, input):
        pass