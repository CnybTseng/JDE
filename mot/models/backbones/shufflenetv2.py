from torch import nn
from mot.models.builder import BACKBONES

@BACKBONES.register_module()
class ShuffleNetV2(nn.Module):
    def __init__(self, *args):
        super(ShuffleNetV2, self).__init__()
    
    def forward(self, input):
        pass