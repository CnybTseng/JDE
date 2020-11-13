from torch import nn
from mot.models.builder import (TRACKERS,
    build_backbone, build_neck, build_head)

@TRACKERS.register_module()
class JDE(nn.Module):
    def __init__(self, config):
        super(JDE, self).__init__()
        self.backbone = build_backbone(config.BACKBONE)
        self.neck = build_neck(config.NECK)
        self.head = build_head(config.HEAD)
    
    def forward(self, input):
        pass