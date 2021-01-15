from torch import nn
from collections import OrderedDict
from mot.models.builder import (TRACKERS,
    build_backbone, build_neck, build_head)

@TRACKERS.register_module()
class JDE(nn.Module):
    '''Towards Real-Time Multi-Object Tracking.
    
    Param
    -----
    config: YAML format configurations. The configurations must contain
            'BACKBONE', 'NECK', and 'HEAD' entries.
    '''
    def __init__(self, config):
        super(JDE, self).__init__()
        self.backbone = build_backbone(config.BACKBONE)
        self.neck = build_neck(config.NECK)
        self.head = build_head(config.HEAD)
    
    def forward(self, input, *args, **kwargs):
        """JDE forward"""
        for module in self.children():
            input = module(input, *args, **kwargs)
        return input
    
    def load_state_dict(self, state_dict, strict=True):
        """Same to torch.nn.Module.load_state_dict"""
        sd = self.state_dict()
        # Remove layers only for training if in test mode.
        state_dict = {k : v for (k, v) in state_dict.items() if k in sd}
        state_dict = OrderedDict(state_dict)
        sd.update(state_dict)
        super().load_state_dict(sd)