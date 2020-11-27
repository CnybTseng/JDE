import torch
from torch import nn
from mot.models.builder import BLOCKS

BLOCKS.register_module(module_name='Conv2d', module_class=nn.Conv2d)