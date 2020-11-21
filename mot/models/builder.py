from torch import nn
from mot.models.utils import Registry, build_from_config

BACKBONES = Registry('backbones')
NECKS = Registry('necks')
HEADS = Registry('heads')
LOSSES = Registry('losses')
TRACKERS = Registry('trackers')
BLOCKS = Registry('blocks')

def build(register, config):
    '''Build module from yaml format configurations.
    
    Param
    -----
    register: Module registry
    config  : YAML format configurations. The config must contain 'module_name'
              and 'args' entries.
    
    Return
    ------
    The corresponding module.
    '''
    if isinstance(config, list):
        modules = [build_from_config(register, c) for c in config]
        return nn.Sequential(*modules)
    return build_from_config(register, config)

def build_backbone(config):
    """Build backbone"""
    return build(BACKBONES, config)

def build_neck(config):
    """Build neck"""
    return build(NECKS, config)
    
def build_head(config):
    """Build task head"""
    return build(HEADS, config)

def build_loss(config):
    """Build loss"""
    return build(LOSSES, config)

def build_tracker(config):
    """Build multiple object tracker"""
    return build(TRACKERS, config)

def build_block(config):
    """Build universal blocks"""
    return build(BLOCKS, config)