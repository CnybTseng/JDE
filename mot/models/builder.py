from torch import nn
from mot.models.utils import Registry, build_from_config

BACKBONES = Registry('backbones')
NECKS = Registry('necks')
HEADS = Registry('heads')
LOSSES = Registry('losses')
TRACKERS = Registry('trackers')

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
    build(BACKBONES, config)

def build_neck(config):
    """Build neck"""
    build(NECKS, config)
    
def build_head(config):
    """Build task head"""
    build(HEADS, config)

def build_loss(config):
    """Build loss"""
    build(LOSSES, config)

def build_tracker(config):
    """Build multiple object tracker"""
    build(TRACKERS, config)