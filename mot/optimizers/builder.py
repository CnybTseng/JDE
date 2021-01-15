from torch import optim
from torch.optim import lr_scheduler
from mot.utils import Registry, build_from_config

OPTIMIZERS = Registry('optimizers')
OPTIMIZERS.register_module(module_name='SGD', module_class=optim.SGD)
OPTIMIZERS.register_module(module_name='Adam', module_class=optim.Adam)

LR_SCHEDULERS = Registry('lr_schedulers')
LR_SCHEDULERS.register_module(module_name='MultiStepLR',
    module_class=lr_scheduler.MultiStepLR)

def build_optimizer(config):
    return build_from_config(OPTIMIZERS, config)

def build_lr_scheduler(config):
    return build_from_config(LR_SCHEDULERS, config)