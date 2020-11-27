import inspect
from yacs.config import CfgNode as CN

class Registry:
    '''This class is used to map module name to class.
    
    Param
    -----
    name: class registered name
    '''
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
    
    def __len__(self):
        return len(self._module_dict)
    
    def __contains__(self, module_name):
        return self.get(module_name) is not None
    
    def __repr__(self):
        return self.__class__.__name__ + '[{}, {}]'.format(
            self._name, self._module_dict)
    
    @property
    def name(self):
        return self._name
    
    @property
    def module_dict(self):
        return self._module_dict
    
    def get(self, module_name):
        '''Get class by its registered name.
        
        Param
        -----
        module_name: Class registered name
        
        Return
        ------
        The corresponding class.
        '''
        return self._module_dict.get(module_name, None)
    
    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module_class must be a class,'
                ' but got {}'.format(type(module_class)))
        
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} has already been registered'.format(module_name))
        
        self._module_dict[module_name] = module_class
    
    def register_module(self, module_name=None, force=False, module_class=None):
        '''Register a class module.
        
        Param
        -----
        module_name : The name of class need to be registered. If not specified,
                      module_class.__name__ will be used.
        force       : Override existing registered class module or not.
                      Default: False.
        module_class: The class need to be registered. Default: None.
        
        Return
        ------
        If module_class is not None, return module_class.
        Otherwise return module register.
        '''
        if not (module_name is None or isinstance(module_name, str)):
            raise TypeError('module_name must be a string,'
                ' but got {}'.format(type(module_name)))
        
        if not isinstance(force, bool):
            raise TypeError('force must be boolean, but got {}'.format(type(force)))
        
        if module_class is not None:
            self._register_module(module_class, module_name, force)
            return module_class
        
        def _register(_module_class):
            self._register_module(_module_class, module_name, force)
            return _module_class
        
        return _register

def build_from_config(register, config):
    '''Build module from yaml format configurations.
    
    Param
    -----
    register: Module registry
    config  : YAML format configurations. The config must contain 'NAME'
              and 'ARGS' entries.
    
    Return
    ------
    The corresponding module.
    '''
    if not isinstance(register, Registry):
        raise TypeError('register must be Registry type,'
            ' but got {}'.format(type(register)))
    
    if not isinstance(config, CN):
        raise TypeError('config must be CN type, but got {}'.format(type(config)))

    module_name = config.NAME
    if isinstance(module_name, str):
        module = register.get(module_name)
        if module is None:
            raise KeyError("{} has not been registered".format(module_name))
    else:
        raise TypeError("{} must be string type, but got {}".format(
            module_name, type(module_name)))
    
    if isinstance(config.ARGS, CN):
        return module(config.ARGS)
    elif isinstance(config.ARGS, list):
        return module(**config.ARGS[0])
    else:
        raise TypeError('{} must be a yacs.config.CfgNode or list,'
            ' but got {}'.format(config.ARGS, type(config.ARGS)))