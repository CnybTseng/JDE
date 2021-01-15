import logging

def get_logger(name='root', path=None):
    '''Get logger.
    
    Param
    -----
    name: Logger name.
    path: Log filepath.
    
    Return
    ------
    The created logger.
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(path, encoding='utf-8')
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger