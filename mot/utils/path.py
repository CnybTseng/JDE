import os
import os.path as osp

def mkdirs(dir, mode=0o777):
    if dir == '':
        return
    dir = osp.expanduser(dir)
    os.makedirs(dir, mode=mode, exist_ok=True)