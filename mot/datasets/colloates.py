import torch
import numpy as np
from functools import partial
import torch.nn.functional as F
from mot.datasets.builder import COLLATES

@COLLATES.register_module()
class TrackerCollate(object):
    '''Merges a list of samples to form a mini-batch of Tensor(s).
    
    Param
    -----
    multiscale: Enable multi-scale training or not.
        Default: True.
    '''
    def __init__(self, multiscale=True):
        self._multiscale =  multiscale

    def __call__(self, *args, **kwargs):
        """Get collate_fn"""
        def collate(batch, shared_size=None):
            images, targets = [], []
            for image_id, (image, target) in enumerate(batch):
                # Target format: image_id, class, identity, x, y, w, h
                target[:, 0] = image_id
                images.append(image)
                targets.append(target)
            images = torch.stack(tensors=images, dim=0)
            if self._multiscale:
                images = F.interpolate(images,
                    size=shared_size.numpy().tolist(), mode='area')
            targets = torch.cat(tensors=targets, dim=0)
            return images, targets, shared_size
        return partial(collate, *args, **kwargs)
    
    @property
    def multiscale(self):
        return self._multiscale