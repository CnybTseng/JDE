import math
import torch
import torch.nn.functional as F

class Stitcher(object):
    '''
    Stitcher: Feedback-driven Data Provider for Object Detection
    
    Params
    ------
    k     : The number of images to stitch.
    catdim: Concatenation dimension, 'batch' or 'spatial'.
    '''
    def __init__(self, k=4, catdim='batch'):
        self.k = k
        self.catdim = catdim
        self.accumulated_batch = []

    def __call__(self, batch):
        '''
        Stitching multiple batch images and targets,
        and keep the total volume not changed.
        
        Param
        -----
        batch: A mini batch including images and targets.
        
        Return
        ------
        Stitched mini batches, or empty batch (None, None) if the number
        of accumulatedbatches is less than 'self.k'.
        '''
        
        if self.k == 1:
            return batch
        
        self.accumulated_batch.append(batch)
        if len(self.accumulated_batch) < self.k:           
            return None, None
        
        if self.catdim == 'batch':
            return self._stitch_along_batch(self.k)
        else:
            return self._stitch_along_spatial(self.k)
    
    def _stitch_along_batch(self, k):
        s = 1 / math.sqrt(k)
        n1, c, h1, w1 = self.accumulated_batch[0][0].size()
        n2, h2, w2 = k * n1, round(s * h1), round(s * w1)
        images, targets = [], []
        for i, (image, target) in enumerate(self.accumulated_batch):
            images.append(F.interpolate(input=image, size=(h2, w2), mode='nearest'))
            # Update image index in stitched batch.
            target[:, 0] += i * n1
            targets.append(target)    
        self.accumulated_batch = []
        return torch.cat(images, dim=0), torch.cat(targets, dim=0)
    
    def _stitch_along_spatial(self, k):
        raise NotImplementedError

if __name__ == '__main__':
    stitcher = Stitcher()
    for i in range(16):
        image = torch.rand(64, 3, 320, 576)
        n = torch.randint(0, 500, (1,)).item()
        target = torch.rand((n, 7))
        batch = stitcher([image, target])
        print('target: {}'.format(target.size()))
        if batch[0] is None:
            continue
        print('stitch {} {} {}'.format(i, batch[0].size(), batch[1].size()))