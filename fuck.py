import math
import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class XXXDataset(IterableDataset):
    def __init__(self, start=0, end=100):
        super(XXXDataset, self).__init__()
        self.start = start
        self.end = end
        self.count = 0
    
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.iter_start = self.start
            self.iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.iter_start = self.start + worker_id * per_worker
            self.iter_end = min(self.iter_start + per_worker, self.end)
        self.count = self.iter_start
        self.worker_id = worker_id
        return self
    
    def __next__(self):        
        batch_size = np.random.choice([1, 4]).item()
        self.count += batch_size
        if self.count >= self.iter_end:
            raise StopIteration
        images, labels = [], []
        for i in range(batch_size):   
            images += [torch.rand(3, 320, 576)]
            labels += [torch.rand(1, 7)]
            labels[-1][:, -1] = self.count + i - batch_size
            labels[-1][:, -2] = self.worker_id
        return torch.stack(images, dim=0), torch.cat(labels)

def collate_fn(batch):
    images, labels = [], []
    for image_id, (image, label) in enumerate(batch):
        images.append(image)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)

if __name__ == '__main__':
    ds = XXXDataset()
    
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(ds, batch_size, num_workers=2, collate_fn=collate_fn)
    
    for batch in dataloader:
        print('{} {}'.format(batch[0].shape, batch[1][:, -2:]))