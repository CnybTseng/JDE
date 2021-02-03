import math
import numpy as np
from abc import abstractmethod
from multiprocessing import Array
from torch.utils.data import IterableDataset, get_worker_info

class IterableDatasetBase(IterableDataset):
    '''Iterable dataset for dynamic size batch loading.
    
    Param
    -----
    start: The start data index of your dataset.
    end  : The last data index (excluding) of your dataset.
    '''
    def __init__(self, start, end):
        super(IterableDatasetBase, self).__init__()
        self.start = start
        self.end = end
        # Shared between multi-process
        self.indices = Array('i', range(start, end))
   
    def __iter__(self):
        worker_info = get_worker_info()
        self.worker_id = worker_info.id
        if worker_info is None: # For single process data loading
            self.iter_start = self.start
            self.iter_end = self.end
        else:                   # For multi-process data loading            
            per_worker = (self.end - self.start + \
                worker_info.num_workers - 1) // worker_info.num_workers
            self.iter_start = self.start + self.worker_id * per_worker
            self.iter_end = min(self.iter_start + per_worker, self.end)
        self.count = self.iter_start
        # Only shuffle once for multi-process
        if self.iter_start == self.start:
            np.random.shuffle(self.indices)
        return self
    
    @abstractmethod
    def __next__(self):
        pass