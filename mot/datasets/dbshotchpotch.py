import os
import math
import torch
import numpy as np
from collections import OrderedDict
from mot.datasets.builder import DATASETS
from mot.datasets import IterableDatasetBase
from mot.datasets.transforms import LoadImagesAndLabels

@DATASETS.register_module()
class DBSHotchpotchDataset(IterableDatasetBase):
    '''Dynamic batch size version of HotchpotchDataset.
        Designing for Stitch or Mosaic data augmentation.
    
    Param
    -----
    root  : The dataset root directory.
    cfg   : The dataset configuration file path.
    imsize: The image normalized size.
    '''
    def __init__(self, root, *args, cfg='train.txt', imsize=(320, 576), **kwargs):
        self.root = root
        self.cfg = cfg
        self.imsize = imsize
        
        # Read dataset files from configuration file.
        self.datasets = open(self.cfg, 'r').readlines()
        self.datasets = [ds.strip() for ds in self.datasets]
        self.datasets = list(filter(lambda x: len(x) > 0, self.datasets))

        # Read image paths from dataset files.
        image_paths = OrderedDict()
        label_paths = OrderedDict()
        for ds in self.datasets:
            ds_name = os.path.basename(ds)  # With suffix
            ds_name = os.path.splitext(ds_name)[0]
            with open(ds, 'r') as file:
                image_paths[ds_name] = file.readlines()
                image_paths[ds_name] = [path.strip() for path in image_paths[ds_name]]
                image_paths[ds_name] = list(filter(lambda x: len(x) > 0, image_paths[ds_name]))
                image_paths[ds_name] = [os.path.join(root, path) for path in image_paths[ds_name]]
            # Inference label paths from image paths
            label_paths[ds_name] = []
            for path in image_paths[ds_name]:
                label_path = path.replace('images', 'labels_with_ids')
                label_path = label_path.replace('.png', '.txt')
                label_path = label_path.replace('.jpg', '.txt')
                label_paths[ds_name].append(label_path)
        self.image_paths = image_paths
        self.label_paths = label_paths
        
        # Count the number of training samples for each dataset.
        self.num_ims = [len(paths) for paths in image_paths.values()]
        # Accumulate total number of training samples by each dataset.
        self.acc_ims = [sum(self.num_ims[:i]) for i in range(len(self.num_ims))]
        self.total_ims = sum(self.num_ims)
        
        # Find the number of identifiers for each dataset.
        # The label format: class identifier centerx centery width height
        self.num_ids = OrderedDict()
        for ds_name, label_paths in self.label_paths.items():
            ds_max_id = -1
            for path in label_paths:
                labels = np.loadtxt(path)
                # Empty label file.
                if len(labels) < 1:
                    continue
                # Find the maximum identifier in current label file
                if len(labels.shape) == 2:
                    file_max_id = np.max(labels[:, 1])
                else:   # Only one label in this file.
                    file_max_id = labels[1]
                if file_max_id > ds_max_id:
                    ds_max_id = file_max_id
            # The valid identifier is begin with 0.
            self.num_ids[ds_name] = ds_max_id + 1

        # Calculate identifier shift for each dataset.
        # We will calculate global identifier based on the shift.
        last_id = 0
        self.id_shifts = OrderedDict()
        for ds_name, num_id in self.num_ids.items():
            self.id_shifts[ds_name] = last_id
            last_id += num_id
        
        self._max_id = last_id - 1
        super(DBSHotchpotchDataset, self).__init__(0, self.total_ims)
    
    def __next__(self):
        batch_size = np.random.choice([1, 4]).item()
        if self.count + batch_size >= self.iter_end:
            batch_size = 1
            if self.count + batch_size >= self.iter_end:
                raise StopIteration
        
        h = int(self.imsize[0] / math.sqrt(batch_size))
        w = int(self.imsize[1] / math.sqrt(batch_size))
        if batch_size == 1:
            augmentor = LoadImagesAndLabels(img_size=(h, w), transforms=None)
        else:
            augmentor = LoadImagesAndLabels(img_size=(h, w), transforms=None, scale=(0.999, 1.001))

        images, labels = [], []
        for b in range(batch_size):
            # Get random index from shared memory.
            index = self.indices[self.count]
            
            # Transform global index to local index in dataset.
            lid = index
            ds_name = ''
            for i, acc_im in enumerate(self.acc_ims):
                if index >= acc_im:
                    ds_name = list(self.label_paths.keys())[i]
                    lid = index - acc_im
            
            if not ds_name:
                print('ERROR: index {} {}'.format(index, self.acc_ims))
            image_path = self.image_paths[ds_name][lid]
            label_path = self.label_paths[ds_name][lid]
           
            image, label, _, _ = augmentor.get_data(image_path, label_path)
            
            # Transform local identifier in dataset to global identifier.
            targets = []
            for c, i, x, y, w, h in label:
                if i > -1:
                    targets.append([0, c, i + self.id_shifts[ds_name], x, y, w, h])
                else:       # Only have bounding box annotations.
                    targets.append([0, c, i, x, y, w, h])
            
            targets = torch.as_tensor(targets, dtype=torch.float32, device=torch.device('cpu'))
            if targets.size(0) == 0:
                targets = torch.FloatTensor(0, 7) 
            
            self.count += 1
            images.append(image)
            labels.append(targets)

        # Mosaic augmentation
        if batch_size == 4:
            im1 = torch.cat(images[:2], dim=2)
            im2 = torch.cat(images[2:], dim=2)
            image = torch.cat([im1, im2], dim=1)
            # Recalculate x, y, w, h
            for l in labels:
                l[:, -4:] *= 0.5
            labels[1][:, -4] += 0.5     # top right, x
            labels[2][:, -3] += 0.5     # bottom left, y
            labels[3][:, -4:-2] += 0.5  # bottom right, x and y
            label = torch.cat(labels)
        else:
            image = images[0]
            label = labels[0]

        return image, label
    
    def __len__(self):
        return self.total_ims
    
    @property
    def max_id(self):
        return self._max_id