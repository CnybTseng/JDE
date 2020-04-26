import os
import cv2
import glob
import numpy as np

import sys
import torch
import utils
import transforms as T
from pascalvoc import PascalVocReader as pvr

def letterbox_image(im, insize=(320,576,3), border=128):
    '''生成letterbox图像
    
    Args:
        im (ndarray): RGB/BGR格式图像
        insize (tuple, optional): 神经网络输入大小, insize=(height, width,
            channels)
        border (float, optional): 图像边沿填充颜色
    Returns:
        lb_im (ndarray): letterbox图像
        s (float): 图像缩放系数
        dx (int): 非填充区域的水平偏移量
        dy (int): 非填充区域的垂直偏移量
    '''
    h, w = im.shape[:2]
    s = min(insize[0] / h, insize[1] / w)
    nh = round(s * h)
    nw = round(s * w)
    dx = (insize[1] - nw) / 2
    dy = (insize[0] - nh) / 2
    left  = round(dx - 0.1)
    right = round(dx + 0.1)
    above = round(dy - 0.1)
    below = round(dy + 0.1)
    lb_im = np.full(insize, border, dtype=np.uint8)
    lb_im[above:above+nh, left:left+nw, :] = cv2.resize(im, (nw,nh), interpolation=
        cv2.INTER_AREA)
    return lb_im, s, dx, dy

class ImagesLoader(object):
    '''图像迭代器
    
    Args:
        path (str): 图像路径
        insize (tuple): 神经网络输入大小, insize=(height, width)
        formats (list of str): 需要解码的图像格式列表
    '''
    def __init__(self, path, insize, formats=['*.jpg']):
        if os.path.isdir(path):
            self.files = []
            for format in formats:
                self.files += sorted(glob.glob(os.path.join(path, format)))
        elif os.path.isfile(path):
            self.files = [path]
        self.insize = insize
        self.count = 0

    def __iter__(self):
        self.count = -1
        return self
    
    def __next__(self):
        self.count += 1
        if self.count == len(self.files):
            raise StopIteration
        path = self.files[self.count]
        im = cv2.imread(path)
        assert im is not None, 'cv2.imread{} fail'.format(path)
        lb_im, s, dx, dy = letterbox_image(im, insize=self.insize)
        lb_im = lb_im[...,::-1].transpose(2, 0, 1)
        lb_im = np.ascontiguousarray(lb_im, dtype=np.float32)
        lb_im /= 255.0
        return path, im, lb_im

def get_transform(train, net_w=416, net_h=416):
    transforms = []
    transforms.append(T.ToTensor())
    if train == True:
        transforms.append(T.RandomSpatialJitter(jitter=0.3,net_w=net_w,net_h=net_h))
        transforms.append(T.RandomColorJitter(hue=0.1,saturation=1.5,exposure=1.5))
        transforms.append(T.RandomHorizontalFlip(prob=0.5))
    else:
        transforms.append(T.MakeLetterBoxImage(width=net_w,height=net_h))
    return T.Compose(transforms)

def collate_fn(batch, in_size=torch.IntTensor([416,416]), train=False):
    transforms = get_transform(train, in_size[0].item(), in_size[1].item())
    images, targets = [], []
    for i,b in enumerate(batch):
        image, target = transforms(b[0], b[1])
        image = image.type(torch.FloatTensor) / 255
        target[:,0] = i
        images.append(image)
        targets.append(target)
    return torch.cat(tensors=images, dim=0), torch.cat(tensors=targets, dim=0)

class CustomDataset(object):
    def __init__(self, root, file='train'):
        self.root = root
        path = open(os.path.join(root, f'{file}.txt')).read().split()
        self.images_path = path[0::2]
        self.annocations_path = path[1::2]
        self.class_names = self.__load_class_names(os.path.join(root, 'classes.txt'))

    def __getitem__(self, index):
        image_path = self.images_path[index]
        annocation_path = self.annocations_path[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        org_size = image.shape[:2]
        annocation = pvr(annocation_path).getShapes()
        
        target = []
        for an in annocation:
            name = an[0]
            xmin = an[1][0][0]
            ymin = an[1][0][1]
            xmax = an[1][1][0]
            ymax = an[1][2][1]
            self.__assert_bbox(xmin, ymin, xmax, ymax, org_size, image_path)
            bx, by, bw, bh = self.__xyxy_to_xywh(xmin, ymin, xmax, ymax, org_size)
            target.append([0, self.class_names.index(name), bx, by, bw, bh])

        target = torch.as_tensor(target, dtype=torch.float32, device=torch.device('cpu'))
        if target.size(0) == 0:
            target = torch.FloatTensor(0, 6)
        return image, target
    
    def __len__(self):
        return len(self.images_path)
        
    def __load_class_names(self, path):
        class_names = []
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                name = line.rstrip('\n')
                if not name: continue
                class_names.append(name)
            file.close()
        return class_names
    
    def __xyxy_to_xywh(self, xmin, ymin, xmax, ymax, size):
        bx = (xmin + xmax) / 2 / size[1]
        by = (ymin + ymax) / 2 / size[0]
        bw = (xmax - xmin + 1) / size[1]
        bh = (ymax - ymin + 1) / size[0]
        return bx, by, bw, bh
    
    def __assert_bbox(self, xmin, ymin, xmax, ymax, size, filename):
        if xmin < 0 or xmin > size[1]:
            print(f'BAD BOUNDING BOX! xmin={xmin}, size={size}, {filename}')
            sys.exit()
        if xmin > 0: xmin -= 1
        if ymin < 0 or ymin > size[0]:
            print(f'BAD BOUNDING BOX! ymin={ymin}, size={size}, {filename}')
            sys.exit()
        if ymin > 0: ymin -= 1
        if xmax < 0 or xmax > size[1]:
            print(f'BAD BOUNDING BOX! xmax={xmax}, size={size}, {filename}')
            sys.exit()
        xmax -= 1
        if ymax < 0 or ymax > size[0]:
            print(f'BAD BOUNDING BOX! ymax={ymax}, size={size}, {filename}')
            sys.exit()
        ymax -= 1