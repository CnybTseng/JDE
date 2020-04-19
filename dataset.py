import os
import cv2
import glob
import numpy as np

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