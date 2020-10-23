# -*- coding: utf-8 -*-
# file: transforms.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/12

import cv2
import torch
import random
import numpy as np
import torch.nn.functional as tnf
from torchvision.transforms import functional as ttf

class RandomSpatialJitter(object):
    def __init__(self, jitter=0.3, net_w=576, net_h=320):
        '''随机的空间扰动.输入输出图像的值域均为[0,255].类型不变.
        
        参数
        ----
        jitter : float
            图像尺寸的变化幅度.以相对原始尺寸的比例表示.
        net_w : int
            神经网络的输入宽度.
        net_h : int
            神经网络的输入高度.
        '''
        self.jitter = jitter
        self.net_w = net_w
        self.net_h = net_h
        self.cuda = False
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
    
    def __call__(self, image, target):
        dw = int(image.size(3) * self.jitter)
        dh = int(image.size(2) * self.jitter)

        rdw = random.randint(-dw, dw)
        rdh = random.randint(-dh, dh)
        ar = float(image.size(3) + rdw) / float(image.size(2) + rdh)

        if ar < self.net_w / self.net_h:
            nh = self.net_h
            nw = int(ar * nh)
        else:
            nw = self.net_w
            nh = int(nw / ar)
        
        dx = random.randint(0, self.net_w - nw)
        dy = random.randint(0, self.net_h - nh)
        
        jit_image = image.new_zeros(image.size(0), image.size(1), self.net_h, self.net_w)
        jit_image.fill_(127)
        
        jit_image[:,:,dy:dy+nh,dx:dx+nw] = tnf.interpolate(\
            input=image.type(self.FloatTensor), size=(nh, nw),
            mode='bilinear', align_corners=True).type_as(image)
        
        if target.numel() > 0:
            target[:,3] = (target[:,3] * nw + dx) / self.net_w
            target[:,4] = (target[:,4] * nh + dy) / self.net_h
            target[:,5] =  target[:,5] * nw / self.net_w
            target[:,6] =  target[:,6] * nh / self.net_h
        
        return jit_image, target

class RandomColorJitter(object):
    def __init__(self, hue=0.1, saturation=1.5, exposure=1.5):
        '''随机的颜色扰动.原始的RGB图像将变换至HSV空间,经颜色扰动后,变换回RGB空间.
           输入输出图像的值域为[0,255].类型不变.
        
        参数
        ----
        hue : float
            色度的扰动幅度.
        saturation : float
            饱和度的扰动幅度.
        exposure : float
            曝光度的扰动幅度.
        '''
        
        self.hue = hue
        self.sat = saturation
        self.exp = exposure
        self.cuda = False
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
    
    def __call__(self, image, target, jitter=True):
        hsv = self.__rgb2hsv(image)

        if not jitter: 
            return hsv, self.__hsv2rgb(hsv)
        
        jhue = random.uniform(-self.hue, self.hue)
        
        jsat = random.uniform(1, self.sat)
        if random.random() > 0.5:
            jsat = 1 / jsat
        
        jexp = random.uniform(1, self.exp)
        if random.random() > 0.5:
            jexp = 1 / jexp
        
        hsv[:,0,:,:] += jhue
        hsv[:,1,:,:] *= jsat
        hsv[:,2,:,:] *= jexp
        
        hsv[:,0,:,:][hsv[:,0,:,:] < 0] += 1
        hsv[:,0,:,:][hsv[:,0,:,:] > 1] -= 1
        
        hsv[hsv < 0] = 0
        hsv[hsv > 1] = 1
                
        rgb = (self.__hsv2rgb(hsv) * 255).type_as(image)

        return rgb, target
    
    def __rgb2hsv(self, image, eps=1e-8):
        image = image.type(self.LongTensor)     # 确保图像是有符号整型
        R = image[:,0,:,:]
        G = image[:,1,:,:]
        B = image[:,2,:,:]

        min, _ = image.min(dim=1)
        max, _ = image.max(dim=1)
        delta = (max - min).type(self.FloatTensor)

        S = delta / (max.type(self.FloatTensor) + eps)
        S[max==0] = 0
        V = max.type(self.FloatTensor) / 255.0
        H = torch.zeros_like(S)

        mask = (max == R)
        H[mask] = (G[mask] - B[mask]).type(self.FloatTensor) / (delta[mask] + eps)

        mask = (max == G)
        H[mask] = (B[mask] - R[mask]).type(self.FloatTensor) / (delta[mask] + eps) + 2
        
        mask = (max == B)
        H[mask] = (R[mask] - G[mask]).type(self.FloatTensor) / (delta[mask] + eps) + 4

        H[max == min] = 0   # 此情况可能包含在上述几种情况中,必须放在最后防止被覆盖
        mask = (H < 0)
        H[mask] = H[mask] + 6
        H = H / 6

        return torch.stack([H,S,V], dim=1)
    
    def __hsv2rgb(self, image):
        H = image[:,0,:,:] * 6
        S = image[:,1,:,:]
        V = image[:,2,:,:]
        
        R = torch.zeros_like(H)
        G = torch.zeros_like(H)
        B = torch.zeros_like(H)
        
        index = H.floor()
        F = H - index
        P = V * (1 - S)
        Q = V * (1 - S * F)
        T = V * (1 - S * (1 - F))
        
        index = index.type(self.LongTensor)
        
        mask = (index == 0)
        R[mask] = V[mask]
        G[mask] = T[mask]
        B[mask] = P[mask]
        
        mask = (index == 1)
        R[mask] = Q[mask]
        G[mask] = V[mask]
        B[mask] = P[mask]
        
        mask = (index == 2)
        R[mask] = P[mask]
        G[mask] = V[mask]
        B[mask] = T[mask]
        
        mask = (index == 3)
        R[mask] = P[mask]
        G[mask] = Q[mask]
        B[mask] = V[mask]
        
        mask = (index == 4)
        R[mask] = T[mask]
        G[mask] = P[mask]
        B[mask] = V[mask]
        
        mask = (index == 5)
        R[mask] = V[mask]
        G[mask] = P[mask]
        B[mask] = Q[mask]
        
        return torch.stack([R,G,B], dim=1)

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        '''随机的水平翻转.
        
        参数
        ----
        prob : float
            水平翻转的概率.
        '''
        
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-1)
            if target.numel() > 0:
                target[:,3] = 1 - target[:,3]

        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0).permute(0, 3, 1, 2).contiguous()
        return image, target

class MakeLetterBoxImage(object):
    def __init__(self, width=576, height=320):
        self.width = width
        self.height = height
        self.cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
    
    def __call__(self, image, target):
        ar = image.size(3) / image.size(2)
        if ar < self.width / self.height:
            nh = self.height
            nw = int(ar * nh)
        else:
            nw = self.width
            nh = int(nw / ar)

        dx = int((self.width - nw) / 2)
        dy = int((self.height - nh) / 2)

        lb_image = self.FloatTensor(image.size(0), image.size(1), self.height, self.width)
        lb_image.fill_(127)
        
        lb_image[:,:,dy:dy+nh,dx:dx+nw] = tnf.interpolate(\
            input=image.type(self.FloatTensor), size=(nh,nw),
            mode='bilinear', align_corners=True).type_as(image)
        
        if target is not None and target.numel() > 0:
            target[:,3] = (target[:,3] * nw + dx) / self.width
            target[:,4] = (target[:,4] * nh + dy) / self.height
            target[:,5] =  target[:,5] * nw / self.width
            target[:,6] =  target[:,6] * nh / self.height
        
        return lb_image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomAffine(object):
    def __init__(self, degrees=(-10,10), translate=(.1,.1),
        scale=(.9,1.1), shear=(-2,2), fillcolor=(127,127,127)):
        '''随机的仿射变换.
        
        参数
        ----
        degrees  : 旋转角度采样范围.
        translate: 平移量采样范围. 该参数为图像宽度和高度的百分比.
        scale    : 缩放系数采样范围.
        shear    : 剪切量采样范围.
        fillcolor: 无定义区域填充色.
        '''
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fillcolor = fillcolor
    
    def __call__(self, image, targets=None):
        '''随机的仿射变换.
        
        参数
        ----
        image  : CV_8UC3类型图像
        targets: 标签
        '''
        
        # 旋转矩阵
        center = (image.shape[1] / 2, image.shape[0] / 2)
        angle = np.random.uniform(low=self.degrees[0], high=self.degrees[1])
        scale = np.random.uniform(low=self.scale[0], high=self.scale[1])        
        
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 平移矩阵
        tx = self.translate[0] * image.shape[1]
        ty = self.translate[1] * image.shape[0]
        
        T = np.eye(3)
        T[0, 2] = np.random.uniform(low=-tx, high=tx)
        T[1, 2] = np.random.uniform(low=-ty, high=ty)
        
        # 剪切矩阵
        shx = np.random.uniform(low=self.shear[0], high=self.shear[1])
        shy = np.random.uniform(low=self.shear[0], high=self.shear[1])
        
        S = np.eye(3)
        S[0, 1] = np.tan(shx * np.pi / 180)
        S[1, 0] = np.tan(shy * np.pi / 180)
        
        # 合成仿射变换矩阵
        M = S @ T @ R
        
        # 对图像进行仿射变换
        dst = cv2.warpPerspective(image, M, dsize=(image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=self.fillcolor)
        
        if targets is None:
            return dst
        
        if len(targets) == 0:
            return dst, targets
        
        # 对标签进行仿射变换
        
        
        return dst

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str,
        help='path to the image')
    args = parser.parse_args()
    
    T = RandomAffine()
    image = cv2.imread(args.image)
    dst = T(image)
    cv2.imwrite("T.png", dst)