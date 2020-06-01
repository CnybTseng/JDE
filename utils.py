# -*- coding: utf-8 -*-
# file: utils.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/12

import os
import re
import cv2
import copy
import torch
import logging
import numpy as np

def get_logger(name='root', path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(path, encoding='utf-8')
    formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class TrainScaleSampler(object):
    '''Multiple scales training scale sampler.
    
    Args:
        size: Default sampling size. The size is [height, width].
        scale_step: Image scale step. The scale_step is [ystart, ystop,
            num[, xstart, xstop]]. The ystart is the starting value of the
            image height sequence, and the ystop is the end value of the
            image height sequence. The num is the number of samples to
            generate. The xstart and xstop are optional, if neither of
            them is seted, xstart and xstop will be equal to ystart and
            ystop respectively.
        rescale_freq: Image rescale frequency. The image will be rescaled
            every #rescale_freq batches.
    '''
    def __init__(self, size=(416,416), scale_step=(320,608,10),
        rescale_freq=320):
        self.scale_step = scale_step
        self.rescale_freq = rescale_freq
        self.size = size
    
    def __call__(self, num_batches=0):
        if num_batches % self.rescale_freq == 0:
            heights = np.linspace(self.scale_step[0], self.scale_step[1],
                num=self.scale_step[2], dtype=np.int32)
            if len(scale_step == 5):
                widths = np.linspace(self.scale_step[3], self.scale_step[4],
                    num=self.scale_step[2], dtype=np.int32)
            else:
                widths = heights
            rn = np.random.randint(len(heights))
            self.size = [heights[rn].item(), widths[rn],item()]
        return self.size

def make_workspace_dirs(workspace='./workspace'):
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    if not os.path.exists(os.path.join(workspace, 'checkpoint')):
        os.makedirs(os.path.join(workspace, 'checkpoint'))     

def lr_lambda(iter, warmup, milestones, lr_gamma):
    if iter < warmup:
        return pow(iter / warmup, 4)
    factor = 1
    for i in milestones:
        factor *= pow(lr_gamma, int(iter > i))
    return factor

def load_class_names(path):
    class_names = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            name = line.rstrip('\n')
            if not name: continue
            class_names.append(name)
        file.close()
    return class_names

def xywh_to_xyxy(box):
    xmin = box[0] - box[2]/2
    ymin = box[1] - box[3]/2
    xmax = box[0] + box[2]/2
    ymax = box[1] + box[3]/2
    
    return xmin, ymin, xmax, ymax

def get_network_boxes(dets, im_size, thresh=0.5):
    dets = dets.squeeze()
    dets = dets[dets[:,4]>thresh,:]
    dets[:,5:] = dets[:,4].view(dets[:,4].size(0), 1) * dets[:,5:]
    dets[:,5:][dets[:,5:]<=thresh] = 0
    long_side = max(im_size)
    for det in dets:
        x, y, w, h = det[:4] * long_side
        if im_size[0] == long_side:
            x = x - (long_side - im_size[1])/2
        else:
            y = y - (long_side - im_size[0])/2
        det[0] = x - w/2
        det[1] = y - h/2
        det[2] = x + w/2
        det[3] = y + h/2
    return dets.numpy()

def box_iou(box1, box2, eps=1e-16):
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])    
    inter_w = max(0, xmax - xmin)
    inter_h = max(0, ymax - ymin)
    inter_a = inter_w * inter_h    
    box1_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_a = (box2[2] - box2[0]) * (box2[3] - box2[1])    
    return inter_a / (box1_a + box2_a - inter_a + eps)

def do_nms_sort(dets, ovthresh=0.45):
    num_dets = dets.shape[0]
    num_classes = dets[:, 5:].shape[1]
    for c in range(num_classes):
        sorted_indices = np.argsort(dets[:, 5+c])[::-1]
        dets = dets[sorted_indices, :]
        for i in range(num_dets):
            if dets[i, 5+c] == 0:
                continue
            for j in range(i+1, num_dets):
                iou = box_iou(dets[i, :4], dets[j, :4])
                if iou > ovthresh:
                    dets[j, 5+c] = 0
    return dets

def overlap_detection(im, dets, class_names):    
    for det in dets:                       
        if det[4] == 0 or np.max(det[5:]) == 0:
            continue

        # 获取字符串尺寸
        id = np.argmax(det[5:])
        [size, baseline] = cv2.getTextSize(text=class_names[id], fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, thickness=1)
        
        # 字符串水平位置限界
        minx, miny, maxx, maxy = det[:4].astype(np.int32)
        tw, th = size[0], size[1]
        tx = minx if minx >= 0 else 0
        tx = tx if tx < im.shape[1]-tw else im.shape[1]-tw-1
        
        # 字符串垂直位置限界.
        # 字符串的高度等于getTextSize返回的高度加上baseline
        ty = miny-baseline if miny-baseline >= th else th 
        ty = ty if ty < im.shape[0]-baseline else im.shape[0]-baseline-1
        
        # 叠加类别标签
        cv2.rectangle(img=im, pt1=(tx,ty-th), pt2=(tx+tw-1,ty+baseline), color=(0,255,255), thickness=-1)
        cv2.putText(img=im, text=class_names[id], org=(tx,ty), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,255), thickness=1)
        cv2.rectangle(img=im, pt1=(minx,miny), pt2=(maxx,maxy), color=(0,255,255), thickness=1)

    return im

def save_detection_result(filename, im_size, dets, classnames):
    num_classes = dets[:,5:].shape[1]
    segments = re.split(r'[\\,/]', filename)
    filename = segments[-1].split('.')[0]
    for det in dets:
        if det[4] == 0 or np.max(det[5:]) == 0:
            continue
        classid = np.argmax(det[5:])
        classname = classnames[classid]
        minx, miny, maxx, maxy = det[:4] + 1
        minx = max(1, minx)
        miny = max(1, miny)
        maxx = min(im_size[1], maxx)
        maxy = min(im_size[0], maxy)
        for c in range(num_classes):
            if det[5+c] > 0:
                with open(f"results/comp4_det_test_{classname}.txt", "a") as file:
                    file.write(f"{filename} {det[5+c]} {minx} {miny} {maxx} {maxy}\n")