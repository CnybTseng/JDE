import os
import re
import cv2
import torch
import argparse
import numpy as np
from torchvision.ops import nms

import yolov3
import darknet
import dataset

def parse_args():
    '''解析命令行参数
    '''
    parser = argparse.ArgumentParser(
        description='single class multiple object tracking')
    parser.add_argument('--img-path', type=str, help='path to image path')
    parser.add_argument('--model', type=str, help='path to tracking model')
    parser.add_argument('--insize', type=str, default='320x576',
        help='network input size, default=320x576')
    return parser.parse_args()

def xywh2tlbr(boxes):
    '''转换建议框的数据格式
    Args:
        boxes (torch.Tensor): [x,y,w,h]格式的建议框
    Returns:
        tlbr (torch.Tensor): [t,l,b,r]格式的建议框
    '''
    tlbr = torch.zeros_like(boxes)
    tlbr[:,0] = boxes[:,0] - boxes[:,2]/2
    tlbr[:,1] = boxes[:,1] - boxes[:,3]/2
    tlbr[:,2] = boxes[:,0] + boxes[:,2]/2
    tlbr[:,3] = boxes[:,1] + boxes[:,3]/2
    return tlbr

def nonmax_suppression(dets, score_thresh=0.5, iou_thresh=0.4):
    '''检测器输出的非最大值抑制
    
    Args:
        dets (torch.Tensor): 检测器输出, dets.size()=[batch_size,
            #proposals, #dim], 其中#proposals是所有尺度输出的建议
            框数量, #dim是每个建议框的属性维度
        score_thresh (float): 置信度阈值, score_thresh∈[0,1]
        iou_thresh (float): 重叠建议框的叫并面积比阈值, iou_thresh∈[0,1]
    Returns:
        nms_dets (torch.Tensor): 经NMS的检测器输出
    '''
    nms_dets = [None for _ in range(dets.size(0))]
    for i, det in enumerate(dets):
        keep = det[:,4] > score_thresh
        det = det[keep]
        if not det.size(0):
            continue
        det[:, :4] = xywh2tlbr(det[:, :4])
        keep = nms(det[:, :4], det[:, 4], iou_thresh)
        det = det[keep]
        nms_dets[i] = det
    return nms_dets

def tlbr_net2img(boxes, net_size, img_size):
    '''将神经网络坐标系下的建议框投影到图像坐标系下
    
    Args:
        boxes (torch.Tensor): 神经网络坐标系下[t, l, b, r]格式的建议框
        net_size (tuple of int): 神经网络输入大小, net_size=(height, width)
        img_size (tuple of int): 图像大小, img_size=(height, width)
    Returns:
        boxes (torch.Tensor): 图像坐标系下[t, l, b, r]格式的建议框
    '''
    net_size = np.array(net_size)
    img_size = np.array(img_size)
    s = (img_size / net_size).max()
    nnet_size = (s * net_size).round()
    dy, dx = (nnet_size - img_size) // 2
    boxes = (s * boxes).round()
    boxes[:, [0,2]] -= dy
    boxes[:, [1,3]] -= dx
    return boxes

def overlap(dets, im):
    '''叠加检测结果到图像上
    Args:
        dets (torch.Tensor): 含检测结果的二维数组, dets[:]=
            [t,l,b,r,objecness,class,embedding]
        im (numpy.ndarray): BGR格式图像
    Returns:
        im (numpy,ndarray): BGR格式图像
    '''
    for det in dets:
        t, l, b, r = det[:4]
        color = np.random.randint(0, 256, size=(3,)).tolist()
        cv2.rectangle(im, (t, l), (b, r), color, 2)
    return im

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = darknet.DarkNet().to(device)
    model.load_state_dict(torch.load(os.path.join(args.model), map_location='cpu'))
    model.eval()

    if '320x576' in args.insize:
        anchors = ((6,16),   (8,23),    (11,32),   (16,45),
                   (21,64),  (30,90),   (43,128),  (60,180),
                   (85,255), (120,360), (170,420), (340,320))
    elif '480x864' in args.insize:
        anchors = ((6,19),    (9,27),    (13,38),   (18,54),
                   (25,76),   (36,107),  (51,152),  (71,215),
                   (102,305), (143,429), (203,508), (407,508))
    elif '608x1088' in args.insize:
        anchors = ((8,24),    (11,34),   (16,48),   (23,68),
                   (32,96),   (45,135),  (64,192),  (90,271),
                   (128,384), (180,540), (256,640), (512,640))

    h, w = [int(s) for s in args.insize.split('x')]
    decoder = yolov3.YOLOv3EvalDecoder((h,w), 1, anchors)
    dataloader = dataset.ImagesLoader(args.img_path, (h,w,3))
    for path, im, lb_im in dataloader:
        input = torch.from_numpy(lb_im).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input)
        outputs = decoder(outputs)
        print('{} {} {} {}'.format(path, im.shape, lb_im.shape, outputs.size()), end=' ')
        outputs =  nonmax_suppression(outputs)[0]
        print('{}'.format(outputs.size()))
        outputs[:, :4] = tlbr_net2img(outputs[:, :4], (h,w), im.shape[:2])
        result = overlap(outputs, im)
        segments = re.split(r'[\\, /]', path)
        cv2.imwrite('result/{}'.format(segments[-1]), result)

if __name__ == '__main__':
    args = parse_args()
    main(args)