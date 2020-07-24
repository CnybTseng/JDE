import sys
sys.path.append('.')

import os
import re
import cv2
import glob
import torch
import argparse
import collections
import numpy as np

import yolov3
import darknet
import dataset

def parse_args():
    parser = argparse.ArgumentParser(description=
        'make atlas 200 dk test data')
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--insize', type=int, nargs='+',
        default=[320,576], help='network input size')
    parser.add_argument('--img-path', type=str,
        help='path to test images')
    parser.add_argument('--save-path', type=str,
        default='./platforms/atlas200dk/run/out/test/data/',
        help='test data saving path')
    args = parser.parse_args()
    return args

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
        lb_im, s, dx, dy = dataset.letterbox_image(im, insize=self.insize)
        yuv = cv2.cvtColor(lb_im, cv2.COLOR_BGR2YUV)
        lb_im = lb_im[...,::-1].transpose(2, 0, 1)
        lb_im = np.ascontiguousarray(lb_im, dtype=np.float32)
        lb_im /= 255.0
        return path, im, lb_im, yuv

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dummy_anchors = np.random.randint(0, 100, (12, 2))
    model = darknet.DarkNet(dummy_anchors).to(device)
    
    model_dict = model.state_dict()
    trained_model_dict = torch.load(args.model, map_location='cpu')
    trained_model_dict = {k : v for (k, v) in trained_model_dict.items() if k in model_dict}
    trained_model_dict = collections.OrderedDict(trained_model_dict)
    model_dict.update(trained_model_dict)
    model.load_state_dict(model_dict)
    
    model.eval()
    
    h, w = args.insize
    dataloader = ImagesLoader(args.img_path, (h,w,3), formats=['*.jpg', '*.png'])
    
    for i, (path, im, lb_im, yuv) in enumerate(dataloader):
        input = torch.from_numpy(lb_im).unsqueeze(0).to(device)
        with torch.no_grad():
            segments = re.split(r'[\\, /]', path)
            backbone = segments[-1].split('.')[0]
            infile = os.path.join(args.save_path, '{}.bin'.format(backbone))
            with open(infile, 'wb') as file:
                lb_im.tofile(infile)
                file.close()
            
            outputs = model(input)            
            for output in outputs:
                name = '{}-{}x{}.bin'.format(backbone, output.size(2), output.size(3))
                outfile = os.path.join(args.save_path, name)
                with open(outfile, 'wb') as file:
                    output.data.cpu().numpy().tofile(file)
                    file.close()