import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help=
        'path to the video file')
    parser.add_argument('--label', type=str, help=
        'path to the cvat csv file')
    parser.add_argument('--minw', type=int, default=32,
        help='minimum width')
    parser.add_argument('--minh', type=int, default=32,
        help='minimum height')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    width = 1920
    height = 1080
    frames = 0
    vc = cv2.VideoCapture(args.video)

    width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print('width {}, height {} frames {}'.format(width, height, frames))
    
    if not os.path.exists('./frames'):
        os.mkdir('./frames')
        num_decode = 0
        while True:
            retval, image = vc.read()
            if not retval:
                break
            outfile = os.path.join('./frames', '%06d.jpg' % num_decode)
            cv2.imwrite(outfile, image)
            num_decode += 1
    
    if not os.path.exists('./img1'):
        os.mkdir('./img1')
    
    if not os.path.exists('./labels_with_ids'):
        os.mkdir('./labels_with_ids')
    
    cnt = 0
    labels = open(args.label, 'r').read().split()
    
    id_dict = dict()
    for label in labels:
        label = label.strip()
        if not label:
            continue
        
        strs = label.split(',')
        frame, id, *ltwh, conf, cate, visi = strs
        
        l, t, w, h = [float(i) for i in ltwh]
        if w < args.minw or h < args.minh:
            continue
        
        frame = int(frame)
        if not frame in id_dict.keys():
            id_dict[frame] = cnt
            cnt += 1
    
    for label in labels:
        label = label.strip()
        if not label:
            continue
        
        strs = label.split(',')
        frame, id, *ltwh, conf, cate, visi = strs
        
        l, t, w, h = [float(i) for i in ltwh]
        if w < args.minw or h < args.minh:
            continue
        
        print('\rdeal {}/{} {}'.format(frame, frames, id), end='', flush=True)
        frame = int(frame)
        id = int(id)
        x, y = (l + w / 2.0) / width, (t + h / 2.0) / height
        w, h = w / width, h / height
        
        outfile = os.path.join('./labels_with_ids', '%06d.txt' % id_dict[frame])
        file = open(outfile, 'a')
        file.write('{} {} {} {} {} {}\n'.format(
            0, id, x, y, w, h))
        file.close()
        
        infile = os.path.join('./frames', '%06d.jpg' % frame)
        outfile = os.path.join('./img1', '%06d.jpg' % id_dict[frame])
        if not os.path.exists(outfile):
            image = cv2.imread(infile)
            cv2.imwrite(outfile, image)