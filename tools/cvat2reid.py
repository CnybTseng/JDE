import os
import cv2
import sys
import argparse
import collections
import numpy as np
sys.path.append(os.getcwd())
from mot.utils import mkdirs

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CVAT mot annotations to ReID annotations')
    parser.add_argument('--video', '-v', type=str, help=
        'path to the video file')
    parser.add_argument('--label', '-l', type=str, help=
        'path to the cvat csv file')
    parser.add_argument('--minw', type=int, default=0,
        help='minimum width')
    parser.add_argument('--minh', type=int, default=0,
        help='minimum height')
    parser.add_argument('--start-id', type=int, default=0,
        help='the start id for this scene')
    parser.add_argument('--camid', type=int, default=0,
        help='camera index')
    parser.add_argument('--seq', type=int, default=0,
        help='video sequence number for the camera')
    parser.add_argument('--size', type=int, nargs='+', default=[64, 128],
        help='normalized clip size')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    vc = cv2.VideoCapture(args.video)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print('width {}, height {}, possible frames {}'.format(width, height, frames))
    
    root, _ = os.path.splitext(args.video)
    cache_path = os.path.join(root, 'frames')
    if not os.path.exists(cache_path):
        mkdirs(cache_path)
        i = 0
        while True:
            retval, im = vc.read()
            if not retval:
                break
            cv2.imwrite(os.path.join(cache_path, '{}.jpg'.format(i)), im)
            i = i + 1
            print('\rextract frame {}/{}'.format(i, frames), end='', flush=True)
    
    print('')
    reid_path = os.path.join(root, 'reid')
    mkdirs(reid_path)

    # cvat has a terrible bug.
    # We have to remove repeated items (in unlabeled frames) manully!
    # Unfortunately, static object will be removed too.
    labels = open(args.label, 'r').readlines()
    labels = [label.strip() for label in labels]
    labels = list(filter(lambda x: len(x) > 0, labels))
    
    # Gather labels belong to the same frame.
    gather_labels = collections.defaultdict(list)
    for label in labels:
        frame, others = label.split(',', maxsplit=1)
        gather_labels[frame].append(others)
    
    no_copy_gather_labels = collections.defaultdict(list)
    for key, value in gather_labels.items():
        if value not in no_copy_gather_labels.values():
            no_copy_gather_labels[key] = value
        else:
            print('find repeate labels: {}'.format(value))

    for frame, values in no_copy_gather_labels.items():
        frame = int(frame)
        infile = os.path.join(cache_path, '{}.jpg'.format(frame))
        im = cv2.imread(infile)
        if im is None:
            print('imread {} failed'.format(infile))
            continue
        
        # All labels for current frame.
        for value in values:    
            strs = value.split(',')
            id, *ltwh, conf, cate, visi = strs                
            l, t, w, h = [int(float(i) + 0.5) for i in ltwh]
            if w < args.minw or h < args.minh:
                continue            
            
            l = np.clip(l, 0, im.shape[1] - 1)
            t = np.clip(t, 0, im.shape[0] - 1)
            w = np.clip(w, 1, im.shape[1] - 1 - l)
            h = np.clip(h, 1, im.shape[0] - 1 - t)
            clip = im[t : t + h, l : l + w, :]
            clip = cv2.resize(clip, tuple(args.size), 0, 0, cv2.INTER_AREA)
            id = int(id)            
            gid = args.start_id + id
            class_path = os.path.join(reid_path, '%04d' % gid)
            mkdirs(class_path)
            name = '%04d_c%ds%d_%06d_%02d' % (gid, args.camid, args.seq, frame, 0)
            outfile = os.path.join(class_path, name + '.jpg')
            cv2.imwrite(outfile, clip)
    print('done')