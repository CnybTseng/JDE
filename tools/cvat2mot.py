import os
import cv2
import argparse
import collections
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str, help=
        'path to the video file')
    parser.add_argument('--label', '-l', type=str, help=
        'path to the cvat csv file')
    parser.add_argument('--minw', type=int, default=0,
        help='minimum width')
    parser.add_argument('--minh', type=int, default=0,
        help='minimum height')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    vc = cv2.VideoCapture(args.video)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print('width {}, height {}, frames {}'.format(width, height, frames))
    
    if not os.path.exists('frames'):
        os.mkdir('frames')
        i = 0
        while True:
            retval, im = vc.read()
            if not retval:
                break
            cv2.imwrite(os.path.join('frames', '{}.jpg'.format(i)), im)
            i = i + 1
            print('\rextract frame {}/{}'.format(i, frames), end='', flush=True)
    
    print('')
    root, _ = os.path.split(args.video)
    im_path = os.path.join(root, 'images')
    if not os.path.exists(im_path):
        os.mkdir(im_path)
    
    lb_path = os.path.join(root, 'labels_with_ids')
    if not os.path.exists(lb_path):
        os.mkdir(lb_path)

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
        outfile = os.path.join(lb_path, '%06d.txt' % frame)
        with open(outfile, 'w') as file:
            for value in values:    
                strs = value.split(',')
                id, *ltwh, conf, cate, visi = strs                
                l, t, w, h = [float(i) for i in ltwh]
                if w < args.minw or h < args.minh:
                    continue
                
                id = int(id)
                x, y = (l + w / 2.0) / width, (t + h / 2.0) / height
                w, h = w / width, h / height
                
                file.write('{} {} {} {} {} {}\n'.format(
                    0, id, x, y, w, h))
        
        infile = os.path.join('frames', '{}.jpg'.format(frame))
        outfile = os.path.join(im_path, '%06d.jpg' % frame)
        # Fuck the OpenCV for unreliable frame location!
        # vc.set(cv2.CAP_PROP_POS_FRAMES, frame)
        # retval, image = vc.read()
        # if retval:
        #     cv2.imwrite(outfile, image)
        # else:
        #     print('read frame {} fail\n'.format(frame))
        cmd = 'cp {} {}'.format(infile, outfile)
        print('\r{}'.format(cmd), end='', flush=True)
        os.system(cmd)
    print('')