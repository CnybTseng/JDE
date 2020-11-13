import os
import cv2
import argparse
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
    
    root, _ = os.path.split(args.video)
    im_path = os.path.join(root, 'images')
    if not os.path.exists(im_path):
        os.mkdir(im_path)
    
    lb_path = os.path.join(root, 'labels_with_ids')
    if not os.path.exists(lb_path):
        os.mkdir(lb_path)

    # cvat has a terrible bug.
    # We have to remove repeated items (in unlabeled frames) manully!
    # Unfortunately, static object will also be removed.
    labels = open(args.label, 'r').readlines()
    labels = [label.strip() for label in labels]
    labels = list(filter(lambda x: len(x) > 0, labels))
    nocopy_labels = []
    without_frame_labels = []
    for label in labels:
        frame, others = label.split(',', maxsplit=1)
        if others in without_frame_labels:
            continue
        nocopy_labels.append(label)
        without_frame_labels.append(others)

    outfile = args.label.replace('.csv', '_nocopy.csv')
    with open(outfile, 'w') as fout:
        for label in nocopy_labels:
            fout.write('{}\n'.format(label))
        fout.close()

    total = len(nocopy_labels)
    for n, label in enumerate(nocopy_labels): 
        strs = label.split(',')
        frame, id, *ltwh, conf, cate, visi = strs
        
        l, t, w, h = [float(i) for i in ltwh]
        if w < args.minw or h < args.minh:
            continue
        
        print('deal {}:{}, {}/{}'.format(int(frame), id, n, total))
        frame = int(frame)
        id = int(id)
        x, y = (l + w / 2.0) / width, (t + h / 2.0) / height
        w, h = w / width, h / height
        
        outfile = os.path.join(lb_path, '%06d.txt' % frame)
        file = open(outfile, 'a')
        file.write('{} {} {} {} {} {}\n'.format(
            0, id, x, y, w, h))
        file.close()

        outfile = os.path.join(im_path, '%06d.jpg' % frame)
        if not os.path.exists(outfile):
            # If print errors like this:
            # [mpeg4 @ 0x55b9bdcbc7c0] warning: first frame is no keyframe
            # The 'set' method will be wrong.
            # https://stackoverflow.com/questions/19404245/opencv-videocapture-set-cv-cap-prop-pos-frames-not-working
            vc.set(cv2.CAP_PROP_POS_FRAMES, frame)
            retval, image = vc.read()
            if retval:
                cv2.imwrite(outfile, image)
            else:
                print('read frame {} fail\n'.format(frame))