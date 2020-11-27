import os
import cv2
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str,
        help='path to images')
    parser.add_argument('--video-file', type=str,
        help='the name of generated video file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img_fnames = list(sorted(glob.glob(os.path.join(args.img_path, '*.jpg'))))
    writer = None
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 20
    for img_fname in img_fnames:
        img = cv2.imread(img_fname)
        if img is None:
            break
        if writer is None:
            frameSize = (img.shape[1], img.shape[0])
            writer = cv2.VideoWriter(args.video_file, fourcc, fps, frameSize)
            if writer is None:
                print('cv2.VideoWriter() fail\n')
                break
        writer.write(img)