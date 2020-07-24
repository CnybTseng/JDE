import cv2
import argparse
import numpy as np

im = None
pos = []
winname = ''

def parse_args():
    parser = argparse.ArgumentParser(
        description='image cropping tool')
    parser.add_argument('--image', '-i', type=str,
        help='path to the image')
    args = parser.parse_args()
    return args

def on_mouse_lbuttondown(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('{} {}'.format(x, y))
        pos.append((x, y))
        # cv2.circle(im, (x, y), 2, (255, 0, 0))
        
        if len(pos) == 2:
            [x1, y1], [x2, y2] = pos
            roi = im[y1:y2, x1:x2, :]
            cv2.imwrite("roi.png", roi)
        
        cv2.imshow(winname, im)
    elif event == 113:
        print('exit')

if __name__ == '__main__':
    args = parse_args()
    im = cv2.imread(args.image)
    winname = "{}".format(args.image)
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, on_mouse_lbuttondown)
    cv2.imshow(winname, im)
    while (True):
        try:
            cv2.waitKey(100)
        except Exception:
            break
    cv2.destroyAllWindows()