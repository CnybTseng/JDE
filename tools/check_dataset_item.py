import cv2
import argparse
import numpy as np

def printf(val, highlight, end='\n'):
    if highlight:
        print('\033[1;33;40m{}\033[0m'.format(val), end=end)
    else:
        print('{}'.format(val), end=end)

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', type=str,
    help='absolutly path to the image')
args = parser.parse_args()

image = cv2.imread(args.image)
h, w, _ = image.shape

label_path = args.image.replace('images', 'labels_with_ids')
label_path = label_path.replace('.png', '.txt')
label_path = label_path.replace('.jpg', '.txt')

labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
for label in labels:
    bx = round((label[2] - label[4] / 2) * w)
    by = round((label[3] - label[5] / 2) * h)
    bw = round(label[4] * w)
    bh = round(label[5] * h)
    # Print wrong bounding box parameters.
    printf(bx, bx < 0 or bx >= w, ',')    
    printf(by, by < 0 or by >= h, ',')    
    printf(bw, bw <= 0 or bw >= w - bx, ',')    
    printf(bh, bh <= 0 or bh >= h - by, '\n')
    image = cv2.rectangle(image, (bx, by, bw, bh), (0,255,255), 2)
cv2.imwrite('check.jpg', image)