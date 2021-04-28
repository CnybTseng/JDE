import cv2
import argparse
import numpy as np
parser = argparse.ArgumentParser(
    description='Generating ignoring mask for mot algorithm')
parser.add_argument('--original', '-o', type=str, help='original image')
parser.add_argument('--mask', '-m', type=str, help='mask image')
args = parser.parse_args()
mask = cv2.imread(args.mask)    # mask image is made by cvat
mask = mask.max(axis=2)
mask[mask > 0] = 255
cv2.imwrite('mask.png', 255 - mask)
mask = np.expand_dims(mask, axis=2)
mask = np.repeat(mask, 3, axis=2)
im = cv2.imread(args.original)
merge = 0.5*mask + 0.5*im
merge = merge.astype(np.uint8)
cv2.imwrite('merge.png', merge)