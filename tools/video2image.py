import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', '-v', type=str,
	help='path to video file')
args = parser.parse_args()

root, _ = os.path.split(args.video)
im_path = os.path.join(root, 'images')
if not os.path.exists(im_path):
	os.mkdir(im_path)

n = 0
cap = cv2.VideoCapture(args.video)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
while True:
	retval, frame = cap.read()
	if not retval:
		break
	outfile = os.path.join(im_path, '%06d.jpg' % n)
	cv2.imwrite(outfile, frame)
	n = n + 1
	print('Deal {}/{}'.format(n, frames))
