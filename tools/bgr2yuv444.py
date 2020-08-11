import cv2
import argparse

parser = argparse.ArgumentParser(description='BGR to YUV420SP (NV12)')
parser.add_argument('--image', '-i', type=str, help='path to the image')
parser.add_argument('--yuv', type=str, help='path to the generated yuv file')
args = parser.parse_args()

bgr = cv2.imread(args.image)
yuv444 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

with open(args.yuv, 'wb') as file:
    yuv444.tofile(file)
    file.close()