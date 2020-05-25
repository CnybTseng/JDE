import argparse
import numpy as np
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser()
parser.add_argument('--ma', type=int, help='ma')
parser.add_argument('--mb', type=int, help='mb')
parser.add_argument('--n', type=int, help='n')
args = parser.parse_args()

XA = np.fromfile(open('XA.bin', 'rb'), dtype=np.float32).reshape(-1, args.n)
print(XA.shape)
XB = np.fromfile(open('XB.bin', 'rb'), dtype=np.float32).reshape(-1, args.n)
print(XB.shape)
Y  = np.fromfile(open('Y.bin', 'rb'),  dtype=np.float32).reshape(args.ma, args.mb)

YY = cdist(XA, XB)

delta = abs(YY - Y).mean()
print('delta is {}'.format(delta))
print('Y is {}'.format(Y))
print('YY is {}'.format(YY))