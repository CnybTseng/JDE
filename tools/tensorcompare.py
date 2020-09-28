import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str, help='pytorch tensor path')
    parser.add_argument('--tt', type=str, help='tensorrt tensor path')
    args = parser.parse_args()
    
    A = torch.load(args.pt)
    n, c, h, w = A.size()
    B = torch.from_numpy(np.fromfile(args.tt, dtype=np.float32)).view(n, c, h, w)
    delta = torch.abs(A - B)
    print('pt == tt ? {} {} {}'.format(delta.min(), delta.max(), delta.mean()))