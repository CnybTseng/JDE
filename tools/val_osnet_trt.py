import os
import torch
import argparse
import torchreid
import numpy as np
import os.path as osp
from torchreid.utils import FeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', '-mp', type=str,
        help='model path')
    parser.add_argument('--files-path', '-fp', type=str,
        default='./tasks',
        help='the directoy where in.bin and out.bin belong to')
    parser.add_argument('--batch-size', '-bs', type=int,
        default=1,
        help='batch size')
    return parser.parse_args()

def main():
    args = parse_args()
    
    pwd = 'sshpass -p sihan123 '
    scp = 'scp sihan@192.168.1.100:/home/sihan/Documents/motoolkit/reid/build/install/bin/'
    os.system(pwd + scp + 'in.bin {}'.format(args.files_path))
    os.system(pwd + scp + 'out.bin {}'.format(args.files_path))

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=args.model_path,
        device='cuda')
    
    with open(osp.join(args.files_path, 'in.bin'), 'rb') as fd:
        input = np.fromfile(fd, dtype=np.float32)
        input = torch.from_numpy(input).view(args.batch_size, 3, 256, 128).cuda()
    
    baseline_output = extractor(input).view(-1)
    print('baseline_output:\n{}'.format(baseline_output))
    with open(osp.join(args.files_path, 'out.bin'), 'rb') as fd:
        test_output = np.fromfile(fd, dtype=np.float32)
        test_output = torch.from_numpy(test_output).cuda()
        print('test_output:\n{}'.format(test_output))
    
    error = torch.abs(baseline_output - test_output)
    print('error: min {}, max {}, mean {}'.format(error.min(), error.max(), error.mean()))

if __name__ == '__main__':
    main()