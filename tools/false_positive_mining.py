import os
import re
import cv2
import sys
import torch
import argparse
import collections

sys.path.append('.')
import jde
import dataset
import tracker
import shufflenetv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', type=str,
        help='video path')
    parser.add_argument('--model', '-m', type=str,
        help='model path')
    parser.add_argument('--in-size', type=int, nargs='+',
        default=[320,576],
        help='network input size')
    parser.add_argument('--embedding', type=int, default=128,
        help='embedding dimension')
    parser.add_argument('--score-thresh', '-s', type=float, default=0.5,
        help='nms score threshold, default=0.5, it must be in [0,1]')
    parser.add_argument('--iou-thresh', '-i', type=float, default=0.4,
        help='nms iou threshold, default=0.4, it must be in [0,1]')
    parser.add_argument('--save-path', type=str,
        default=os.path.join(os.getcwd(), 'mining'),
        help='path to the mining samples')
    args = parser.parse_args()
    return args

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = shufflenetv2.ShuffleNetV2().to(device)
    
    model_dict = model.state_dict()
    trained_model_dict = torch.load(args.model, map_location='cpu')
    trained_model_dict = {k : v for (k, v) in trained_model_dict.items() if k in model_dict}
    trained_model_dict = collections.OrderedDict(trained_model_dict)
    model_dict.update(trained_model_dict)
    model.load_state_dict(model_dict)
    
    model.eval()
    
    h, w = args.in_size
    decoder = jde.JDEcoder(args.in_size, embd_dim=args.embedding)
    dataloader = dataset.VideoLoader(args.video, (h, w, 3))
    
    mkdir(os.path.join(args.save_path, 'images'))
    mkdir(os.path.join(args.save_path, 'labels_with_ids'))
    for i, (path, im, lb_im) in enumerate(dataloader):
        input = torch.from_numpy(lb_im).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input)
        outputs = decoder(outputs)
        outputs = tracker.nonmax_suppression(outputs, args.score_thresh, args.iou_thresh)[0]
        if outputs is not None:
            segments = re.split(r'[\\, /]', path)
            outdir = os.path.join(args.save_path, 'images', segments[-1])
            cv2.imwrite(outdir, im)
            outdir = outdir.replace('images', 'labels_with_ids').replace('jpg', 'txt')
            with open(outdir, 'w') as file:
                file.close()
        print('\rprocess: {}/{}'.format(i, len(dataloader)), end='', flush=True)