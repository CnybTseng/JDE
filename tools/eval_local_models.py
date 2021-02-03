import os
import sys
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', '-dr', type=str,
    default='/data/tseng/dataset/jde/MOT16/train',
    help='dataset root directory')
parser.add_argument('--model-path', '-mp', type=str,
    help='path to model candidates')
parser.add_argument('--version', type=str, default='2.0',
    help='lightweight JDE version, 1.0 or 2.0')
parser.add_argument('--save-path', '-sp', type=str,
    default='./model_performance',
    help='path to the result')
args = parser.parse_args()
print(args)

sys.path.append(os.getcwd())
if os.path.isfile(args.model_path):
    paths = [args.model_path]
else:
    paths = sorted(glob.glob(os.path.join(args.model_path, '*.pth')))

for path in paths:
    _, name_with_ext = os.path.split(path)
    name, _ = os.path.splitext(name_with_ext)
    if name == 'trainer-ckpt' or name == 'latest':
        continue
    rpath = os.path.join(args.save_path, name)
    if not os.path.exists(rpath):
        os.makedirs(rpath)

    os.system('python 3rdparty/Towards-Realtime-MOT/track.py'
        ' --cfg 3rdparty/Towards-Realtime-MOT/cfg/yolov3_576x320.cfg'
        ' --weights {} --test-mot16 --data-root {} --version {}'.format(
        path, args.data_root, args.version))
    
    exp_name = path.split('/')[-2]
    results = os.path.join(args.data_root, '..', 'results', exp_name, '*')
    os.system('mv {} {}'.format(results, rpath))