import os
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='convert MOT format '
        'labels to format like [class_id, identifier, x, y, w, h]')
    parser.add_argument('--root-dir', type=str, help=
        'path to the MOT dataset root directory')
    parser.add_argument('--save-dir', type=str, default='', help=
        'path to the generated label file')
    return parser.parse_args()

def mkdirs(path):
    '''Make directory if path is not exists.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def main(args):
    mkdirs(args.save_dir)
    width, height = 0, 0
    for dirpath, dirnames, filenames in os.walk(args.root_dir):        
        if dirnames:
            for filename in filenames:
                if filename != 'seqinfo.ini':
                    continue
                fullpath = os.path.join(dirpath, filename)
                infos = open(fullpath, 'r').read().split()
                for info in infos:
                    info = info.strip()
                    if not info:
                        continue
                    strs = info.split('=')
                    strs = [s.strip() for s in strs]
                    if len(strs) == 2:
                        if strs[0] == 'imWidth':
                            width = int(strs[1])
                        elif strs[0] == 'imHeight':
                            height = int(strs[1])
            continue
        for filename in filenames:
            if filename != 'gt.txt':
                continue
            if width == 0 or height == 0:
                print('read image size fail {} {}'.format(width, height))
                continue
            strs = re.split(r'[/,\\]', dirpath)
            subdir = os.path.join(args.save_dir, strs[-2], 'labels_with_ids')
            mkdirs(subdir)
            fullpath = os.path.join(dirpath, filename)
            labels = open(fullpath, 'r').read().split()
            print('{}, {}x{}'.format(fullpath, width, height))
            for label in labels:
                label = label.strip()
                if not label:
                    continue
                strs = label.split(',')
                frame, id, *ltwh, conf, cate, visi = strs
                if int(conf) == 0:
                    continue
                outfile = os.path.join(subdir,
                    '%06d' % int(frame) + '.txt')
                id = int(id)
                l, t, w, h = [float(i) for i in ltwh]
                x, y = (l + w / 2.0) / width, (t + h / 2.0) / height
                w, h = w / width, h / height
                file = open(outfile, 'a')
                file.write('{} {} {} {} {} {}\n'.format(
                    0, id, x, y, w, h))
                file.close()
            width, height = 0, 0

if __name__ == '__main__':
    args = parse_args()
    main(args)