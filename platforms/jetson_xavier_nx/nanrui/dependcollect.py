import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='colloect dependencies automatically')
    parser.add_argument('--dir-exe', '-d', type=str,
        help='path to the execution file')
    parser.add_argument('--package-dir', '-p', type=str, default='./dependencies',
        help='path to the dependency copies')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cmd = 'ldd {}'.format(args.dir_exe)
    if not os.path.exists(args.package_dir):
        os.makedirs(args.package_dir)
    with os.popen(cmd) as fd:
        ret = fd.read()
    lines = ret.split('\n')
    lines = [l.strip() for l in lines]
    lines = list(filter(lambda l: len(l) > 0, lines))
    lines = [l for l in lines if '=>' in l]
    for l in lines:
        strs = l.split()
        if len(strs) == 4:
            os.system('cp {} {}'.format(strs[2], args.package_dir))
        else:
            print('ERROR: {}'.format(l))