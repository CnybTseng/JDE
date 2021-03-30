import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser('top -p pid')
    parser.add_argument('--pid', type=int, help='process identity')
    parser.add_argument('--log', type=str, help='log file path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    while True:
        cmd = 'top -bn1 -p {}'.format(args.pid)
        with os.popen(cmd) as fd:
            ret = fd.read()
        lines = ret.split('\n')
        lines = [l.strip() for l in lines]
        lines = list(filter(lambda l: len(l) > 0, lines))
        if len(lines) < 7:
            break
        with open(args.log, 'a') as fd:
            fd.write(lines[-1] + '\n')
        time.sleep(10)