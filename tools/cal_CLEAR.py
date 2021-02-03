import os
import sys
import glob
import xlwt
import time
import argparse
import os.path as osp
sys.path.append(os.getcwd())
from mot.utils import mkdirs
from mot.utils import get_logger
from mot.utils import build_excel, append_excel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', type=str, default='192.168.1.174',
        help='Server IP address')
    parser.add_argument('--port', '-p', type=int, default=22,
        help='Server port')
    parser.add_argument('--username', '-un', type=str, default='root',
        help='Server username')
    parser.add_argument('--password', '-pw', type=str,
        help='Server login password')
    parser.add_argument('-dir', type=str,
        default='/home/image/tseng/project/JDE/tasks/evals',
        help='Models directory on server')
    parser.add_argument('--mot-path', '-mp', type=str,
        default='/d/Tseng/project/thirdparty/MOTChallengeEvalKit/',
        help='path to the MOT devkit')
    parser.add_argument('--save-path', '-sp', type=str,
        default='/c/Users/SH0095/Downloads/evals/',
        help='path to the result')
    parser.add_argument('--num-model', '-nm', type=int,
        help='number of models to evaluation')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Remove prefix 'C:/msys64' from dir in Windows.
    # args.dir = args.dir.split('msys64')[1]
    mkdirs(args.save_path)
    logger = get_logger(path=osp.join(args.save_path, 'log.txt'))    
    head = ['MOTA', 'MOTP', 'IDF1', 'IDP', 'IDR', 'Rcll', 'Prcn',
        'TP', 'FP', 'FN', 'MTR', 'PTR', 'MLR', 'MT', 'PT', 'ML', 'FAR',
        'FM', 'FMR', 'IDSW', 'IDSWR']
    xlspath = osp.join(args.save_path, 'mot_results.xls')
    build_excel(xlspath, head, 'CLEAR', override=False)
    done = []
    while True:
        # Exit dead loop.
        if len(done) >= args.num_model:
            logger.info('All works have been done.')
            break
        
        # Read file list from server.
        cmd = "sshpass -p {} ssh {}@{} -p {} \"cd \"{}\"; basename -a *; exit\"".format(
            args.password, args.username, args.ip, args.port, args.dir)
        
        logger.info('File read command: {}'.format(cmd))
        with os.popen(cmd) as fd:
            returns = fd.read()

        folders = returns.split(sep='\n')
        folders = [f.strip() for f in folders]
        folders = list(filter(lambda f: len(f) > 0, folders))
        folders = [f for f in folders if not 'pth' in f]
        folders = [f for f in folders if not 'log' in f]
        folders = [f for f in folders if not '*' in f]
        logger.info('File read result:\n{}'.format(folders))

        for folder in folders:
            if folder in done:
                continue
            
            # Make sure that the folder have been prepared by server.
            time.sleep(1)
            
            # Download files from server.
            cmd = 'sshpass -p {} scp -r -P {} {}@{}:{} {}'.format(
                args.password, args.port, args.username, args.ip,
                args.dir + '/' + folder, args.save_path)
            logger.info('Download command: {}'.format(cmd))
            os.system(cmd)

            # Copy files to MOTChallengeEvalKit\res\MOT16res
            dpath = osp.join(args.mot_path, 'res', 'MOT16res/').replace('\\', '/')
            cmd = 'cp {} {}'.format(
                osp.join(args.save_path, folder, '*.txt').replace('\\', '/'),
                dpath)
            logger.info('copy command: {}'.format(cmd))
            os.system(cmd)
          
            # Excute MOT\evalMOT.py
            cmd = 'sh tools/evalMOT.sh {}'.format(args.mot_path)
            logger.info('MOT run command: {}'.format(cmd))
            with os.popen(cmd) as fd:
                text = fd.read()
            
            # Write result to excel.
            lines = text.split(sep='\n')
            lines = [line.strip() for line in lines]
            lines = list(filter(lambda x: len(x) > 0, lines))
            if len(lines) == 0:
                logger.info('Whoops! No inference output for current model!')
                words = ['NULL', 'NULL']
            else:
                words = lines[-1].split()   # Overall performance
            append_excel(xlspath, [words[1:]])
            done.append(folder)
            
            cmd = 'rm -f ' + osp.join(dpath, '*.txt').replace('\\', '/')
            logger.info('clean command: {}'.format(cmd))
            os.system(cmd)
        logger.info('waiting for data ...')
        time.sleep(10)