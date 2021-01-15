import os
import glob
import xlwt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', '-rp', type=str,
    help='path to the data')
parser.add_argument('--mot-path', '-mp', type=str,
    help='path to the MOT devkit')
parser.add_argument('--save-path', '-sp', type=str,
    help='path to the result')
args = parser.parse_args()

paths = sorted(os.listdir(args.data_path))
wb = xlwt.Workbook()
sh = wb.add_sheet('mot16 performance')

indicators = ['MOTA', 'MOTP', 'IDF1', 'IDP', 'IDR', 'Rcll', 'Prcn',
    'TP', 'FP', 'FN', 'MTR', 'PTR', 'MLR', 'MT', 'PT', 'ML', 'FAR',
    'FM', 'FMR', 'IDSW', 'IDSWR']
for i, indicator in enumerate(indicators):
    sh.write(0, i, indicator)

for i, path in enumerate(paths):
    os.system('copy {} {}'.format(os.path.join(args.data_path, path, '*.txt'),
        os.path.join(args.mot_path, 'res', 'MOT16res')))
    cmd = 'cd {} && python {}'.format(args.mot_path,
        os.path.join('MOT', 'evalMOT.py'))
    print(cmd)
    with os.popen(cmd) as fd:
        text = fd.read()
    
    lines = text.split(sep='\n')
    lines = [line.strip() for line in lines]
    lines = list(filter(lambda x: len(x) > 0, lines))
    words = lines[-1].split()   # Overall performance
    print(words)
    for j in range(1, len(words)):
        sh.write(i + 1, j - 1, float(words[j]))

wb.save(os.path.join(args.save_path, 'mot_results.xls'))