import lap
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cost', type=str, help='cost')
parser.add_argument('--rows', type=int, help='rows')
parser.add_argument('--cols', type=int, help='cols')
args = parser.parse_args()

cost = np.fromfile(open(args.cost, 'rb'), dtype=np.float32).reshape(args.rows, args.cols)

opt, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=0.7)

print(opt)
print(x)
print(y)