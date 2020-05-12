import lap
import numpy as np

cost = np.array([[1, 5, 6, 2], [4, 5, 8, 0], [1, 1, 6, 5]])

opt, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=0.7)

print(opt)
print(x)
print(y)