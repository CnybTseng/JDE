import numpy as np
from scipy.spatial.distance import mahalanobis

u = np.fromfile(open('u.bin', 'rb'), dtype=np.float32)
v = np.fromfile(open('v.bin', 'rb'), dtype=np.float32)
VI = np.fromfile(open('VI.bin', 'rb'), dtype=np.float32).reshape(u.shape[0], v.shape[0])

dist = mahalanobis(u, v, VI)

print('dist is {}'.format(dist))