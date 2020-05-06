import numpy as np
import matplotlib.pyplot as plt

num_idents = np.arange(1, 1001)
embd_scale = np.ones(1000)
embd_scale[1:] = np.sqrt(2) * np.log(num_idents[1:] - 1)

plt.xlabel('num_idents')
plt.ylabel('embd_scale')
plt.plot(num_idents, embd_scale, '-')
plt.show()