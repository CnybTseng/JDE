import os
import cv2
import sys
import torch
import pickle
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('.')
import dataset as ds

boxw0 = []
boxh0 = []
cachefile = './wh.pkl'
if not os.path.isfile(cachefile):
    dataset = ds.HotchpotchDataset('/data/tseng/dataset/jde', './data/train.txt', 'shufflenetv2', False)
    in_size = torch.IntTensor([320, 576])
    collate_fn = partial(ds.collate_fn, in_size=in_size, train=False)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn)
    for batch, (images, targets) in enumerate(data_loader):
        n, c, h, w = images.size()
        if targets.size(0) > 0:
            tw = (targets[:, 5] * w).numpy().round().astype(np.int)
            th = (targets[:, 6] * h).numpy().round().astype(np.int)
            mask = (tw < 8) | (th < 8)
            if mask.sum() > 0:
                with open('error_box.txt', 'a') as file:
                    file.write('{}\n'.format(targets.numpy()))
                    file.write('{}x{}\n'.format(h, w))
                    file.write('{}\n'.format(tw))
                    file.write('{}\n\n\n'.format(th))
                    file.close()
            boxw0.append(tw[~mask].tolist())
            boxh0.append(th[~mask].tolist())
        print('\rDeal {}/{}'.format(batch, len(data_loader)), end='', flush=True)
        # if batch > 100: break
    with open(cachefile, 'wb') as file:
        pickle.dump([boxw0, boxh0], file)
else:
    with open(cachefile, 'rb') as file:
        boxw0, boxh0 = pickle.load(file)

print('')
boxw, boxh = [], []
for bw0, bh0 in zip(boxw0, boxh0):
    boxw += bw0
    boxh += bh0

with open('./wh.txt', 'w') as file:
    for hi, wi in zip(boxh, boxw):
        file.write('{} {}\n'.format(hi, wi))
    file.close()

maxw = max(boxw)
maxh = max(boxh)
minw, minh = min(boxw), min(boxh)
print('maxh:{}, maxw:{}, minh:{}, minw:{}'.format(maxh, maxw, minh, minw))

area = [hi * wi for hi, wi in zip(boxh, boxw)]
area_nocopy = list(tuple(area))
print('area: {}, area_nocopy: {}'.format(len(area), len(area_nocopy)))
area_sorted = area_nocopy.copy()
area_sorted.sort()
print('min(area): {}, max(area): {}'.format(area_sorted[0], area_sorted[-1]))
n = len(area_sorted)
small_object_thresh = 32*32 # area_sorted[round(n / 3)]
median_object_thresh = 96*96 # area_sorted[round(n * 2 / 3)]
print('small_object_thresh: {}, median_object_thresh: {}'.format(small_object_thresh, median_object_thresh))

small_object_ratio, median_object_ratio, large_object_ratio = [], [], []
for bw0, bh0 in zip(boxw0, boxh0):
    area0 = np.array([w * h for w, h in zip(bw0, bh0)])
    small_mask = area0 < small_object_thresh
    media_mask = area0 < median_object_thresh
    small_object_ratio.append(sum(small_mask) / (len(area0) + 1e-9))
    median_object_ratio.append(sum(~small_mask & media_mask) / (len(area0) + 1e-9))
    large_object_ratio.append(sum(~media_mask) / (len(area0) + 1e-9))
small_object_ratio = np.array(small_object_ratio)
median_object_ratio = np.array(median_object_ratio)
large_object_ratio = np.array(large_object_ratio)
small_object_in_image_ratio = sum(small_object_ratio > 0) / len(small_object_ratio)
median_object_in_image_ratio = sum(median_object_ratio > 0) / len(median_object_ratio)
large_oject_in_image_ratio = sum(large_object_ratio > 0) / len(large_object_ratio)
print('small_object_in_image_ratio: {}'.format(small_object_in_image_ratio))
print('median_object_in_image_ratio: {}'.format(median_object_in_image_ratio))
print('large_oject_in_image_ratio: {}'.format(large_oject_in_image_ratio))

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_title('small(red)/median(green)/large(blue) object ratio per image')
ax3.set_xlabel('image index')
ax3.set_ylabel('ratio')
ax3.bar(range(len(small_object_ratio)), small_object_ratio, color='r')
ax3.bar(range(len(median_object_ratio)), median_object_ratio, color='g')
ax3.bar(range(len(large_object_ratio)), large_object_ratio, color='b')
fig3.savefig('./ratio.png', dpi=2000)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('object area histogram')
ax1.set_xlabel('area')
ax1.set_ylabel('total')
ax1.hist(area, bins=len(area_nocopy))
fig1.savefig('./area.png', dpi=1000)
# plt.show()

x, y = np.meshgrid(range(maxw + 1), range(maxh + 1))
z = np.zeros((maxh + 1, maxw + 1))
for hi, wi in zip(boxh, boxw):
    z[hi, wi] += 1

minz, maxz = z.min(), z.max()
im = 255 * (z - minz) / (maxz - minz)
im = im.astype(np.uint8)
cv2.imwrite('wh-2d.png', im)

x, y, z = x.ravel(), y.ravel(), z.ravel()
bottom = np.zeros_like(z)
width = depth = 1

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title('w/h distribution')
ax2.set_xlabel('w')
ax2.set_ylabel('h')
ax2.set_zlabel('total')
ax2.bar3d(x, y, bottom, width, depth, z, shade=True)
fig2.savefig('./wh-3d.png', dpi=2000)
# plt.show()