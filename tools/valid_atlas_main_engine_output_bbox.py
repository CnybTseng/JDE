import cv2
import numpy as np

data = np.fromfile('./platforms/atlas200dk/run/out/test/data/000002.bin', dtype=np.float32)
data = 255 * data
data = data.astype(np.uint8)

im = data.reshape(3, 320, 576).transpose(1, 2, 0)
im = im[:, :, ::-1]

dets = np.loadtxt('./platforms/atlas200dk/run/out/test/data/result.txt')

for det in dets:
    t, l, b, r = det[2:].round().astype(np.int32).tolist()
    color = np.random.randint(0, 256, size=(3,)).tolist()
    im = cv2.rectangle(im, (l, t), (r, b), color, 1)

cv2.imwrite('./platforms/atlas200dk/run/out/test/data/test_image.png', im)