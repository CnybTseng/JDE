import cv2
import numpy as np

with open('000001.yuv420sp', 'rb') as file:
    data = np.fromfile(file, dtype=np.uint8)
    data = data.reshape(-1, 576)
    bgr = cv2.cvtColor(data, cv2.COLOR_YUV420sp2BGR)
    cv2.imshow('test', bgr)
    cv2.waitKey(0)