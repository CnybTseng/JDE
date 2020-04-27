import cv2
import torch
import argparse
import numpy as np

import transforms as T

transforms = []
transforms.append(T.ToTensor())
transforms.append(T.RandomSpatialJitter(jitter=0.3,net_w=576,net_h=320))
transforms.append(T.RandomColorJitter(hue=0.1,saturation=1.5,exposure=1.5))
transforms.append(T.RandomHorizontalFlip(prob=0.5))

transforms = T.Compose(transforms)

im = cv2.imread('dog.jpg')
print(im.shape)

im2, _ = transforms(im, torch.FloatTensor(0, 7))

print(im2.size())

im3 = im2.squeeze().permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
cv2.imwrite('im3.png', im3)