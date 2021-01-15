import math
import torch
import matplotlib.pyplot as plt
import torchvision.models as models

lr_max = 0.025
lr_min = 0.00025
model = models.shufflenet_v2_x1_0()
optimizer = torch.optim.SGD(model.parameters(), lr=(lr_max - lr_min), momentum=0.9, weight_decay=0.0005)

epochs = 12
batchs = 50

def lr_lambda(batch):
    return 0.5 * math.cos((batch % batchs) / (batchs - 1) * math.pi)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

lrs = []
for epoch in range(epochs):
    for batch in range(batchs):
        for i, g in enumerate(optimizer.param_groups):
            g['lr'] += (lr_max - lr_min) * 0.5 + lr_min
        lrs.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step()

fig = plt.figure()
axe = fig.add_subplot(111)
axe.plot(lrs, '-s')
plt.show()