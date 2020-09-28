# -*- coding: utf-8 -*-
# file: shufflenetv2.py
# brief: YOLOv3 implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2019/9/12

import sys
import torch
import numpy as np
import torch.nn.functional as F

import yolov3

class ShuffleNetV2Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride):
        super(ShuffleNetV2Block, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        
        padding = kernel_size // 2
        major_out_channels = out_channels - in_channels
        self.major_branch = torch.nn.Sequential(\
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False),
            torch.nn.BatchNorm2d(num_features=mid_channels),
            torch.nn.Conv2d(mid_channels, major_out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=major_out_channels),
            torch.nn.ReLU(inplace=True))
        
        if stride == 1: return
        self.minor_branch = torch.nn.Sequential(\
            torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            torch.nn.BatchNorm2d(num_features=in_channels),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=in_channels),
            torch.nn.ReLU(inplace=True))
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = self.channel_shuffle(x)
            return torch.cat((x1, self.major_branch(x2)), dim=1)
        elif self.stride == 2:
            return torch.cat((self.minor_branch(x), self.major_branch(x)), dim=1)
    
    def channel_shuffle(self, x):
        n, c, h, w = x.data.size()
        assert c % 4 == 0
        x = x.reshape(n * c // 2, 2, h * w)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, c // 2, h, w)
        return x[0], x[1]

class ShuffleNetV2(torch.nn.Module):
    def __init__(self, anchors, num_classes=1, num_ids=0, model_size='2.0x'):
        super(ShuffleNetV2, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.detection_channels = (5 + self.num_classes) * 4
        self.embedding_channels = 128
        
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24,  48,  96, 192, 512, 256, 128]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 512, 256, 128]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 512, 256, 128]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 512, 256, 128]
        else:
            raise NotImplementedError
        
        # Backbone
        
        in_channels = 3
        out_channels = self.stage_out_channels[1]
        self.conv1 = torch.nn.Sequential(\
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True))
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = []
        in_channels = out_channels
        out_channels = self.stage_out_channels[2]
        self.stage2.append(ShuffleNetV2Block(in_channels, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=2))
        in_channels = out_channels
        for r in range(3):
            self.stage2.append(ShuffleNetV2Block(in_channels//2, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=1)) 
        self.stage2 = torch.nn.Sequential(*self.stage2)
        
        self.stage3 = []
        out_channels = self.stage_out_channels[3]
        self.stage3.append(ShuffleNetV2Block(in_channels, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=2))
        in_channels = out_channels
        for r in range(7):
            self.stage3.append(ShuffleNetV2Block(in_channels//2, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=1))            
        self.stage3 = torch.nn.Sequential(*self.stage3)
        
        self.stage4 = []
        out_channels = self.stage_out_channels[4]
        self.stage4.append(ShuffleNetV2Block(in_channels, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=2))
        in_channels = out_channels
        for r in range(3):
            self.stage4.append(ShuffleNetV2Block(in_channels//2, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=1))   
        self.stage4 = torch.nn.Sequential(*self.stage4)
        
        # YOLO1 192->192->512
        
        self.stage5 = []
        in_channels = out_channels
        for repeat in range(1):
            self.stage5.append(ShuffleNetV2Block(in_channels//2, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=1))
        self.stage5 = torch.nn.Sequential(*self.stage5)
        out_channels = self.stage_out_channels[5]
        
        # fusion groups
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True)
        )
        
        in_channels = out_channels
        self.conv11 = torch.nn.Conv2d(in_channels, out_channels=self.detection_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv12 = torch.nn.Conv2d(in_channels, out_channels=self.embedding_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        # YOLO2 96+512=608->608->256
        
        in_channels = self.stage_out_channels[3] + out_channels
        out_channels = in_channels
        self.stage7 = []
        for repeat in range(1):
            self.stage7.append(ShuffleNetV2Block(in_channels//2, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=1))
        self.stage7 = torch.nn.Sequential(*self.stage7)
        
        # fusion groups
        out_channels = self.stage_out_channels[6]
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True)
        )
        
        in_channels = out_channels
        self.conv13 = torch.nn.Conv2d(in_channels, out_channels=self.detection_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv14 = torch.nn.Conv2d(in_channels, out_channels=self.embedding_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        # YOLO3 48+256=304->304->128
        
        in_channels = self.stage_out_channels[2] + out_channels
        out_channels = in_channels
        self.stage9 = []
        for repeat in range(1):
            self.stage9.append(ShuffleNetV2Block(in_channels//2, out_channels, mid_channels=out_channels//2, kernel_size=3, stride=1))
        self.stage9 = torch.nn.Sequential(*self.stage9)
        
        # fusion groups
        out_channels = self.stage_out_channels[7]
        self.conv10 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True)
        )
        
        in_channels = out_channels
        self.conv15 = torch.nn.Conv2d(in_channels, out_channels=self.detection_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv16 = torch.nn.Conv2d(in_channels, out_channels=self.embedding_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        '''Shared identifiers classifier'''
        
        self.classifier = torch.nn.Linear(self.embedding_channels, num_ids) if num_ids > 0 else torch.nn.Sequential()
        self.criterion = yolov3.YOLOv3Loss(num_classes, anchors, num_ids, embd_dim=self.embedding_channels) if num_ids > 0 else torch.nn.Sequential()
        
        self.__init_weights()
        
    def __init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight.data)
                module.weight.data *= (2.0/module.weight.numel())
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight.data, 1)
                torch.nn.init.constant_(module.bias.data, 0)
                torch.nn.init.constant_(module.running_mean.data, 0)
                torch.nn.init.constant_(module.running_var.data, 0) 
    
    def forward(self, x, targets=None, size=None):
        outputs = []
        
        # Backbone
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        stage2_out = x.clone()
        x = self.stage3(x)
        stage3_out = x.clone()
        x = self.stage4(x)

        # YOLO1
        x = self.stage5(x)
        x = self.conv6(x)
        _, _, h, w = x.size()
        conv6_out = F.interpolate(input=x, size=(h*2,w*2), mode='nearest')
        y = self.conv11(x)
        z = self.conv12(x)
        outputs.append(torch.cat(tensors=[y, z], dim=1))
        
        # YOLO2
        x = torch.cat([stage3_out, conv6_out], dim=1)
        x = self.stage7(x)
        x = self.conv8(x)
        _, _, h, w = x.size()
        conv8_out = F.interpolate(input=x, size=(h*2,w*2), mode='nearest')
        y = self.conv13(x)
        z = self.conv14(x)
        outputs.append(torch.cat(tensors=[y, z], dim=1))
        
        # YOLO3
        x = torch.cat([stage2_out, conv8_out], dim=1)
        x = self.stage9(x)
        x = self.conv10(x)
        y = self.conv15(x)
        z = self.conv16(x)
        outputs.append(torch.cat(tensors=[y, z], dim=1))
        
        if targets is not None:
            return self.criterion(outputs, targets, size, self.classifier)
        
        return outputs

if __name__ == '__main__':
    anchors = np.random.randint(low=10, high=150, size=(12,2))
    model = ShuffleNetV2(anchors, model_size='0.5x')
    model.eval()
    x = torch.rand(1, 3, 320, 576)
    ys = model(x)
    for i,y in enumerate(ys):
        print(f'the {i+1}th output size is {y.size()}')