# -*- coding: utf-8 -*-
# file: darknet.py
# brief: JDE implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2020/4/8

import torch
import torch.nn.functional as F

class ConvBnReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, momentum, negative_slope):
        super(ConvBnReLU, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = torch.nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
        self.relu = torch.nn.LeakyReLU(negative_slope, inplace=True)
    
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Residual(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, momentum, negative_slope):
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0, bias=False)
        self.norm1 = torch.nn.BatchNorm2d(num_features=mid_channels, momentum=momentum)
        self.relu1 = torch.nn.LeakyReLU(negative_slope, inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
        self.relu2 = torch.nn.LeakyReLU(negative_slope, inplace=True)
        
    def forward(self, x):
        y = self.relu1(self.norm1(self.conv1(x)))
        y = self.relu2(self.norm2(self.conv2(y)))
        return x + y

class Upsample(torch.nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return F.interpolate(input=x, scale_factor=self.scale_factor, mode=self.mode)

class Route(torch.nn.Module):
    def __init__(self):
        super(Route, self).__init__()
        
    def forward(self, tensors):
        return torch.cat(tensors=tensors, dim=1)
       
class DarkNet(torch.nn.Module):
    def __init__(self, num_classes=1, num_ids=0):
        super(DarkNet, self).__init__()
        self.num_classes = num_classes
        self.momentum = 0.01
        self.negative_slope = 0.1
        self.detection_channels = (5 + self.num_classes) * 4
        self.embedding_channels = 512
        
        '''backbone'''
        
        self.cbrl1 = ConvBnReLU(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.zpad1 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl2 = ConvBnReLU(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage1 = Residual(in_channels=64, mid_channels=32, out_channels=64, momentum=self.momentum, negative_slope=self.negative_slope)
        self.zpad2 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl3 = ConvBnReLU(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage2 = []
        for repeate in range(2):
            self.stage2.append(Residual(in_channels=128, mid_channels=64, out_channels=128, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage2 = torch.nn.Sequential(*self.stage2)
        self.zpad3 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl4 = ConvBnReLU(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage3 = []
        for repeate in range(8):
            self.stage3.append(Residual(in_channels=256, mid_channels=128, out_channels=256, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage3 = torch.nn.Sequential(*self.stage3)
        self.zpad4 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl5 = ConvBnReLU(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage4 = []
        for repeate in range(8):
            self.stage4.append(Residual(in_channels=512, mid_channels=256, out_channels=512, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage4 = torch.nn.Sequential(*self.stage4)
        self.zpad5 = torch.nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.cbrl6 = ConvBnReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.stage5 = []
        for repeate in range(4):
            self.stage5.append(Residual(in_channels=1024, mid_channels=512, out_channels=1024, momentum=self.momentum, negative_slope=self.negative_slope))
        self.stage5 = torch.nn.Sequential(*self.stage5)
        
        '''YOLO1'''
        
        self.pair1 = []
        for repeate in range(2):
            self.pair1.append(ConvBnReLU(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope))
            self.pair1.append(ConvBnReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope))
        self.pair1 = torch.nn.Sequential(*self.pair1)
        
        self.cbrl7 = ConvBnReLU(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        self.cbrl8 = ConvBnReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.conv1 = torch.nn.Conv2d(in_channels=1024, out_channels=self.detection_channels, kernel_size=1, padding=0, bias=True)

        self.route5 = Route()
        self.conv4 = torch.nn.Conv2d(in_channels=512, out_channels=self.embedding_channels, kernel_size=3, padding=1, bias=True)
        self.route6 = Route()

        '''YOLO2'''

        self.route1 = Route()
        self.cbrl9 = ConvBnReLU(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        self.upsample1 = Upsample(scale_factor=2)
        self.route2 = Route()
        self.cbrl10 = ConvBnReLU(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.pair2 = []
        for repeate in range(2):
            self.pair2.append(ConvBnReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope))
            self.pair2.append(ConvBnReLU(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope))
        self.pair2 = torch.nn.Sequential(*self.pair2)

        self.cbrl11 = ConvBnReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.conv2 = torch.nn.Conv2d(in_channels=512, out_channels=self.detection_channels, kernel_size=1, padding=0, bias=True)
        
        self.route7 = Route()
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=self.embedding_channels, kernel_size=3, padding=1, bias=True)
        self.route8 = Route()
        
        '''YOLO3'''

        self.route3 = Route()
        self.cbrl12 = ConvBnReLU(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        self.upsample2 = Upsample(scale_factor=2)
        self.route4 = Route()
        self.cbrl13 = ConvBnReLU(in_channels=384, out_channels=128, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope)
        
        self.pair3 = []
        for repeate in range(2):
            self.pair3.append(ConvBnReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope))
            self.pair3.append(ConvBnReLU(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, momentum=self.momentum, negative_slope=self.negative_slope))
        self.pair3 = torch.nn.Sequential(*self.pair3)
        
        self.cbrl14 = ConvBnReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, momentum=self.momentum, negative_slope=self.negative_slope)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=self.detection_channels, kernel_size=1, padding=0, bias=True)

        self.route9 = Route()
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=self.embedding_channels, kernel_size=3, padding=1, bias=True)
        self.route10 = Route()

        '''Shared identities classifier'''

        self.classifier = torch.nn.Linear(self.embedding_channels, num_ids) if num_ids > 0 else torch.nn.Sequential()

        self.__init_weights()
    
    def __init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight.data)
                module.weight.data *= (2.0/module.weight.numel())
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, torch.nn.BatchNorm2d):
                # torch.nn.init.constant_(module.weight.data, 1)
                # torch.nn.init.constant_(module.bias.data, 0)
                # torch.nn.init.constant_(module.running_mean.data, 0)
                # torch.nn.init.constant_(module.running_var.data, 0)
                torch.nn.init.uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        '''前向传播.
        
        参数
        ----
        x : Tensor
            输入图像张量.
        '''

        outputs = []
        tensors1, tensors2, tensors3, tensors4 = [], [], [], []
        tensors5, tensors6, tensors7, tensors8 = [], [], [], []
        tensors9, tensors10 = [], []
        
        # backbone
        x = self.cbrl1(x)
        x = self.zpad1(x)
        x = self.cbrl2(x)
        x = self.stage1(x)
        x = self.zpad2(x)
        x = self.cbrl3(x)
        x = self.stage2(x)
        x = self.zpad3(x)
        x = self.cbrl4(x)
        x = self.stage3(x)
        tensors4.insert(0, x.clone())
        x = self.zpad4(x)
        x = self.cbrl5(x)
        x = self.stage4(x)
        tensors2.insert(0, x.clone())
        x = self.zpad5(x)
        x = self.cbrl6(x)
        x = self.stage5(x)
        
        # YOLO1
        x = self.pair1(x)
        x = self.cbrl7(x)
        tensors1.insert(0, x.clone())
        tensors5.append(x.clone())
        x = self.cbrl8(x)
        x = self.conv1(x)
        
        tensors6.append(x.clone())
        x = self.route5(tensors5)
        x = self.conv4(x)
        tensors6.append(x.clone())
        x = self.route6(tensors6)
        outputs.append(x)
        
        # YOLO2
        x = self.route1(tensors1)
        x = self.cbrl9(x)
        x = self.upsample1(x)
        tensors2.insert(0, x.clone())
        x = self.route2(tensors2)
        x = self.cbrl10(x)
        x = self.pair2(x)
        tensors3.insert(0, x.clone())
        tensors7.append(x.clone())
        x = self.cbrl11(x)
        x = self.conv2(x)
        
        tensors8.append(x.clone())
        x = self.route7(tensors7)
        x = self.conv5(x)
        tensors8.append(x.clone())
        x = self.route8(tensors8)
        outputs.append(x)

        # YOLO3
        x = self.route3(tensors3)
        x = self.cbrl12(x)
        x = self.upsample2(x)
        tensors4.insert(0, x.clone())
        x = self.route4(tensors4)
        x = self.cbrl13(x) 
        x = self.pair3(x)
        tensors9.append(x.clone())
        x = self.cbrl14(x)
        x = self.conv3(x)
        
        tensors10.append(x.clone())
        x = self.route9(tensors9)
        x = self.conv6(x)
        tensors10.append(x.clone())
        x = self.route10(tensors10)
        outputs.append(x)

        return outputs