import torch
from torch import nn
import torch.nn.functional as F

import jde
import iou
import sosnet
import focalloss
from sosnet import (SOSNet, ShuffleNetV2BuildBlock)

class SOSMOT(nn.Module):
    def __init__(self, anchors=None, num_classes=1, num_ids=0,
        box_loss='smoothl1loss', cls_loss='crossentropyloss'):
        super(SOSMOT, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.detection_channels = (5 + self.num_classes) * 4
        self.embedding_channels = 128
        
        self.backbone = SOSNet()
        
        # YOLO1 128->128
        
        in_channels = self.backbone.out_channels['stage4']
        out_channels = 128
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True)
        )
        
        in_channels = out_channels
        self.shbk6 = []
        for repeat in range(3):
            self.shbk6.append(ShuffleNetV2BuildBlock(in_channels, out_channels, stride=1))
        self.shbk6 = torch.nn.Sequential(*self.shbk6)        
        self.conv7 = torch.nn.Conv2d(in_channels, out_channels=self.detection_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shbk8 = []
        for repeat in range(3):
            self.shbk8.append(ShuffleNetV2BuildBlock(in_channels, out_channels, stride=1))
        self.shbk8 = torch.nn.Sequential(*self.shbk8)        
        self.conv9 = torch.nn.Conv2d(in_channels, out_channels=self.embedding_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        # YOLO2 128+96=224->128
        
        in_channels = self.backbone.out_channels['stage3'] + 128
        out_channels = 128
        self.conv10 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            torch.nn.BatchNorm2d(num_features=in_channels),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True)
        )
        
        in_channels = out_channels
        self.shbk11 = []
        for repeat in range(3):
            self.shbk11.append(ShuffleNetV2BuildBlock(in_channels, out_channels, stride=1))
        self.shbk11 = torch.nn.Sequential(*self.shbk11)        
        self.conv12 = torch.nn.Conv2d(in_channels, out_channels=self.detection_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shbk13 = []
        for repeat in range(3):
            self.shbk13.append(ShuffleNetV2BuildBlock(in_channels, out_channels, stride=1))
        self.shbk13 = torch.nn.Sequential(*self.shbk13)        
        self.conv14 = torch.nn.Conv2d(in_channels, out_channels=self.embedding_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        # YOLO3 128+48=176->128
        
        in_channels = self.backbone.out_channels['stage2'] + 128
        out_channels = 128
        self.conv15 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            torch.nn.BatchNorm2d(num_features=in_channels),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True)
        )
        
        in_channels = out_channels
        self.shbk16 = []
        for repeat in range(3):
            self.shbk16.append(ShuffleNetV2BuildBlock(in_channels, out_channels, stride=1))
        self.shbk16 = torch.nn.Sequential(*self.shbk16)        
        self.conv17 = torch.nn.Conv2d(in_channels, out_channels=self.detection_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shbk18 = []
        for repeat in range(3):
            self.shbk18.append(ShuffleNetV2BuildBlock(in_channels, out_channels, stride=1))
        self.shbk18 = torch.nn.Sequential(*self.shbk18)        
        self.conv19 = torch.nn.Conv2d(in_channels, out_channels=self.embedding_channels, kernel_size=3, stride=1, padding=1, bias=True)

        '''Shared identifiers classifier'''
        
        self.classifier = torch.nn.Linear(self.embedding_channels, num_ids) if num_ids > 0 else torch.nn.Sequential()
        box_loss = iou.DIOULoss() if box_loss == 'diouloss' else torch.nn.SmoothL1Loss()
        cls_loss = focalloss.SoftmaxFocalLoss(ignore_index=-1) if cls_loss == 'softmaxfocalloss' else None
        self.criterion = jde.JDELoss(num_ids, embd_dim=self.embedding_channels, box_loss=box_loss, cls_loss=cls_loss)  if num_ids > 0 else torch.nn.Sequential()
        
        self.__init_weights()
        
    def forward(self, input, target=None, size=None):
        outputs = []
        backbone_outputs = self.backbone(input)

        # YOLO1
        conv5_out = self.conv5(backbone_outputs[-1])
        x = self.shbk6(conv5_out)
        y = self.conv7(x)
        x = self.shbk8(conv5_out)
        z = self.conv9(x)
        outputs.append(torch.cat(tensors=[y, z], dim=1))
        
        # YOLO2
        _, _, h, w = backbone_outputs[-2].size()
        x = F.interpolate(input=conv5_out, size=(h, w), mode='nearest')
        x = torch.cat(tensors=[backbone_outputs[-2], x], dim=1)
        conv10_out = self.conv10(x)
        x = self.shbk11(conv10_out)
        y = self.conv12(x)
        x = self.shbk13(conv10_out)
        z = self.conv14(x)
        outputs.append(torch.cat(tensors=[y, z], dim=1))
        
        # YOLO3
        _, _, h, w = backbone_outputs[-3].size()
        x = F.interpolate(input=conv10_out, size=(h, w), mode='nearest')
        x = torch.cat(tensors=[backbone_outputs[-3], x], dim=1)
        conv15_out = self.conv15(x)
        x = self.shbk16(conv15_out)
        y = self.conv17(x)
        x = self.shbk18(conv15_out)
        z = self.conv19(x)
        outputs.append(torch.cat(tensors=[y, z], dim=1))
        
        if target is not None:
            return self.criterion(outputs, target, size, self.classifier)
        
        return outputs
    
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

if __name__ == '__main__':
    model = SOSMOT()
    x = torch.rand(64, 3, 320, 576)
    y = model(x)
    for yi in y:
        print(yi.shape)