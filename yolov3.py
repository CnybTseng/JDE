# -*- coding: utf-8 -*-
# file: yolov3.py
# brief: JDE implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2020/4/8

import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3SingleDecoder(torch.nn.Module):
    '''single YOLOv3 decoder layer.
    
    Args:
        in_size: network input size. in_size should be write as [height, width].
        num_classes: number of classes
        anchors: anchor boxes list. anchors is the list of [height, width].
        embd_dim: embedding dimension.
    '''
    def __init__(self, in_size, num_classes, anchors, embd_dim=512):
        super(YOLOv3SingleDecoder, self).__init__()
        self.LongTensor = torch.cuda.LongTensor if \
            torch.cuda.is_available() else torch.LongTensor
        self.FloatTensor = torch.cuda.FloatTensor if \
            torch.cuda.is_available() else torch.FloatTensor
        self.in_size = in_size
        self.num_classes = num_classes
        self.anchors = self.FloatTensor(anchors)
        self.anchor_masks = self.LongTensor(((8,9,10,11),
            (4,5,6,7),(0,1,2,3)))
        self.embd_dim = embd_dim
        
    def forward(self, xs):
        '''YOLOv3SingleDecoder forward propagation.
        
        Args:
            xs: list of three backbone outputs.
        Returns:
            list of the decoded outputs.
        '''
        return [self.__decoder(x,self.anchors[mask,:]) for (x,mask) in \
            zip(xs,self.anchor_masks)]
    
    def __decoder(self, x, anchors):
        size = anchors.size(0) * (5 + self.num_classes)
        y = x[:, size:, ...]
        x = x[:, :size, ...]      
        
        n, c, h, w = x.size()
        num_anchors = self.anchor_masks.size(1)

        x = x.view(n, num_anchors, 5 + self.num_classes, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()        
        
        x[..., 4:6] = torch.softmax(x[..., 4:6], dim=-1)[...,[1,0]]
        x[..., 5] = 0
        
        y = y.permute(0, 2, 3, 1).unsqueeze(1).repeat(1, num_anchors,
            1, 1, 1).contiguous()
        y = F.normalize(y, dim=-1)
        
        aw = anchors[:, 0].view(1, num_anchors, 1, 1)
        ah = anchors[:, 1].view(1, num_anchors, 1, 1)
        
        [gy, gx] = torch.meshgrid(torch.arange(h), torch.arange(w))        
        gx = gx.view(1, 1, gx.size(0), gx.size(1)).type(self.FloatTensor)
        gy = gy.view(1, 1, gy.size(0), gy.size(1)).type(self.FloatTensor)                
        bx = (x[..., 0].data * aw.data + gx.data * self.in_size[1] / w)
        by = (x[..., 1].data * ah.data + gy.data * self.in_size[0] / h)
       
        bw = torch.exp(x[..., 2].data) * aw.data
        bh = torch.exp(x[..., 3].data) * ah.data
        
        bbox = torch.stack([bx,by,bw,bh], dim=-1)        
        return bbox, torch.cat([x, y], dim=-1)

class YOLOv3Decoder(YOLOv3SingleDecoder):
    '''composed YOLOv3 decoder layer.
    
    Args:
        in_size: network input size. in_size should be write as [height, width].
        num_classes: number of classes
        anchors: anchor boxes list. anchors is the list of [height, width].
        embd_dim: embedding dimension.
    '''
    def __init__(self, in_size, num_classes, anchors, embd_dim=512):
        super(YOLOv3Decoder, self).__init__(in_size, num_classes,
            anchors, embd_dim)
    
    def forward(self, xs):
        '''YOLOv3Decoder forward propagation.
        
        Args:
            xs: list of three backbone outputs.
        Returns:
            all of the decoded outputs.
        '''
        xs = super().forward(xs)
        bboxes = [x[0].view(x[0].size(0), -1, 4) for x in xs]
        confds = [x[1][..., 4].view(x[1].size(0), -1, 1) for x in xs]
        classs = [x[1][..., 5 : 5 + self.num_classes].view(x[1].size(0),
            -1, self.num_classes) for x in xs]
        embeds = [x[1][..., 5 + self.num_classes:].view(x[1].size(0),
            -1, self.embd_dim) for x in xs]
        outputs = [torch.cat((b, o, c, e), dim=-1) for (b, o, c, e) \
            in zip(bboxes, confds, classs, embeds)]
        return torch.cat(outputs, dim=1).detach().cpu()

class YOLOv3Loss(YOLOv3SingleDecoder):
    '''YOLOv3 loss layer
    
    Args:
        num_classes:
        anchors:
        classifier:
    '''
    def __init__(self, num_classes, anchors, classifier=torch.nn.Sequential()):
        super(YOLOv3Loss, self).__init__((608,1088), num_classes, anchors)
        self.classifier = classifier                     # embedding mapping to class logits
        self.sb = nn.Parameter(-4.85 * torch.ones(1))    # location task uncertainty
        self.sc = nn.Parameter(-4.15 * torch.ones(1))    # classify task uncertainty
        self.se = nn.Parameter(-2.30 * torch.ones(1))    # embedding task uncertainty
        self.bbox_lossf = nn.SmoothL1Loss()
        self.clas_lossf = nn.CrossEntropyLoss(ignore_index=-1)   # excluding no-identifier samples
        self.embd_lossf = nn.CrossEntropyLoss(ignore_index=-1)
        self.iden_thresh = 0.5  # identifier threshold
        self.frgd_thresh = 0.5  # foreground confidence threshold
        self.bkgd_thresh = 0.4  # background confidence threshold
        
    def forward(xs, targets, in_size):
        '''YOLOv3Loss layer forward propagation
        
        Args:
            xs: list of three backbone outputs.
            targets: training targets. targets=[
                [batch, category, identifier, x, y, w, h], ...]
            in_size: input size. in_size=(height, width)
        '''
        
        # update input size
        self.in_size = in_size

        # decode backbone outputs
        outputs = super().forward(xs)
        pbboxes = [output[0] for output in outputs]                                     # n*a*h*w*4
        pclasss = [output[1][..., 4 : 5 + self.num_classes] for output in outputs]      # n*a*h*w*2
        pclasss = [pclass.permute(0, 4, 1, 2, 3).contiguous() for pclass in pclasss]    # n*2*a*h*w
        pembeds = [output[1][..., 5 + self.num_classes:] for output in outputs]         # n*a*h*w*512
        
        # build targets for three heads
        tbboxes, tclasss, tidents = self._build_targets(targets)
        masks = [tc > 0 for tc in tclasss]
        
        # compute losses
        lbboxes = [self.bbox_lossf(pb[m], tb[m]) for pb, tb, m in \
            zip(pbboxes, tbboxes, masks) if m.sum() > 0 else self.FloatTensor([0])]
        lclasss = [self.clas_lossf(pc, tc) for pc, tc in zip(pclasss, tclasss)]
    
    def _build_targets(self, targets):
        '''build training targets.
        
        Args:
            targets: training targets. the targets is [
                [batch, category, identifier, x, y, w, h], ...]
        Returns:
            tbboxes: truth boundding boxes. the shape is n*a*h*w*4
            tclasss: truth confidences. the shape is n*a*h*w
            tidents: truth identifiers. the shape is n*a*h*w*1
        '''
        return None, None, None
        
if __name__ == '__main__':
    p_conf = torch.rand(1, 2, 4, 10, 18).type(torch.FloatTensor)
    tconf  = torch.randint(low=-1, high=1, size=(1, 4, 10, 18)).type(torch.LongTensor)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-1)
    loss = lossf(p_conf, tconf)
    print(p_conf)
    print(tconf)
    print(loss)