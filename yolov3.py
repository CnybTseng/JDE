# -*- coding: utf-8 -*-
# file: yolov3.py
# brief: JDE implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2020/4/8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3SingleDecoder(torch.nn.Module):
    '''single YOLOv3 decoder layer.
    
    Args:
        in_size: network input size. in_size should be write as [height, width].
        num_classes: number of classes
        anchors: anchor boxes list. anchors is the list of [width, height].
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
        
        if self.training:
            x[..., 4:6] = torch.softmax(x[..., 4:6].clone(), dim=-1)
        else:
            x[..., 4:6] = torch.softmax(x[..., 4:6], dim=-1)[...,[1,0]]
            x[..., 5] = 0
        
        y = F.normalize(y).permute(0, 2, 3, 1).contiguous()
        
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
        return bbox, x, y

class YOLOv3Decoder(YOLOv3SingleDecoder):
    '''composed YOLOv3 decoder layer.
    
    Args:
        in_size: network input size. in_size should be write as [height, width].
        num_classes: number of classes
        anchors: anchor boxes list. anchors is the list of [width, height].
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
        a = self.anchor_masks.size(1)
        embeds = [x[2].unsqueeze(1).repeat(1, a, 1, 1, 1).view(
            x[2].size(0), -1, self.embd_dim).contiguous() for x in xs]
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
    def __init__(self, num_classes, anchors, num_idents, classifier=torch.nn.Sequential()):
        super(YOLOv3Loss, self).__init__((608,1088), num_classes, anchors,
            embd_dim=512)
        self.num_idents = num_idents
        self.classifier = classifier                     # embedding mapping to class logits
        self.sb = nn.Parameter(-4.85 * torch.ones(1))    # location task uncertainty
        self.sc = nn.Parameter(-4.15 * torch.ones(1))    # classification task uncertainty
        self.si = nn.Parameter(-2.30 * torch.ones(1))    # identification task uncertainty
        self.bbox_lossf = nn.SmoothL1Loss()
        self.clas_lossf = nn.CrossEntropyLoss(ignore_index=-1)   # excluding no-identifier samples
        self.iden_lossf = nn.CrossEntropyLoss(ignore_index=-1)
        self.iden_thresh = 0.5  # identifier threshold
        self.frgd_thresh = 0.5  # foreground confidence threshold
        self.bkgd_thresh = 0.4  # background confidence threshold
        self.embd_scale  = math.sqrt(2) * math.log(self.num_idents) \
            if self.num_idents > 1 else 1
        self.BoolTensor = torch.cuda.BoolTensor if \
            torch.cuda.is_available() else torch.BoolTensor
        
    def forward(self, xs, targets, in_size):
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
        pbboxes = [output[1][..., : 4] for output in outputs]                           # n*a*h*w*4
        pclasss = [output[1][..., 4 : 5 + self.num_classes] for output in outputs]      # n*a*h*w*2
        pclasss = [pclass.permute(0, 4, 1, 2, 3).contiguous() for pclass in pclasss]    # n*2*a*h*w
        pembeds = [output[2] for output in outputs]                                     # n*h*w*512
        
        # build targets for three heads
        batch_size = xs[0].size(0)
        gsizes = [list(output[1].shape[2:4]) for output in outputs]
        tbboxes, tclasss, tidents = self._build_targets(batch_size, targets, gsizes)
        
        # select predictions and truths based on object mask
        masks = [tc > 0 for tc in tclasss]                                          # n*a*h*w
        obj_masks = [mask.max(1)[0] for mask in masks]                              # n*h*w
        tidents = [tident.max(1)[0] for tident in tidents]                          # n*h*w
        tidents = [tident[m] for tident, m in zip(tidents, obj_masks)]              # x
        pembeds = [pembed[m] for pembed, m in zip(pembeds, obj_masks)]              # x*512
        pembeds = [self.embd_scale * pembed for pembed in pembeds]                  # x*512
        
        # compute losses
        lbboxes = [self.FloatTensor([0])] * len(pbboxes)
        for i, (pb, tb, m) in enumerate(zip(pbboxes, tbboxes, masks)):
            if m.sum() > 0:
                lbboxes[i] = self.bbox_lossf(pb[m], tb[m])                          # x*4
        lclasss = [self.clas_lossf(pc, tc) for pc, tc in zip(pclasss, tclasss)]
        lidents = [self.FloatTensor([0])] * len(pembeds)
        for i, (pembed, tident) in enumerate(zip(pembeds, tidents)):
            if pembed.size(0) > 0:
                pident = self.classifier(pembed).contiguous()                       # x*num_idents
                lidents[i] = self.iden_lossf(pident, tident)
        
        # total loss
        loss = self.FloatTensor([0])
        for lb, lc, li in zip(lbboxes, lclasss, lidents):
            loss += (torch.exp(-self.sb) * lb + torch.exp(-self.sc) * lc + \
                torch.exp(self.si) * li + self.sb + self.sc + self.si) * 0.5
        
        # just for log        
        metrics = []
        for lb, lc, li in zip(lbboxes, lclasss, lidents):
            metrics.append({'LBOX':lb.detach().cpu().item(),
                'LCLA':lc.detach().cpu().item(), 'LIDE':li.detach().cpu().item()})
        
        return loss, metrics
    
    def _build_targets(self, n, targets, gsizes):
        '''build training targets.
        
        Args:
            n: batch size
            targets: training targets. the targets is [
                [batch, category, identifier, x, y, w, h], ...]
            gsizes: backbone output tensor grid sizes. gsizes are [
                [height, width], ...]
        Returns:
            tbboxes: truth boundding boxes. the shape is n*a*h*w*4
            tclasss: truth confidences. the shape is n*a*h*w
            tidents: truth identifiers. the shape is n*a*h*w*1
        '''
        a = self.anchor_masks.size(1)        
        tbboxes = [self.FloatTensor(n, a, h, w, 4).fill_(0) for h, w in gsizes]
        tclasss = [self.LongTensor(n, a, h, w).fill_(0) for h, w in gsizes]
        tidents = [self.LongTensor(n, a, h, w).fill_(-1) for h, w in gsizes]
        if targets.size == 0:
            return tbboxes, tclasss, tidents

        batch, identifier, xywh = targets[:, 0].long(), targets[:, 2].long(), targets[:, 3:]
        for layer, (amask, (h, w)) in enumerate(zip(self.anchor_masks, gsizes)):
            # ground truth boundding boxes
            xywh = targets[:, 3:] * self.FloatTensor([w, h, w, h])
            xywh[:, 0] = torch.clamp(xywh[:, 0], min=0, max=w-1)
            xywh[:, 1] = torch.clamp(xywh[:, 1], min=0, max=h-1)                    # (t1+t2+...+tn)*4
            
            # IOU between anchor boxes and ground truths
            anchors = self.anchors[amask, :] / (self.in_size[0] / h)                # a*2
            anchor_boxes = self._make_anchor_boxes(h, w, anchors).view(-1, 4)       # (a*h*w)*4
            ious = [self._xywh_iou(anchor_boxes, xywh[batch==i]) for i in range(n)] # (a*h*w)*ti            
            vis = [self._iou_max(iou) for iou in ious]                              # (a*h*w)
            values  = torch.stack([vi[0].view(a, h, w) for vi in vis])              # n*a*h*w
            indices = torch.stack([vi[1].view(a, h, w) for vi in vis])              # n*a*h*w
            
            # foreground and background selecting and ignoring masks
            keep_iden = values > self.iden_thresh                                   # n*a*h*w
            keep_frgn = values > self.frgd_thresh
            keep_bkgn = values < self.bkgd_thresh
            ignore    = (values > self.bkgd_thresh) & (values < self.frgd_thresh)
            
            # set object confidence truths
            tclasss[layer][keep_frgn] =  1
            tclasss[layer][keep_bkgn] =  0
            tclasss[layer][ignore]    = -1

            # every image has different number of matched boxes and identifiers
            for i in range(n):
                idens = identifier[batch==i][indices[i][keep_iden[i]]]
                tidents[layer][i][keep_iden[i]] = idens
                tb = xywh[batch==i][indices[i][keep_frgn[i]]]                       # ti*4
                ab = anchor_boxes.view(a, h, w, 4)[keep_frgn[i]]                    # ti*4
                tbboxes[layer][i][keep_frgn[i]] = self._encode_bbox(tb, ab)

        return tbboxes, tclasss, tidents
    
    def _make_anchor_boxes(self, h, w, anchors):
        '''make anchor boxes.
        
        Args:
            h: the row number of anchor boxes
            w: the column number of anchor boxes
        Returns:
            xywh: h by w number of anchor boxes
        '''
        a = self.anchor_masks.size(1)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))                 # h*w
        xy = torch.stack([x, y])                                                # 2*h*w
        xy = xy.unsqueeze(0).repeat(a, 1, 1, 1).type(self.FloatTensor)          # a*2*h*w
        wh = anchors.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)             # a*2*h*w
        xywh = torch.cat([xy, wh], dim=1).permute(0, 2, 3, 1).contiguous()      # a*h*w*4
        return xywh
    
    def _xywh_iou(self, A, B, eps=1e-10):
        '''calculate intersection over union (IOU) between boxes A and boxes B.
            
        Args:
            A: the format of boxes is [[x, y, w, h], ...]
            B: the format of boxes is [[x, y, w, h], ...]
        Returns:
            ious: the IOU matrix between A and B
        '''
        m, n = A.size(0), B.size(0)
        t1, l1 = A[:, 1] - A[:, 3] / 2, A[:, 0] - A[:, 2] / 2   # m
        b1, r1 = A[:, 1] + A[:, 3] / 2, A[:, 0] + A[:, 2] / 2   # m
        t2, l2 = B[:, 1] - B[:, 3] / 2, B[:, 0] - B[:, 2] / 2   # n
        b2, r2 = B[:, 1] + B[:, 3] / 2, B[:, 0] + B[:, 2] / 2   # n
        
        t, l = torch.max(t1.unsqueeze(1), t2), torch.max(l1.unsqueeze(1), l2)   # m*n
        b, r = torch.min(b1.unsqueeze(1), b2), torch.min(r1.unsqueeze(1), r2)   # m*n
        w, h = torch.clamp(r - l, min=0), torch.clamp(b - t, min=0)
        
        inter = w * h
        area1 = (A[:, 2] * A[:, 3]).view(-1, 1).expand(m, n)
        area2 = (B[:, 2] * B[:, 3]).view(1, -1).expand(m, n)
        
        return inter / (area1 + area2 - inter + eps)
    
    def _iou_max(self, iou, dim=1):
        '''find the maximum values and corresponding indices along
            specific dimension of the iou matrix. the columns of
            the iou matrix can be zero.
        
        Args:
            iou: intersection over union matrix.
        Returns:
            (values, indices): the maximum values and corresponding
                indices
        '''
        if iou.size(1) > 0:
            return iou.max(dim)
        else:
            values = self.FloatTensor(iou.size(0),).fill_(0)
            indices = self.LongTensor(iou.size(0),).fill_(False)
            return (values, indices)

    def _encode_bbox(self, tbboxes, anchor_boxes):
        '''encode the truth boxes as expecting prediction output.
        
        Args:
            tbboxes: truth boxes which have the format [x, y, w, h]
            anchor_boxes: anchor boxes witch have the same format as tbboxes
        Returns:
            encoded_boxes: encoded boxes
        '''
        tx, ty, tw, th = tbboxes.t()
        px, py, pw, ph = anchor_boxes.t()
        ex = (tx - px) / pw
        ey = (ty - py) / ph
        ew = torch.log(tw / pw)
        eh = torch.log(th / ph)
        return torch.stack([ex, ey, ew, eh], dim=1)