# -*- coding: utf-8 -*-
# file: yolov3.py
# brief: JDE implementation based on PyTorch
# author: Zeng Zhiwei
# date: 2020/4/8

import torch
import torch.nn.functional as F

class YOLOv3Decoder(torch.nn.Module):
    def __init__(self, in_size, num_classes, anchors):
        super(YOLOv3Decoder, self).__init__()
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.in_size = in_size
        self.num_classes = num_classes
        self.anchors = self.FloatTensor(anchors)
        self.anchor_masks = self.LongTensor(((8,9,10,11),(4,5,6,7),(0,1,2,3)))
        
    def forward(self, xs):
        return [self.__decoder(x,self.anchors[mask,:]) for (x,mask) in zip(xs,self.anchor_masks)]
    
    def __decoder(self, x, anchors):
        size = anchors.size(0)*(5+self.num_classes)
        y = x[:,size:, ...]
        x = x[:,:size, ...]      
        
        n, c, h, w = x.size()
        num_anchors = self.anchor_masks.size(1)

        x = x.view(n, num_anchors, 5 + self.num_classes, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()        
        
        x[..., 4:6] = torch.softmax(x[..., 4:6], dim=-1)[...,[1,0]]
        x[..., 5] = 0
        
        y = y.permute(0, 2, 3, 1).unsqueeze(1).repeat(1, num_anchors, 1, 1, 1).contiguous()
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

class YOLOv3EvalDecoder(YOLOv3Decoder):
    def __init__(self, in_size, num_classes, anchors):
        super(YOLOv3EvalDecoder, self).__init__(in_size, num_classes, anchors)
    
    def forward(self, xs):
        dxs = super().forward(xs)
        boxes = [dx[0].view(dx[0].size(0), -1, 4) for dx in dxs]
        objes = [dx[1][...,4].view(dx[1].size(0), -1, 1) for dx in dxs]
        cpros = [dx[1][...,5:5+self.num_classes].view(dx[1].size(0), -1, self.num_classes) for dx in dxs]
        embds = [dx[1][...,5+self.num_classes:].view(dx[1].size(0), -1, 512) for dx in dxs]
        outputs = [torch.cat((b,o,c,e), dim=-1) for (b,o,c,e) in zip(boxes,objes,cpros,embds)]
        return torch.cat(outputs, dim=1).detach().cpu()

class YOLOv3Loss(YOLOv3Decoder):
    def __init__(self, num_classes, anchors):
        super(YOLOv3Loss, self).__init__((416,416), num_classes, anchors)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if int(torch.__version__.replace('.', '').split('+')[0]) >= 120:
            self.BoolTensor = torch.cuda.BoolTensor if torch.cuda.is_available() else torch.BoolTensor
        else:
            self.BoolTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
        self.ignore_thresh = 0.5
        
    def forward(self, xs, targets, in_size):
        self.in_size = in_size
        dxs = super().forward(xs)
                        
        iahwc_index, layer_index, txywho, tc = self.__parse_targets(dxs, targets, in_size)                
        layer_mask = [layer_index == i for i in range(self.anchor_masks.size(0))]        
        iahwc_indices = [iahwc_index[mask,:] for mask in layer_mask]                
        txywhos = [txywho[mask,:] for mask in layer_mask]
        tcs = [tc[mask,:] for mask in layer_mask]
                
        pxywhos = [dx[1][iahwc[:,0],iahwc[:,1],iahwc[:,2],iahwc[:,3],:5] for dx,iahwc in zip(dxs,iahwc_indices)]
        pcs     = [dx[1][iahwc[:,0],iahwc[:,1],iahwc[:,2],iahwc[:,3],5:] for dx,iahwc in zip(dxs,iahwc_indices)]
                
        boxes = [targets[mask,2:] for mask in layer_mask]
        ss = [torch.sqrt(2 - box[:,2]*box[:,3]) if box.numel() > 0 else self.FloatTensor([0]) for box in boxes]
        ss = [s.view(box.size(0), 1) if box.numel() > 0 else self.FloatTensor([0]) for s, box in zip(ss, boxes)]
        tbs, pbs = self.__calc_bkg_pt(dxs, targets[:,2:], iahwc_indices)
        
        lxywhs = [self.__sl1_loss(s*pxywho[:,:4], s*txywho[:,:4]) for s,pxywho,txywho in zip(ss,pxywhos,txywhos)]
        los = [self.__bce_loss(pxywho[:,4], txywho[:,4]) for pxywho,txywho in zip(pxywhos,txywhos)]
        lcs = [self.__bce_loss(pc, tc) for pc,tc in zip(pcs,tcs)]
        lbs = [self.__bce_loss(pb, tb) for pb,tb in zip(pbs,tbs)]
        loss = sum([lxywh+lo+lc+lb for lxywh,lo,lc,lb in zip(lxywhs,los,lcs,lbs)])
        
        metrics = self.__calc_metrics(dxs, boxes, iahwc_indices, pxywhos, pcs, pbs, lxywhs, los, lcs, lbs)                
        return loss, metrics
        
    def __parse_targets(self, dxs, targets, in_size):
        nobjs = targets.size(0)
        anchors = self.anchors / in_size[0]
        grid_sizes = [dx[1].size(2) for dx in dxs]
        iahwc_index = torch.zeros(size=(nobjs,5), dtype=torch.int64, device=self.device)
        txywho = torch.zeros(size=(nobjs, 5), dtype=torch.float32, device=self.device)
        tc = torch.zeros(size=(nobjs, self.num_classes), dtype=torch.float32, device=self.device)
        
        if nobjs == 0:
            layer_index = self.LongTensor([])
            return iahwc_index, layer_index, txywho, tc
        
        iahwc_index[:,0] = targets[:,0].type(self.LongTensor)
        iahwc_index[:,4] = targets[:,1].type(self.LongTensor)
        
        ious = torch.stack([self.__cal_IoU_lrca(box, targets[:,4:].t()) for box in anchors])
        values, indices = ious.max(dim=0)
        iahwc_index[:,1] = indices % self.anchor_masks.size(1)
        layer_index = self.anchor_masks.size(0) - 1 - indices / self.anchor_masks.size(1)
        
        bx = self.FloatTensor([targets[i,2] * grid_sizes[layer_id] for i, layer_id in enumerate(layer_index)])
        by = self.FloatTensor([targets[i,3] * grid_sizes[layer_id] for i, layer_id in enumerate(layer_index)])
        cx = bx.floor()
        cy = by.floor()
        iahwc_index[:,2] = cy.type(self.LongTensor)
        iahwc_index[:,3] = cx.type(self.LongTensor)
        
        txywho[:,0] = bx - cx
        txywho[:,1] = by - cy        
        txywho[:,2] = torch.log(targets[:,4]/anchors[indices,0])
        txywho[:,3] = torch.log(targets[:,5]/anchors[indices,1])        
        txywho[:,4] = 1
        
        tc[torch.arange(nobjs, dtype=torch.int64, device=self.device),iahwc_index[:,4]] = 1
        for i in range(nobjs):
            for j in range(i+1,nobjs):
                if iahwc_index[i,0] != iahwc_index[j,0]: break
                if not torch.equal(iahwc_index[i,2:4], iahwc_index[j,2:4]): continue
                mask_i = tc[i,:] == 1
                mask_j = tc[j,:] == 1
                tc[i, mask_j] = 1
                tc[j, mask_i] = 1
        
        return iahwc_index, layer_index, txywho, tc
    
    def __calc_bkg_pt(self, dxs, tboxes, iahwc_indices):
        sizes = [dx[0].size() for dx in dxs]
        tboxes = self.__xywh_to_xyxy(tboxes)
        bkg_masks = [self.BoolTensor(s[0], s[1], s[2], s[3]).fill_(1) for s in sizes]
        for (mask, index, dx, size) in zip(bkg_masks, iahwc_indices, dxs, sizes):
            if index.size(0) == 0: continue
            mask[index[:,0], index[:,1], index[:,2], index[:,3]] = 0            
            pboxes = self.__xywh_to_xyxy(dx[0].view(-1, 4))
            ious = torch.stack([self.__cal_IoU(tbox, pboxes.t()) for tbox in tboxes])
            values, indices = ious.max(dim=0)
            mask[(values > self.ignore_thresh).view(size[:4])] = 0
        
        tbs = [self.FloatTensor(s[0], s[1], s[2], s[3]).fill_(0)[m] for s, m in zip(sizes, bkg_masks)]
        pbs = [dx[1][m][:,4] for (dx, m) in zip(dxs, bkg_masks)]
        return tbs, pbs
    
    def __calc_metrics(self, dxs, tboxes, iahwc_indices, pxywhos, pcs, pbs, lxywhs, los, lcs, lbs):
        ccs = [(index[:,4]==pc.argmax(-1)).float() if pc.numel() else self.FloatTensor([0]) for index,pc in zip(iahwc_indices, pcs)]
        caccs = [cc.mean().detach().cpu().item() for cc in ccs]
        
        confs = [pxywho[:,4].mean().detach().cpu().item() if pxywho.numel() else .0 for pxywho in pxywhos]
        bkgcs = [pb.mean().detach().cpu().item() for pb in pbs]
        
        conf50pps = [(dx[1][...,4]>0.5).float().sum() for dx in dxs]
        conf50tps = [(pxywho[:,4]>0.5).float().sum() if pxywho.numel() else self.FloatTensor([0]) for pxywho in pxywhos]
        precs = [conf50tp/(conf50pp+1e-6) for conf50pp,conf50tp in zip(conf50pps, conf50tps)]
        precs = [prec.detach().cpu().item() for prec in precs]
        
        tboxes = [self.__xywh_to_xyxy(tbox) if tbox.numel() else None for tbox in tboxes]
        pboxes = [self.__xywh_to_xyxy(dx[0][ind[:,0],ind[:,1],ind[:,2],ind[:,3]]) if ind.numel() else None for dx,ind in zip(dxs, iahwc_indices)]
        ious = [self.__cal_IoU(tbox.t(), pbox.t()) if tbox is not None else self.FloatTensor([0]) for tbox,pbox in zip(tboxes,pboxes)]
        iou50s = [(iou > 0.50).float() for iou in ious]
        iou75s = [(iou > 0.75).float() for iou in ious]
        rc50s = [iou50.mean().detach().cpu().item() for iou50 in iou50s]
        rc75s = [iou75.mean().detach().cpu().item() for iou75 in iou75s]
        aious = [iou.mean().detach().cpu().item() for iou in ious]
        
        rows = [torch.arange(index.size(0), dtype=torch.int64, device=self.device) if index.numel() else None for index in iahwc_indices]
        acats = [pc[row,ind[:,4]].mean().detach().cpu().item() if row is not None else .0 for pc,row,ind in zip(pcs,rows,iahwc_indices)]
        
        metrics = list()
        for cacc,conf,bkgc,prec,rc50,rc75,aiou,acat in zip(caccs,confs,bkgcs,precs,rc50s,rc75s,aious,acats):
            metrics.append({'CACC':cacc,'CONF':conf,'BKGC':bkgc,'PREC':prec,'RC50':rc50,'RC75':rc75,'AIOU':aiou,'ACAT':acat})
        
        for (metric,lxywh,lo,lc,lb) in zip(metrics,lxywhs,los,lcs,lbs):
            metric['LBOX'] = lxywh.detach().cpu().item()
            metric['LOBJ'] = lo.detach().cpu().item()
            metric['LCLS'] = lc.detach().cpu().item()
            metric['LBKG'] = lb.detach().cpu().item()
        
        return metrics
    
    def __cal_IoU_lrca(self, box1, box2, eps=1e-16):
        intersection = torch.min(box1[0], box2[0]) * torch.min(box1[1], box2[1])
        area1 = box1[0] * box1[1]
        area2 = box2[0] * box2[1]        
        return intersection / (area1 + area2 - intersection + eps)
    
    def __cal_IoU(self, box1, box2, eps=1e-16):
        intersection = self.__cal_intersection_area(box1, box2)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])        
        return intersection / (area1 + area2 - intersection + eps)
        
    def __xywh_to_xyxy(self, xywh):
        xmin = xywh[:,0] - xywh[:,2]/2
        xmax = xywh[:,0] + xywh[:,2]/2
        ymin = xywh[:,1] - xywh[:,3]/2
        ymax = xywh[:,1] + xywh[:,3]/2
        return torch.stack((xmin,ymin,xmax,ymax), dim=1)
    
    def __cal_intersection_area(self, box1, box2):
        minx = torch.max(box1[0], box2[0])
        miny = torch.max(box1[1], box2[1])
        maxx = torch.min(box1[2], box2[2])
        maxy = torch.min(box1[3], box2[3])
        w = torch.max(maxx-minx, self.FloatTensor([0]))
        h = torch.max(maxy-miny, self.FloatTensor([0]))        
        return w * h
        
    def __mse_loss(self, input, target):
        assert input.numel() == target.numel()
        if input.numel() == 0 or target.numel() == 0:
            return self.FloatTensor([0]).requires_grad_()
        return torch.nn.MSELoss(reduction='sum')(input, target)
    
    def __sl1_loss(self, input, target):
        assert input.numel() == target.numel()
        if input.numel() == 0 or target.numel() == 0:
            return self.FloatTensor([0]).requires_grad_()
        return torch.nn.SmoothL1Loss(reduction='sum')(input, target)
    
    def __bce_loss(self, input, target, weight=None):
        assert input.numel() == target.numel()
        if input.numel() == 0 or target.numel() == 0:
            return self.FloatTensor([0]).requires_grad_()
        return torch.nn.BCELoss(weight=weight, reduction='sum')(input, target)