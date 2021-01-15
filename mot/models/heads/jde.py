import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from mot.models.builder import (HEADS, LOSSES, BLOCKS)
from mot.models.losses import DIOULoss

@HEADS.register_module()
class JDEHead(nn.Module):
    '''JDEHead.
    
    Param
    -----
    build_block : JDE head building block. 'ShuffleNetV2BuildBlock' or
                  'Conv2d'. Default: 'ShuffleNetV2BuildBlock'. The
                  build_block will be used by both detect and identify
                  branchs.
    block_repeat: Building block repeats. Default: {'detect':3, 'identify':3}.
    block_args  : Building arguments.
    num_ide     : Number of identifiers.
    anchor      : Anchor parameters.
    num_class   : Number of classes. Only support one class tracking.
    embd_dim    : Embedding dimension. Default: 128.
    box_loss    : Box loss function. 'DIOULoss' or 'SmoothL1Loss'.
                  Default: 'DIOULoss'.
    ide_thresh  : Identification positive threshold. Default: 0.5.
    obj_thresh  : Objectness positive threshold. Default: 0.5.
    bkg_thresh  : Background positive threshold. Default: 0.4.
    s_box       : Weight for the box regression task.
    s_cls       : Weight for the classification task.
    s_ide       : Weight for the identifier classification task.
    im_size     : Neural network input size.
    '''
    def __init__(self,
        build_block='ShuffleNetV2BuildBlock',
        block_repeat={'detect':3, 'identify':3},
        block_args=[128, 128, 1, {}],
        num_ide=0,
        anchor=[[[85, 255], [120, 360], [170, 420], [340, 320]],
                [[21, 64], [30, 90], [43, 128], [60, 180]],
                [[6, 16], [8, 23], [11, 32], [16, 45]]],
        num_class=1,
        embd_dim=128,
        box_loss='DIOULoss',
        ide_thresh=0.5,
        obj_thresh=0.5,
        bkg_thresh=0.4,
        s_box=[0, 0, 0],
        s_cls=[0, 0, 0],
        s_ide=[0, 0, 0],
        im_size=[320, 576]):
        super(JDEHead, self).__init__()
        
        self.box_dim = 4
        self.class_dim = num_class + 1
        self.embd_dim = embd_dim
        block = BLOCKS.get(build_block)
        self.anchor = torch.from_numpy(np.array(anchor, dtype=np.float32))
        m, n, _ = self.anchor.size()
        out_channels = {
            'detect': n * (self.box_dim + self.class_dim),
            'identify' : embd_dim}
        self.detect = nn.ModuleList()
        self.identify = nn.ModuleList()
        for i in range(m):
            for branch, repeat in block_repeat.items():
                blocks = [block(*block_args[:-1], **block_args[-1])
                    for r in range(repeat)]
                blocks.append(nn.Conv2d(blocks[-1].out_channels,
                    out_channels[branch], kernel_size=3, padding=1))
                blocks = nn.Sequential(*blocks)
                self.__dict__['_modules'][branch].append(blocks)
        self.classifier = nn.Sequential()
        if num_ide > 0:
            self.classifier = nn.Linear(embd_dim, num_ide)
        self._init_weights()

        self.ide_scale = 1
        if num_ide > 1:
            self.ide_scale = math.sqrt(2) * math.log(num_ide - 1)
        self.lf_box = LOSSES.get(box_loss)()
        self.lf_cls = nn.CrossEntropyLoss(ignore_index=-1)
        self.lf_ide = nn.CrossEntropyLoss(ignore_index=-1)
        self.ide_thresh = ide_thresh
        self.obj_thresh = obj_thresh
        self.bkg_thresh = bkg_thresh
        self.s_box = nn.Parameter(torch.FloatTensor(s_box))
        self.s_cls = nn.Parameter(torch.FloatTensor(s_cls))
        self.s_ide = nn.Parameter(torch.FloatTensor(s_ide))
        self.im_size = im_size
        
    def forward(self, input, target=None, im_size=None):
        '''JDEHead forward.
        
        Param
        -----
        input  : Tuple of input tensor with type torch.FloatTensor.
        target : Target tensor with type torch.FloatTensor. Only used in
                 training mode.
        im_size: Neural network input size with type list. Only used in
                 training mode.
        
        Return
        ------
        In training model, loss and metrics will be returned. In evaluation
        mode, decoded results will be returned.
        '''
        assert(isinstance(input, tuple))
        outputs = []
        for i, inp in enumerate(reversed(input)):
            det = self.detect[i](inp)
            ide = self.identify[i](inp)
            outputs.append(torch.cat([det, ide], dim=1))
        if self.training:
            assert(isinstance(target, torch.Tensor))
            if im_size is None:
                im_size = self.im_size
            else:
                if isinstance(im_size, list):
                    pass
                elif isinstance(im_size, torch.Tensor):
                    im_size = im_size.cpu().numpy().tolist()
                else:
                    raise TypeError('expect list or torch.Tensor,'
                        ' but got {}'.format(type(im_size)))
            return self._forward_train(outputs, target, im_size)
        else:
            return self._forward_eval(outputs)
    
    def _init_weights(self):
        """Initialize weights"""
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
    
    def _forward_train(self, input, target, im_size):
        '''JDELoss forward propagation.
        
        Param
        -----
        input  :    List of JDELoss input tensors.
        target :    Training target tensor. The format of target tensor is
                    [[batch id, class id, identity, x, y, w, h], ...].
        im_size:    Neural network input image size with format [height, width].
        classifier: Identify classifier.
        '''
        self.im_size = im_size
        loss_box, loss_cls, loss_ide = [], [], []
        for inp, anchor in zip(input, self.anchor):
            n, c, h, w = inp.size()
            stride = im_size[0] / h
            anc = (anchor / stride).type(inp.type()) # Scale anchors to match grid size
            num_anchor = anc.size(0)
            det_dim = num_anchor * (self.box_dim + self.class_dim)
            
            # Bounding box and class probability predictions.
            det = inp[:, : det_dim, ...]
            det = det.view(n, num_anchor, self.box_dim + self.class_dim, h, w)          # NACHW
            box = det[:, :, : self.box_dim, ...].permute(0, 1, 3, 4, 2).contiguous()    # NAHWC
            if isinstance(self.lf_box, DIOULoss):
                box = self._decode_box(box, anchor, w, h)
            cls = det[:, :, self.box_dim: , ...].permute(0, 2, 1, 3, 4).contiguous()    # NCAHW
            
            # Identify embedding predictions.
            ide = inp[:, det_dim :, ...]
            ide = F.normalize(ide).permute(0, 2, 3, 1).contiguous() # NHWC

            # Build ground truth based on target.
            gt_box, gt_cls, gt_ide = self._build_ground_truth(inp, target, im_size, anc)

            # Bounding boxes loss.
            obj_mask = gt_cls > 0                   # NAHW
            loss_box.append(torch.FloatTensor([0]).type(inp.type()))
            if obj_mask.sum() > 0:
                loss_box[-1] = self.lf_box(box[obj_mask], gt_box[obj_mask])
                
            # Class probability loss.
            loss_cls.append(self.lf_cls(cls, gt_cls))
            
            # Identify loss.
            ide_mask = obj_mask.max(dim=1)[0]       # NHW, collapse along anchor dimension
            gt_ide = gt_ide.max(dim=1)[0]           # NHW, collapse along anchor dimension
            gt_ide = gt_ide[ide_mask]               # T
            ide = self.ide_scale * ide[ide_mask]    # T[128]
            loss_ide.append(torch.FloatTensor([0]).type(inp.type()))
            if ide.size(0) > 0:
                ide = self.classifier(ide)          # T[num_ide]
                loss_ide[-1] = self.lf_ide(ide, gt_ide)
        
        # Automatic loss balancing.
        loss = []
        for i in range(len(loss_box)):
            lbox = torch.exp(-self.s_box[i]) * loss_box[i]
            lcls = torch.exp(-self.s_cls[i]) * loss_cls[i]
            lide = torch.exp(-self.s_ide[i]) * loss_ide[i]
            loss.append((lbox + lcls + lide + self.s_box[i] + self.s_cls[i] + self.s_ide[i]) * 0.5)
        loss = sum(loss)
        
        # Make up log information.
        metrics = {
            'SIZE': '{}x{}'.format(im_size[0], im_size[1]),
            'LBOX': sum([l.detach().cpu().item() for l in loss_box]),
            'LCLS': sum([l.detach().cpu().item() for l in loss_cls]),
            'LIDE': sum([l.detach().cpu().item() for l in loss_ide]),
            'LOSS': loss.detach().cpu().item(),
            'SBOX': self.s_box[0].detach().cpu().item(),
            'SCLS': self.s_cls[0].detach().cpu().item(),
            'SIDE': self.s_ide[0].detach().cpu().item()}

        return loss, metrics
    
    def _forward_eval(self, input):
        '''JDEcoder forward propagation.
        
        Param
        -----
        input: List of JDEcoder input tensors.
        
        Return
        ------
        Decoded JDE output including bounding boxes and embeddings.
        The shape of returned tensor is (N, M, box_dim+class_dim+embd_dim).
        N is the batch size, and M is the total number of detection outputs.
        For example, if input tensor shapes are (N,152,10,18), (N,152,20,36),
        and (N,152,40,72), embd_dim is 128, M will be (10*18+20*36+40*72)*4.
        '''
        output = []
        for inp, anc in zip(input, self.anchor):
            output.append(self._decode(inp, anc))
        return torch.cat(output, dim=1).detach().cpu()
    
    def _decode(self, input, anchor):
        '''Decode JDE output.
        
        Param
        -----
        input : Encoded input tensor.
        anchor: Anchors with format [[width, height], ...].
        
        Return
        ------
        Decoded output.
        '''
        n, c, h, w = input.size()
        num_anchor = anchor.size(0)
        det_dim = num_anchor * (self.box_dim + self.class_dim)
        
        # Decode object detection output.
        det = input[:, : det_dim, ...]
        det = det.view(n, num_anchor, self.box_dim + self.class_dim, h, w)
        det = det.permute(0, 1, 3, 4, 2).contiguous()
        det[..., :self.box_dim] = self._decode_box(det[..., :self.box_dim], anchor, w, h)
        det[..., self.box_dim:] = torch.softmax(det[..., self.box_dim:], dim=-1)
        det[..., self.box_dim:] = det[..., self.box_dim:][..., [1, 0]]
        det[..., self.box_dim + 1] = 0
        det = det.view(n, -1, self.box_dim + self.class_dim)
        
        # Decode identity embedding output
        ide = input[:, det_dim :, ...]
        ide = F.normalize(ide)
        ide = ide.unsqueeze(1).repeat(1, num_anchor, 1, 1, 1)
        ide = ide.permute(0, 1, 3, 4, 2).contiguous()
        ide = ide.view(n, -1, self.embd_dim)

        return torch.cat([det, ide], dim=-1)
    
    def _decode_box(self, box, anchor, gw, gh):
        '''Decode boxes with following equations:
        x' = (x * (anchor_w / stride) + anchor_x) * stride, or
        x' = x * anchor_w + anchor_x * stride.
        y' = (y * (anchor_h / stride) + anchor_y) * stride, or
        y' = y * anchor_h + anchor_y * stride.
        w' = ((anchor_w / stride) * exp(w)) * stride, or
        w' = anchor_w * exp(w).
        h' = ((anchor_h / stride) * exp(h)) * stride, or
        h' = anchor_h * exp(h).
        Be carefully! The anchor have not been strided.
        
        Param
        -----
        box   : Predicted boxes with format [[x, y, w, h], ...].
        anchor: Anchor boxes with format [[w, h], ...].
        gw    : Feature tensor grid width.
        gh    : Feature tensor grid height.

        Return
        ------
        Decoded boxes.
        '''
        num_anchor = anchor.size(0)
        aw = anchor[:, 0].view(1, num_anchor, 1, 1).type(box.type())
        ah = anchor[:, 1].view(1, num_anchor, 1, 1).type(box.type())
        gy, gx = torch.meshgrid(torch.arange(gh), torch.arange(gw))
        gy = gy.view(1, 1, gy.size(0), gy.size(1)).type(box.type())
        gx = gx.view(1, 1, gx.size(0), gx.size(1)).type(box.type())
        box[..., 0] = box[..., 0] * aw + gx * self.im_size[1] / gw  # x
        box[..., 1] = box[..., 1] * ah + gy * self.im_size[0] / gh  # y
        box[..., 2] = torch.exp(box[..., 2]) * aw                   # w
        box[..., 3] = torch.exp(box[..., 3]) * ah                   # h        
        return box
    
    def _build_ground_truth(self, input, target, im_size, anchor):
        '''Build ground truth.
        
        Param
        -----
        input  : Input tensor.
        target : Training target tensor. The format of target tensor is
                 [[image id, class id, identity, x, y, w, h], ...].
        im_size: Neural network input image size with format [height, width].
        anchor : Anchor parameters with format [[w, h], ...].
        
        Return
        ------
        List of ground truth tensors including box, class probability,
        and identity.
        '''
        n, c, h, w = input.size()
        im_id = target[:, 0].long()
        identity = target[:, 2].long()
        a = anchor.size(0)
        
        # Initialize ground truth.
        box = torch.FloatTensor(n, a, h, w, self.box_dim).fill_(0).to(input.device)
        cls = torch.LongTensor(n, a, h, w).fill_(0).to(input.device)
        ide = torch.LongTensor(n, a, h, w).fill_(-1).to(input.device)
        
        # No object in current batch.
        if target.size(0) == 0:
            return box, cls, ide

        # Make target and anchor boxes and calculate IOU matrix between them.
        tg_box = self._make_target_box(target[:, 3:], w, h)                 # T4
        ac_box = self._make_anchor_box(anchor, w, h)                        # [AHW]4
        iou = self._iou_xywh(tg_box, ac_box)                                # T[AHW]
        val, ind = self._iou_max(iou, im_id, n, dim=0)                      # [AHW],[AHW],...
        max_iou = torch.stack([v.view(a, h, w) for v in val])               # NAHW
        tg_id = torch.stack([i.view(a, h, w) for i in ind])                 # NAHW
        
        # Calculate ground truth masks based on IOU.
        ide_mask = max_iou > self.ide_thresh                                # NAHW
        obj_mask = max_iou > self.obj_thresh
        bkg_mask = max_iou < self.bkg_thresh
        ignore = (~obj_mask) & (~bkg_mask)
        
        # Mask box ground truth.
        tg_box_left = tg_box[tg_id[obj_mask]]
        if isinstance(self.lf_box, nn.SmoothL1Loss):
            ac_box = ac_box.view(a, h, w, self.box_dim)                     # AHW4
            ac_box_left = []
            for i in range(n):
                ac_box_left.append(ac_box[obj_mask[i]])
            ac_box_left = torch.cat(ac_box_left, dim=0)
            box[obj_mask] = self._encode_box(tg_box_left, ac_box_left)
        else:
            box[obj_mask] = tg_box_left * im_size[0] / h
        
        # Mask class probability ground truth.
        cls[obj_mask] = 1
        cls[ignore] = -1
        
        # Mask identity ground truth.
        ide[ide_mask] = identity[tg_id[ide_mask]]
        
        return box, cls, ide
    
    def _make_target_box(self, norm_box, grid_w, grid_h):
        '''Make target boxes.
        
        Param
        -----
        norm_box: Normalized boxes with format [[x, y, w, h], ...].
        grid_w  : Feature tensor grid width.
        grid_h  : Feature tensor grid height.
        
        Return
        ------
        The target box matrix with format [[x, y, w, h], ...].
        '''
        x = torch.clamp(norm_box[:, 0] * grid_w, min=0, max=grid_w-1)
        y = torch.clamp(norm_box[:, 1] * grid_h, min=0, max=grid_h-1)
        w = norm_box[:, 2] * grid_w
        h = norm_box[:, 3] * grid_h
        return torch.stack([x, y, w, h], dim=-1)
    
    def _make_anchor_box(self, anchor, grid_w, grid_h):
        '''Make anchor boxes.
        
        Param
        -----
        anchor: Anchor parameters with format [[w, h], ...].
        grid_w: Feature tensor grid width.
        grid_h: Feature tensor grid height.
        
        Return
        ------
        The anchor box matrix with format [[x, y, w, h], ...].
        '''
        a = anchor.size(0)
        y, x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w))       # HW
        x = x.unsqueeze(0).repeat(a, 1, 1).unsqueeze(-1).type(anchor.type())    # AHW1
        y = y.unsqueeze(0).repeat(a, 1, 1).unsqueeze(-1).type(anchor.type())    # AHW1
        wh = anchor.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, grid_h, grid_w)    # A2HW
        wh = wh.permute(0, 2, 3, 1).contiguous()                                # AHW2
        return torch.cat([x, y, wh], dim=-1).view(-1, self.box_dim)
    
    def _iou_xywh(self, A, B, eps=1e-10):
        '''Calculate IOU matrix between boxes A and boxes B.
        
        Param
        -----
        A: Boxes with format [[x, y, w, h], ...].
        B: Boxes with format [[x, y, w, h], ...].
        
        Return
        ------
        The IOU matrix. If the number of boxes A and B is M and N
        respectively, the IOU matrix shape will be [M, N].
        '''
        m, n = A.size(0), B.size(0)
        t1, l1 = A[:, 0] - A[:, 2] / 2, A[:, 1] - A[:, 3] / 2
        b1, r1 = A[:, 0] + A[:, 2] / 2, A[:, 1] + A[:, 3] / 2
        t2, l2 = B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2
        b2, r2 = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
        area1 = (A[:, 2] * A[:, 3]).view(-1, 1).expand(m, n)
        area2 = (B[:, 2] * B[:, 3]).view(1, -1).expand(m, n)
        
        ti, li = torch.max(t1.unsqueeze(1), t2), torch.max(l1.unsqueeze(1), l2)
        bi, ri = torch.min(b1.unsqueeze(1), b2), torch.min(r1.unsqueeze(1), r2)
        wi, hi = torch.clamp(ri - li, 0), torch.clamp(bi - ti, 0)
        inter = wi * hi
        
        return inter / (area1 + area2 - inter + eps)
    
    def _iou_max(self, iou, im_id, batch_size, dim=0):
        '''Find the maximum IOUs between target and anchor boxes.
        
        Param
        -----
        iou       : IOU matrix between target and anchor boxes.
        im_id     : Image indices of target boxes.
        batch_size: Batch size.
        dim       : The dimension looking for the maximum IOUs.
        
        Return
        ------
        The maximum IOUs and corresponding indices
        along dimension /dim/.
        '''
        val, ind = [], []
        last_ind = 0
        for i in range(batch_size):
            iou_i = iou[im_id == i]
            if iou_i.size(0) > 0:
                val_i, ind_i = iou_i.max(dim)
                ind_i += last_ind   # Transform index in image to index in batch.
            else:   # No object in current image.
                val_i = torch.FloatTensor(iou_i.size(1),).fill_(0).to(iou.device)
                ind_i = torch.LongTensor(iou_i.size(1),).fill_(False).to(iou.device)
            val.append(val_i)
            ind.append(ind_i)
            last_ind += iou_i.size(0)
        return val, ind

    def _encode_box(self, tg_box, ac_box):
        '''Encode truth boxes with following equations:
        x = ((x' / stride) - anchor_x) / (anchor_w / stride).
        y = ((y' / stride) - anchor_y) / (anchor_h / stride).
        w = log((w' / stride) / (anchor_w / stride)).
        h = log((h' / stride) / (anchor_h / stride)).
        Be carefully! The tg_box and ac_box have been strided. Dividing
        by stride can be ignored.
        
        Param
        -----
        tg_box: Target boxes with format [[x, y, w, h], ...].
        ac_box: Anchor boxes with format [[x, y, w, h], ...].
        
        Return
        ------
        Encoded boxes.
        '''
        tx, ty, tw, th = tg_box.t()
        ax, ay, aw, ah = ac_box.t()
        px = (tx - ax) / aw
        py = (ty - ay) / ah
        pw = torch.log(tw / aw)
        ph = torch.log(th / ah)
        return torch.stack([px, py, pw, ph], dim=1)