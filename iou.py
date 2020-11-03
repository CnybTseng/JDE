import torch
from torch import nn

class DIOULoss(nn.Module):
    '''Initialize distance IOU loss module.
    
    Param
    -----
    reduction: Specifies the reduction to apply to the output:
              'none' | 'mean' | 'sum'. 'none': no reduction will be applied. 'mean':
              the sum of output will be divided by the number of elements in the output.
              'sum': the output will be summed. Default: 'mean'.
    '''
    def __init__(self, reduction='mean'):
        super(DIOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        '''Distance IOU loss forward propagation.
        
        Param
        -----
        input : Input boxes with format [[x, y, w, h], ...]
        target: Target boxes with the same format as input. The number of input boxes
                must be equal to the number of boxes of target boxes.
        
        Return
        The distance IOU loss between input and target boxes.
        ------
        '''
        assert(input.size() == target.size())
        
        # IOUs between input and target boxes.
        input_xyxy = self._xywh2xyxy(input)
        target_xyxy = self._xywh2xyxy(target)
        ious = self._iou(input, target, A1=input_xyxy, B1=target_xyxy)
        
        # Calculate the diagonal length of the smallest enclosing box.
        enc_box = self._enclose_box(input_xyxy, target_xyxy)
        enc_w = (enc_box[:, 2] - enc_box[:, 0])
        enc_h = (enc_box[:, 3] - enc_box[:, 1])
        diag_leng = enc_w * enc_w + enc_h * enc_h
        
        # Calculate the Euclidiean distance between the central points of input and target boxes.
        cent_dx = input[:, 0] - target[:, 0]
        cent_dy = input[:, 1] - target[:, 1]
        cent_dist = cent_dx * cent_dx + cent_dy * cent_dy
        
        # The distance IOU loss.
        loss = 1 - ious + cent_dist / diag_leng
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss
    
    def _xywh2xyxy(self, xywh):
        '''Transform the box format from xywh to xyxy.
        
        Param
        -----
        xywh: XYWH format boxes. The (x, y) is the box center and the (w, h) is the
              width and height of box respectively.
        
        Return
        ------
        XYXY format boxes. Every returned box is arranged as [left, top, right, bottom].
        '''
        hw, hh = xywh[:, 2] / 2,  xywh[:, 3] / 2
        x1, y1 = xywh[:, 0] - hw, xywh[:, 1] - hh
        x2, y2 = xywh[:, 0] + hw, xywh[:, 1] + hh
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _iou(self, A, B, A1=None, B1=None, eps=1e-12):
        '''Calculate IOUs between boxes A and boxes B. The number of boxes A
           must be equal to the number of boxes B.
        
        Param
        A  : Boxes A with format [[x, y, w, h], ...].
        B  : Boxes B with format [[x, y, w, h], ...].
        A1 : The same boxes as A but with format [[x1, y1, x2, y2], ...].
        B1 : The same boxes as B but with format [[x1, y1, x2, y2], ...].
        eps: Small value to avoid division by zero. Default: 1e-12.
        
        Return
        ------
        IOUs between boxes A and boxes B.
        '''
        area_a, area_b = A[:, 2] * A[:, 3], B[:, 2] * B[:, 3]
        if A1 is None:
            A1 = self._xywh2xyxy(A)
        if B1 is None:
            B1 = self._xywh2xyxy(B)
        li, ri = torch.max(A1[:, 0], B1[:, 0]), torch.min(A1[:, 2], B1[:, 2])
        ti, bi = torch.max(A1[:, 1], B1[:, 1]), torch.min(A1[:, 3], B1[:, 3])
        wi, hi = torch.clamp(ri - li, 0), torch.clamp(bi - ti, 0)
        area_i = wi * hi 
        return area_i / (area_a + area_b - area_i + eps)
    
    def _enclose_box(self, A, B):
        '''Calculate the smallest enclosing box covering two boxes.
        
        Param
        A  : Boxes A with format [[x1, y1, x2, y2], ...].
        B  : Boxes B with format [[x1, y1, x2, y2], ...].
        
        Return
        ------
        Enclosing boxes covering boxes A and boxes B with the same format.
        '''
        l = torch.min(A[:, 0], B[:, 0])
        r = torch.max(A[:, 2], B[:, 2])
        t = torch.min(A[:, 1], B[:, 1])
        b = torch.max(A[:, 3], B[:, 3])
        return torch.stack([l, t, r, b], dim=1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    def _decode_box(box, anchor, gx, gy, gw, gh, im_size):
        aw, ah = anchor
        dbox = torch.zeros_like(box, requires_grad=True)
        dbox[0, 0] = box[0, 0] * aw + gx * im_size[1] / gw  # x
        dbox[0, 1] = box[0, 1] * ah + gy * im_size[0] / gh  # y
        dbox[0, 2] = torch.exp(box[0, 2]) * aw              # w
        dbox[0, 3] = torch.exp(box[0, 3]) * ah              # h
        return dbox
    
    width = 576
    height = 320    
    lr = 1
    diou = DIOULoss()
    stride = 32
    anchor = [6, 16]
    gx, gy = 280, 170
    gw, gh = width / stride, height / stride
    im_size = [height, width]
    
    init = torch.FloatTensor([280, 170, 1, 1]).view(-1, 4) / stride
    print('init: {}'.format(init))
    
    gt = torch.FloatTensor([288, 160, 16, 16]).view(-1, 4) / stride
    print('ground truth: {}'.format(gt))
       
    pred = init
    criterion = DIOULoss()
    iou_diou = []
    for epoch in range(10000):
        pred.requires_grad_(True)
        pred = _decode_box(pred, anchor, gx, gy, gw, gh, im_size)
        loss = criterion(pred, gt)
        loss.backward()
        with torch.no_grad():
            pred = pred - lr * pred.grad
            iou_diou.append(diou._iou(pred, gt))
            print('epoch: {}, prediction: {}, loss: {}'.format(epoch, pred, loss))
    
    pred = init
    criterion = nn.SmoothL1Loss()
    iou_sml1 = []
    for epoch in range(10000):
        pred.requires_grad_(True)
        pred = _decode_box(pred, anchor, gx, gy, gw, gh, im_size)
        loss = criterion(pred, gt)
        loss.backward()
        with torch.no_grad():
            pred = pred - lr * pred.grad
            iou_sml1.append(diou._iou(pred, gt))
            print('epoch: {}, prediction: {}, loss: {}'.format(epoch, pred, loss))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('DIOULoss(red) VS SmoothL1Loss(blue)')
    ax.set_xlabel('iteration')
    ax.set_ylabel('iou')
    ax.plot(iou_diou, 'r-')
    ax.plot(iou_sml1, 'b-')
    fig.savefig('converge.png', dpi=1200)