import math
import torch
from torch import nn

class JDE(nn.Module):
    def __init__(anchors, num_idents, num_classes=1, embd_dim=512):
        super(JDE, self).__init__()
        self.anchors = torch.FloatTensor(anchors)
        self.num_idents = num_idents
        self.num_classes = num_classes
        self.embd_dim = embd_dim
        self.sbbox = nn.Parameter(torch.FloatTensor([-4.85]))
        self.sclas = nn.Parameter(torch.FloatTensor([-4.15]))
        self.sembd = nn.Parameter(torch.FloatTensor([-2.30]))
        self.scale = math.sqrt(2) * math.log(num_idents - 1) if num_idents > 1 else 1
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, input, insize, targets=None, classifier=None):
        n, c, h, w = input.size()
        num_anchors = self.anchors.size(0)
        
        xywhc = input[:, :24, ...]
        xywhc = xywhc.view(n, num_anchors, 5 + self.num_classes, h, w)  # NACHW
        xywhc = xywhc.permute(0, 1, 3, 4, 2).contiguous()   # NAHWC        
        pbbox = xywhc[..., :4]  # NAHWC
        pconf = xywhc[..., 4:]  # NAHWC
        
        embed = input[:, 24:, ...]
        embed = embed.permute(0, 2, 3, 1)   # NHWC
        
        if self.training:
            assert targets is not None 'training need targets!'
            assert classifier is not None 'training need classifier'
        else:
            pconf = torch.softmax(pconf, dim=-1)[...,[1,0]]          
                      
            gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w))        
            gx = gx.view(1, 1, h, w)
            gy = gy.view(1, 1, h, w)                
            
            aw = anchors[:, 0].view(1, num_anchors, 1, 1)
            ah = anchors[:, 1].view(1, num_anchors, 1, 1)
            px = (pbbox[..., 0] * aw + gx * insize[1] / w)
            py = (pbbox[..., 1] * ah + gy * insize[0] / h)        
            pw = torch.exp(pbbox[..., 2]) * aw
            ph = torch.exp(pbbox[..., 3]) * ah            
            
            bbox = torch.stack([px, py, pw, ph], dim=-1)
            xywhc = torch.cat()
            
            return bbox, xywhc, embed