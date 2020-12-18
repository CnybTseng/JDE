import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_class=2,
        ignore_index=-100, reduction='mean'):
        super(SoftmaxFocalLoss, self).__init__()
        if not isinstance(alpha, (float, list)):
            raise TypeError('alpha expect float or list type,'
                ' but got {}'.format(type(alpha)))
        if isinstance(alpha, float):
            self.alpha = [1 - alpha]
            for c in range(num_class - 1):
                self.alpha += [alpha]
        else:
            assert(len(alpha) == num_class)
            self.alpha = alpha
        self.alpha = torch.FloatTensor(self.alpha)
        self.gamma = gamma
        self.num_class = num_class
        self.ignore_index = ignore_index
        assert(reduction in ['none', 'mean', 'sum'])
        self.reduction = reduction

    def forward(self, input, target):
        '''
        
        Param
        -----
        input:  (N,C,D1,D2,...,DK)
        target: (N,D1,D2,...,DK)
        '''        
        # Ensure that all target index are valid!
        ignore_mask = target == self.ignore_index
        target[ignore_mask] = 0
        
        # p_t
        p = F.softmax(input, dim=1)                                 # N,C,D1,D2,...,DK
        index = target.unsqueeze(dim=1)                             # N,1,D1,D2,...,DK
        p_t = torch.gather(p, 1, index).squeeze(dim=1)              # N,D1,D2,...,DK
        
        # alpha_t
        alpha_t = self.alpha[target.view(-1)].view(p_t.size())
        alpha_t = alpha_t.type(p_t.type())
        
        # Focal loss.
        weight = torch.pow(1 - p_t, self.gamma)
        eps = torch.finfo(torch.float32).tiny   # The smallest positive representable number
        loss = -alpha_t * weight * torch.log(torch.clamp(p_t, eps))  # N,D1,D2,...,DK
        loss[ignore_mask] = 0
        
        # Loss reduction.
        if self.reduction == 'mean':
            return loss.sum() / (~ignore_mask).sum()
        elif self.reduction == 'sum':
            return loss.sum()

        return loss

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-epo', type=int, default=20000,
        help='total epochs')
    parser.add_argument('--lr', '-lr', type=float, default=1000,
        help='learning rate')
    parser.add_argument('--method', '-m', type=int, default=0,
        help='loss method')
    args = parser.parse_args()

    print(args)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    if args.method == 0:    
        lf = SoftmaxFocalLoss(ignore_index=-1)
    elif args.method == 1:
        lf = torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        try:
            from mmcv.ops import SoftmaxFocalLoss as BaseSoftmaxFocalLoss
            lf = BaseSoftmaxFocalLoss(2, 0.25)
        except ImportError:
            import sys
            print('from mmcv.ops import SoftmaxFocalLoss ERROR!')
            sys.exit()
    
    x0 = torch.rand(64, 2, 4, 10, 18).cuda()
    t  = torch.randint(-1, 2, (64, 4, 10, 18)).cuda()
    x = x0
    for epoch in range(args.epochs):
        x.requires_grad_(True)
        if args.method == 2:
            loss = lf(x.permute(0, 2, 3, 4, 1).reshape(-1, 2), t.reshape(-1))
        else:
            loss = lf(x, t)
        loss.backward()
        with torch.no_grad():
            l = criterion(x, t)
            x = x - args.lr * x.grad
            print('epoch {}, loss {}, lr {}'.format(epoch, l, args.lr))