import torch
import warnings
from torch import nn

class ShuffleNetV2BuildBlock(nn.Module):
    '''ShuffleNetV2 building block. Currently only support half
       splitting when channel shuffling.
    
    Param
    -----
    block_in_channels : Block input channels.
    block_out_channels: Block output channels. 
    stride            : Spatial stride for ShuffleNetV2BuildBlock.
    '''
    def __init__(self, block_in_channels, block_out_channels, stride):
        super(ShuffleNetV2BuildBlock, self).__init__()
        
        if not stride in [1, 2]:
            raise ValueError('illegle stride value: {}'.format(stride))
        
        if stride == 1 and block_in_channels != block_out_channels:
            warnings.warn('for stride 1, block_in_channels !='
                ' block_out_channels violates Guide One!')
        
        if stride == 2 and block_in_channels >= block_out_channels:
            raise ValueError('when stride is 2, the output channels'
                ' must be greater than the input one')
        
        if stride == 2 and block_in_channels != block_out_channels // 2:
            warnings.warn('for stride 2, block_in_channels !='
                ' block_out_channels // 2 violates Guide One!')
        
        self.block_in_channels = block_in_channels
        self.block_out_channels = block_out_channels
        self.stride = stride
        in_channels = block_in_channels if stride == 2 else block_in_channels // 2
        out_channels = block_out_channels - in_channels
        mid_channels = block_out_channels // 2
        self.major_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        # Shortcut connection for stride one.
        if stride == 1:
            return
        
        # Keep channels unchanged for minor branch.
        self.minor_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1,
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
    
    def forward(self, input):
        if self.stride == 1:
            in1, in2 = self._channel_shuffle(input)
            return torch.cat((in1, self.major_branch(in2)), dim=1)
        else:
            return torch.cat((self.minor_branch(input),
                self.major_branch(input)), dim=1)
    
    def _channel_shuffle(self, input):
        n, c, h, w = input.data.size()
        assert c % 4 == 0
        # ABCDEF => AB;CD;EF
        input = input.reshape(n * c // 2, 2, h * w)
        # AB;CD;EF => ACE;BDF
        input = input.permute(1, 0, 2)
        input = input.reshape(2, -1, c // 2, h, w)
        return input[0], input[1]
    
    @property
    def in_channels(self):
        return self.block_in_channels
    
    @property
    def out_channels(self):
        return self.block_out_channels

class AG(nn.Module):
    '''Aggregation gate.
    
    Param
    -----
    in_channels: Gate input channels.
    reduction  : Hidden layer dimension reduction.
    activation : Hidden layer activation method.
        - 'sigmoid', default setting
        - 'relu'
        - 'linear'
    '''
    def __init__(self, in_channels, reduction=8,
        activation='sigmoid'):
        super(AG, self).__init__()
        if not isinstance(in_channels, int):
            raise TypeError('integer type expected, but got'
                ' {}'.format(type(in_channels)))
        if in_channels <= 0:
            raise ValueError('in_channels must be greater than'
                ' zero: {}'.format(in_channels))
        
        if not isinstance(reduction, int):
            raise TypeError('integer type expected, but got'
                ' {}'.format(type(reduction)))
        
        if reduction <= 0 or reduction > in_channels:
            raise ValueError('illegle reduction value:'
                ' {}'.format(reduction))
        
        if not isinstance(activation, str):
            raise TypeError('integer type expected, but got'
                ' {}'.format(type(activation)))
        
        if not activation in ['sigmoid', 'relu', 'linear']:
            raise ValueError('unsupported activation method:'
                ' {}'.format(activation))
        
        self._in_channels = in_channels
        self._out_channels = in_channels
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            # nn.LayerNorm((in_channels // reduction, 1, 1)),
            nn.ReLU(inplace=True))
        
        act = nn.Sequential()
        if activation == 'sigmoid':
            act = nn.Sigmoid()
        elif activation == 'relu':
            act = nn.ReLU(inplace=True)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            act)
    
    def forward(self, input):
        output = self.aap(input)
        output = self.fc1(output)
        output = self.fc2(output)
        return input * output
    
    @property
    def in_channels(self):
        return self._in_channels
    
    @property
    def out_channels(self):
        return self._out_channels

class SOSBlock(nn.Module):
    '''Shared and Omni-Scale Block.
    
    Param
    -----
    in_channels: SOSBlock input channels.
    '''
    def __init__(self, in_channels, out_channels, reduction=4):
        super(SOSBlock, self).__init__()        
        if not isinstance(in_channels, int):
            raise TypeError('integer type expected, but got'
                ' {}'.format(type(in_channels)))
        
        if in_channels <= 0:
            raise ValueError('in_channels must be greater than'
                ' zero: {}'.format(in_channels))
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        
        mid_channels = out_channels // reduction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
        self.sblk1 = ShuffleNetV2BuildBlock(mid_channels, mid_channels, 1)
        self.sblk2 = ShuffleNetV2BuildBlock(mid_channels, mid_channels, 1)
        self.sblk3 = ShuffleNetV2BuildBlock(mid_channels, mid_channels, 1)
        self.sblk4 = ShuffleNetV2BuildBlock(mid_channels, mid_channels, 1)
        
        # Shared aggregation gate.
        self.ag = AG(mid_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.resiudal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_channels)) \
        if in_channels != out_channels else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, input):
        """SOSBlock forward"""
        output1 = self.conv1(input)
        output1 = self.sblk1(output1)
        output2 = self.sblk2(output1)
        output3 = self.sblk3(output2)
        output4 = self.sblk4(output3)
        output  = self.ag(output1) + self.ag(output2) + \
            self.ag(output3) + self.ag(output4)
        output = self.conv2(output) + self.resiudal(input)
        return self.relu(output)
    
    @property
    def in_channels(self):
        return self._in_channels
    
    @property
    def out_channels(self):
        return self._out_channels

class SOSNet(nn.Module):
    '''Shared and Omni-Scale Network.
    
    Param
    -----
    arch     : SOSNet architecture.
        Default: SOSNet-0.5x
    with_head: Build network with classifier head or not.
        Default: False.
    num_class: Number of classes for classifier.
        Default: 1000
    '''
    def __init__(self, arch={
            'conv1':  {'out_channels': 16},
            'stage2': {'out_channels': 64, 'repeate': 2, 'out': True},
            'stage3': {'out_channels': 96, 'repeate': 2, 'out': True},
            'stage4': {'out_channels': 128, 'repeate': 2, 'out': True},
            'conv5':  {'out_channels': 1024}},
        with_head=False, num_class=1000):
        super(SOSNet, self).__init__()
        if not isinstance(arch, dict):
            raise TypeError('dict type expected, but got'
                ' {}'.format(type(arch)))
        
        if not isinstance(num_class, int):
            raise TypeError('integer type expected, but got'
                ' {}'.format(type(num_class)))
        
        if num_class <= 0:
            raise ValueError('num_class must be greater than'
                ' zero: {}'.format(num_class))
        
        self.arch = arch
        self.with_head = with_head
        self.num_class = num_class
        in_channels = 3
        out_channels = arch['conv1']['out_channels']
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        for stage in list(filter(lambda x: 'stage' in x, arch.keys())):
            in_channels, out_channels = out_channels, arch[stage]['out_channels']
            blocks = [ShuffleNetV2BuildBlock(in_channels, out_channels, 2)]
            for _ in range(arch[stage]['repeate']):
                blocks.append(SOSBlock(out_channels, out_channels))
            setattr(self, stage, nn.Sequential(*blocks))
        
        if with_head:
            in_channels = out_channels
            out_channels = arch['conv5']['out_channels']
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
            self.gap = nn.AvgPool2d(kernel_size=7)
            self.fc = nn.Linear(out_channels, num_class, bias=False)
            self.arch['fc'] = {'out_channels': num_class, 'out': True}
            for stage in list(filter(lambda x: 'stage' in x, arch.keys())):
                self.arch[stage]['out'] = False
        self._init_weights()
        
    def forward(self, input):
        """SOSNet forward"""
        outputs = []
        for name, module in self.named_children():
            if name == 'fc':
                input = input.contiguous().view(-1, input.shape[1])
            input = module(input)
            if name in self.arch.keys() and self.arch[name].get('out', False):
                outputs.append(input)
        return outputs[0] if self.with_head else outputs
    
    def _init_weights(self):
        """Initialize weights"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    @property
    def out_channels(self):
        _out_channels = {}
        for key, value in self.arch.items():
            if value.get('out', False):
                _out_channels[key] = value['out_channels']
        return _out_channels
    
if __name__ == '__main__':
    print('test SOSNet')
    x = torch.rand(64, 3, 224, 224)
    model = SOSNet(with_head=False)
    y = model(x)
    for yi in y:
        print(yi.shape)
    print(model.out_channels)