import torch
import warnings
from torch import nn
from mot.models.builder import BLOCKS
from mot.models.builder import BACKBONES

@BLOCKS.register_module()
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

@BACKBONES.register_module()
class ShuffleNetV2(nn.Module):
    '''ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.
    
    Param
    -----
    arch      : ShuffleNetV2 architecture.
        Default: ShuffleNetV2.0.5x.
    with_head : Building model with classifier head or not.
                Default: False.
    num_class : Number of classes if set with_head as True.
                Default: 1000.
    pretrained: Pretrained model path. Currently only support models from
                https://github.com/megvii-model/ShuffleNet-Series.git
                Default: None.
    '''
    def __init__(self, arch={
            'conv1':  {'out_channels': 24},
            'stage2': {'out_channels': 48, 'repeate': 4, 'out': True},
            'stage3': {'out_channels': 96, 'repeate': 8, 'out': True},
            'stage4': {'out_channels': 192, 'repeate':4, 'out': True},
            'conv5':  {'out_channels': 1024}},
            with_head=False, num_class=1000, pretrained=None):
        super(ShuffleNetV2, self).__init__()
        
        self.arch = arch
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
            for _ in range(1, arch[stage]['repeate']):
                blocks.append(ShuffleNetV2BuildBlock(out_channels, out_channels, 1))
            setattr(self, stage, nn.Sequential(*blocks))
        
        self.with_head = with_head
        if with_head:        
            in_channels = out_channels
            out_channels = arch['conv5']['out_channels']
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
            
            self.globalpool = nn.AvgPool2d(kernel_size=7)
            self.fc = nn.Linear(out_channels, num_class, bias=False)
            self.arch['fc'] = {'out_channels': num_class, 'out': True}
            for stage in list(filter(lambda x: 'stage' in x, arch.keys())):
                self.arch[stage]['out'] = False

        self._init_weights()
        if pretrained is not None:
            self._load_pretrained_model(pretrained)
        
    def forward(self, input, *args, **kwargs):
        """ShuffleNetV2 forward"""
        outputs = []
        for name, module in self.named_children():
            if name == 'fc':
                input = input.contiguous().view(-1, input.shape[1])
            input = module(input)
            if name in self.arch.keys() and self.arch[name].get('out', False):
                outputs.append(input)
        return tuple(outputs)
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                std = 0.01 if 'conv1' in name else 1.0 / module.weight.shape[1]
                nn.init.normal_(module.weight, mean=0, std=std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _load_pretrained_model(self, path):
        model = torch.load(path, map_location='cpu')
        state_dict = model['state_dict']
        params = []
        for key, val in state_dict.items():
            if (not self.with_head) and 'conv_last' in key:
                break
            params.append(val)
        
        i = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                print('load {}...'.format(name))
                assert(module.weight.size() == params[i].size())
                module.weight.data, i = params[i].data, i + 1
                if module.bias is not None:
                    assert(module.bias.size() == params[i].size())
                    module.bias.data, i = params[i].data, i + 1
            elif isinstance(module, nn.BatchNorm2d):
                print('load {}...'.format(name))
                assert(module.weight.size() == params[i].size())
                module.weight.data, i = params[i].data, i + 1
                assert(module.bias.size() == params[i].size())
                module.bias.data, i = params[i].data, i + 1
                assert(module.running_mean.size() == params[i].size())
                module.running_mean.data, i = params[i].data, i + 1
                assert(module.running_var.size() == params[i].size())
                module.running_var.data, i = params[i].data, i + 1
                assert(module.num_batches_tracked.size() == params[i].size())
                module.num_batches_tracked.data, i = params[i].data, i + 1
        assert(i == len(params))
    
    @property
    def out_channels(self):
        _out_channels = {}
        for key, value in self.arch.items():
            if value.get('out', False):
                _out_channels[key] = value['out_channels']
        return _out_channels

if __name__ == '__main__':
    x = torch.rand(64, 3, 224, 224)
    model = SOSNet(with_head=False)
    y = model(x)
    for yi in y:
        print(yi.shape)
    print(model.out_channels)