import torch
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
            raise ValueError('when stride is 1, the input and output'
                ' channels must be equal')
        
        self.block_in_channels = block_in_channels
        self.block_out_channels = block_out_channels
        self.stride = stride
        out_channels = block_out_channels // 2
        in_channels = block_in_channels if stride == 2 else out_channels
        self.major_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1,
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        if stride == 1:
            return

        self.minor_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
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
    stage_repeat      : Number of building blocks for each stage.
                        Default: ShuffleNetV2.0.5x configuration.
    stage_out_channels: Output channels for each stage.
                        Default: ShuffleNetV2.0.5x configuration.
    with_head         : Building model with classifier head or not.
                        Default: False.
    num_class         : Number of classes if set with_head as True.
                        Default: 1000.
    pretrained        : Pretrained model path. Currently only support models from
                        https://github.com/megvii-model/ShuffleNet-Series.git
                        Default: None.
    '''
    def __init__(self, stage_repeat={'stage2': 4, 'stage3': 8, 'stage4': 4},
        stage_out_channels={'conv1': 24, 'stage2': 48, 'stage3': 96,
        'stage4': 192, 'conv5': 1024}, with_head=False, num_class=1000,
        pretrained=None, output_modules=['stage2', 'stage3', 'stage4']):
        super(ShuffleNetV2, self).__init__()
        
        in_channels = 3
        out_channels = stage_out_channels['conv1']
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for stage, repeat in stage_repeat.items():
            in_channels, out_channels = out_channels, stage_out_channels[stage]
            blocks = [ShuffleNetV2BuildBlock(in_channels, out_channels, 2)]
            for r in range(1, repeat):
                blocks.append(ShuffleNetV2BuildBlock(out_channels, out_channels, 1))
            setattr(self, stage, nn.Sequential(*blocks))
        
        self.with_head = with_head
        if with_head:        
            in_channels, out_channels = map(stage_out_channels.get,
                ['stage4', 'conv5'])
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
            
            self.globalpool = nn.AvgPool2d(kernel_size=7)
            self.fc = nn.Linear(out_channels, num_class, bias=False)
        
        self.output_modules = output_modules
        self._init_weights()
        if pretrained:
            self._load_pretrained_model(pretrained)
        
    def forward(self, input, *args, **kwargs):
        """ShuffleNetV2 forward"""
        outputs = []
        for name, module in self.named_children():
            input = module(input)
            if name in self.output_modules:
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