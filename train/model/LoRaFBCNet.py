import torch
import torch.nn as nn


__all__ = ['LoRaFBCNet', 'lorafb_cnet18', 'lorafb_cnet34', 'lorafb_cnet50', 'lorafb_cnet101', 'lorafb_cnet152']


class ConvRecurrent(nn.Module):
    def __init__(self, sub_module, in_channels, out_channels, stride):
        super().__init__()
        self.sub_module_out_channels = out_channels
        self.sub_module = sub_module
        
        if stride > 1:
            self.dwconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, 
                                             padding=1, output_padding=1, groups=out_channels, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.dwconv = None
        self.pwconv = nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x, y=None):
        if y is None:
            if self.dwconv is None:
                y = torch.zeros(x.size(0), self.sub_module_out_channels, x.size(2), x.size(3)).to(x)
            else:
                h = (x.size(2) + 2 - (3 - 1) - 1) // 2 + 1
                w = (x.size(3) + 2 - (3 - 1) - 1) // 2 + 1
                y = torch.zeros(x.size(0), self.sub_module_out_channels, h, w).to(x)
        if self.dwconv is None:
            out = y
        else:
            out = self.relu1(self.bn1(self.dwconv(y)))
        out = torch.cat((x, out), dim=1)
        out = self.bn(self.pwconv(out))
        out = self.relu2(out)
        out = self.sub_module(out)
        
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out


class LoRaFBCNet(nn.Module):
    def __init__(self, block, layers, num_classes=101, groups=1, width_per_groups=64, 
                 norm_layer=None, zero_init_residual=False, cnf=None):
        super(LoRaFBCNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_groups
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(block, 64, layers[0])
        self.recurrent_layer1 = ConvRecurrent(layer1, in_channels=64, out_channels=64 * block.expansion, stride=1)
        layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.recurrent_layer2 = ConvRecurrent(layer2, in_channels=64 * block.expansion, out_channels=128 * block.expansion, stride=2)
        layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.recurrent_layer3 = ConvRecurrent(layer3, in_channels=128 * block.expansion, out_channels=256 * block.expansion, stride=2)
        layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.recurrent_layer4 = ConvRecurrent(layer4, in_channels=256 * block.expansion, out_channels=512 * block.expansion, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.y = [None, None, None, None]
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, self._norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        times = x.size(0)
        h, w = x.size(3), x.size(4)
        
        out = x.contiguous().view(x.size(0) * x.size(1), x.size(2), h, w)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(times, -1, 64, h // 4, w // 4)
        
        outs = [out, [], [], [], []]
        for t in range(times):
            for i in range(1, 5):
                outs[i].append(getattr(self, f"recurrent_layer{i}")(outs[i - 1][t], self.y[i - 1]))
                self.y[i - 1] = outs[i][-1]
        
        out = torch.stack(outs[-1], dim=0)
        out = out.view(out.size(0) * out.size(1), out.size(2), out.size(3), out.size(4))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = out.view(times, -1, self.num_classes)
        
        return out
    
    def reset(self):
        self.y = [None, None, None, None]

        
def lorafb_cnet18(**kwargs):
    return LoRaFBCNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def lorafb_cnet34(**kwargs):
    return LoRaFBCNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def lorafb_cnet50(**kwargs):
    return LoRaFBCNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def lorafb_cnet101(**kwargs):
    return LoRaFBCNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def lorafb_cnet152(**kwargs):
    return LoRaFBCNet(Bottleneck, [3, 8, 36, 3], **kwargs)
