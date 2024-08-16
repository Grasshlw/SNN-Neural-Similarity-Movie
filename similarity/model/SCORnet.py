## Reproduce from https://github.com/dicarlolab/CORnet.git

from collections import OrderedDict
import torch
from torch import nn
from spikingjelly.activation_based import base, layer, neuron, surrogate


__all__ = ["s_cornet"]


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class SpikCORblock_RT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = layer.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = layer.GroupNorm(32, out_channels)
        self.nonlin_input = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.conv1 = layer.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = layer.GroupNorm(32, out_channels)
        self.nonlin1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            inp = inp.to(self.conv_input.weight.device)
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state

    
class SR_CORnet_RT(nn.Module):

    def __init__(self, num_classes=101):
        super().__init__()
        self.num_classes = num_classes

        self.V1 = SpikCORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = SpikCORblock_RT(64, 128, stride=2, out_shape=28)
        self.V4 = SpikCORblock_RT(128, 256, stride=2, out_shape=14)
        self.IT = SpikCORblock_RT(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', layer.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', layer.Linear(512, num_classes))
        ]))
        
        self.states = {'V1': 0, 'V2': 0, 'V4': 0, 'IT': 0}
        self.final_outputs = {'V1': None, 'V2': None, 'V4': None, 'IT': None}
        
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, layer.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        times = inp.size(0)
        
        outputs = {'inp': inp, 'V1': [], 'V2': [], 'V4': [], 'IT': []}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                this_inp = outputs['inp'][0]
            else:  # at t=0 there is no input yet to V2 and up
                prev_block = blocks[blocks.index(block) - 1]
                this_inp = self.final_outputs[prev_block]
            new_output, new_state = getattr(self, block)(this_inp, self.states[block], batch_size=outputs['inp'].size(1))
            outputs[block].append(new_output)
            self.states[block] = new_state

        for t in range(1, times):
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                if prev_block == 'inp':
                    prev_output = outputs[prev_block][t]
                else:
                    prev_output = outputs[prev_block][t - 1]
                prev_state = self.states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                outputs[block].append(new_output)
                self.states[block] = new_state
        
        for block in blocks[1:]:
            self.final_outputs[block] = outputs[block][-1]
        
        out = torch.stack(outputs['IT'], dim=0)
        out = out.view(out.size(0) * out.size(1), out.size(2), out.size(3), out.size(4))
        out = self.decoder(out)
        out = out.view(times, -1, self.num_classes)
        return out
    
    def reset_state(self):
        self.states = {'V1': 0, 'V2': 0, 'V4': 0, 'IT': 0}
        self.final_outputs = {'V1': None, 'V2': None, 'V4': None, 'IT': None}


def s_cornet(**kwargs):
    return SR_CORnet_RT(**kwargs)
