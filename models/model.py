from torch import nn
import torch
from collections import OrderedDict
from fastai.layers import *
import fastai
from fastcore.basics import *
from fastcore.meta import *
import numpy as np
import pandas as pd
from enum import Enum
from torchvision import models

# V1_2
class CNN1D(nn.Module):
    def __init__(self, input_length, num_class=6):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)  #[5, 17, 47]
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=17)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=47)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_conv_output_size(input_length), 128)
        self.fc2 = nn.Linear(128, num_class)

    def _get_conv_output_size(self, input_length):
        size = input_length
        size = (size - 4) // 2  # conv1 and pool1
        size = (size - 16) // 2  # conv2 and pool2
        size = (size - 46) // 2  # conv3 and pool3
        return size * 128

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# V2_5(best)
class V2(nn.Module):
    def __init__(self, input_length, num_class=2):
        super(V2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=47)  #[5, 17, 47]
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=17)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4992, 128)
        self.fc2 = nn.Linear(128, num_class)

    def _get_conv_output_size(self, input_length):
        size = input_length
        size = (size - 4) // 2  # conv1 and pool1
        size = (size - 16) // 2  # conv2 and pool2
        size = (size - 46) // 2  # conv3 and pool3
        return size * 128

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self, *args, **kwargs): super().__init__()
    def __init__(self): pass

class Reshape(Module):
    def __init__(self, *shape): self.shape = shape
    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(-1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"
        
class Concat(Module):
    def __init__(self, dim=1): self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'

class Add(Module):
    def forward(self, x, y): return x.add(y)
    def __repr__(self): return f'{self.__class__.__name__}'

def Norm(nf, ndim=1, norm='Batch', zero_norm=False, init=True, **kwargs):
    "Norm layer with `nf` features and `ndim` with auto init."
    bn = nn.BatchNorm1d(nf)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.)
    return bn

class AdaptiveConcatPool1d(Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

class GAP1d(Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gap(x))

class WaveCblock(nn.Sequential):
    def __init__(self, ni, nf, kernel_size, padding='same', stride=1, act=nn.ReLU):
        layers = []
        if padding == 'same':
            padding = (kernel_size - 1) // 2
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding)
        layers += [conv, Norm(nf), act()]
        super().__init__(*layers)

class WaveBlock(Module):
    def __init__(self, n_in, nf=[47, 47, 47, 47], kernel_sizes=47, stride=1, padding='same', **kwargs):
        ks = [kernel_sizes // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]
        print(ks)
        self.bottleneck = WaveCblock(n_in, nf[0], 1, padding='same')
        self.layer = len(nf) - 1
        self.waves = nn.ModuleList()
        for i in range(self.layer):
            in_channels = nf[i]
            out_channels = nf[i + 1]
            wave = WaveCblock(in_channels, out_channels, ks[i], stride=stride)
            self.waves.append(wave)
    
    def forward(self, x):
        x = self.bottleneck(x)
        for i in range(self.layer):
            x = self.waves[i](x)
        return x

@delegates(WaveBlock.__init__)
class WaveLength(Module):
    def __init__(self, c_in, c_out, nf, **kwargs):
        self.backbone = WaveBlock(c_in, nf=nf)
        self.head = GAP1d(1)
        self.fc = nn.Linear(nf[-1], c_out)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = self.fc(x)
        return x

        
class ConvBlock(nn.Sequential):
    def __init__(self, ni, nf, kernel_size, padding='same', stride=1, dilation=1, batch_std=0.01, norm='batch', act=nn.ReLU, bn_first=True, act_kwargs = {}):
        ndim = 1
        layers = []
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size//2 * dilation, dilation=dilation)
        layers += [conv]
        act_bn = []

        if act is not None:
            act_bn.append(act())
            
        if norm is not None:
            act_bn.append(Norm(nf))
        
        if bn_first: act_bn.reverse()
        layers += act_bn
        return super().__init__(*layers)

class InceptionModulePlus(Module):
    def __init__(self, ni, nf, ks=10, bottleneck=True, padding='same', act=nn.ReLU, act_kwargs={}, **kwargs):
        ks = [ks // (2**i) for i in range(3)]
        ks = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in ks]
        bottleneck = False if ni == nf else bottleneck
        self.bottleneck = ConvBlock(ni, nf, 1, norm=None, act=None) if bottleneck else noop
        self.convs = nn.ModuleList()
        for i in range(len(ks)):
            self.convs.append(ConvBlock(nf if bottleneck else ni, nf, ks[i],act=act))
        self.mp_conv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), ConvBlock(ni, nf, 1)])
        self.concat = Concat()
        self.norm = Norm(nf * 4)
        self.act = act(**act_kwargs) if act else noop
        self._init_cnn(self)
        
            
    def _init_cnn(self, m):
        if getattr(self, 'bias', None) is not None: nn.init.constant_(self.bias, 0)
        if isinstance(self, (nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear)): nn.init.kaiming_normal_(self.weight)
        for l in m.children(): self._init_cnn(l)

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        x = self.concat([l(x) for l in self.convs] + [self.mp_conv(input_tensor)])
        x = self.norm(x)
        x = self.act(x)
        return x

@delegates(InceptionModulePlus.__init__)
class InceptionBlockPlus(Module):
    def __init__(self, ni, nf, depth=3, act=nn.ReLU, act_kwargs={}, **kwargs):
        self.depth = depth
        self.inceptions = nn.ModuleList()
        self.shortcut = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.add = Add()
        for d in range(self.depth):
            if d == 0:
                self.inceptions.append(InceptionModulePlus(ni, nf, **kwargs, act=None))

            elif d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.inceptions.append(InceptionModulePlus(nf * 4, nf, act=None, **kwargs))
                self.shortcut.append(Norm(n_in) if n_in == n_out else ConvBlock(n_in, n_out, 1, act=None))
                self.acts.append(act(**act_kwargs))
                
            else:
                self.inceptions.append(InceptionModulePlus(nf * 4, nf, **kwargs))
                
            
    def forward(self, x):
        res = x
        for i in range(self.depth):
            x = self.inceptions[i](x)
            if i % 3 == 2:
                res = x = self.acts[i//3](self.add(x, self.shortcut[i//3](res)))
        return x
        
@delegates(InceptionModulePlus.__init__)
class InceptionTimePlus(nn.Sequential):
    def __init__(self, c_in, c_out, nf=32, depth=1, **kwargs):
        self.head_nf = nf * 4
        backbone = InceptionBlockPlus(c_in, nf, depth, **kwargs)
        
        head = self.create_head(self.head_nf, c_out)
        
        layers = OrderedDict([('backbone', nn.Sequential(backbone)), ('head', nn.Sequential(head))]) 
        super().__init__(layers)
        
    def create_head(self, nf, c_out, concat_pool=False):
        layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [Norm(nf)]
        layers += [nn.Linear(nf, c_out)]
        return nn.Sequential(*layers)

class myModel(nn.Module):
    def __init__(self, pretrained=True):
        super(myModel, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)
        # replace the last layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5, inplace=True),   # input 2048
            nn.Linear(self.backbone.fc.in_features, 6)
        )

    def forward(self, x):
        out = self.backbone(x)
        return out