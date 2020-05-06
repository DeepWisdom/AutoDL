import logging
import sys
from collections import OrderedDict
import copy
import torch
import torchvision.models as models
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls

import skeleton
import torch.nn as nn

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activate(out)

        return out


class ResNet18(models.ResNet):
    Block = BasicBlock

    def __init__(self, in_channels, num_classes=10, **kwargs):
        Block = BasicBlock
        self.in_channels = in_channels
        super(ResNet18, self).__init__(Block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )
        self.last_channels = 512 * Block.expansion
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.last_channels, num_classes, bias=False)
        self._half = False
        self._class_normalize = True

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(model_urls['resnet18'], model_dir=self.model_dir)
        del sd['fc.weight']
        del sd['fc.bias']
        self.load_state_dict(sd, strict=False)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_origin(self, x, targets):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs, targets)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum(dim=0)
            nneg = neg.sum(dim=0)

            positive_ratio = torch.clamp((npos) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            negative_ratio = torch.clamp((nneg) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss
        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self

class ResLayer(nn.Module):
    def __init__(self, in_c, out_c, groups=1):
        super(ResLayer, self).__init__()
        self.act = nn.CELU(0.075, inplace=False)
        conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                         bias=False, groups=groups)
        norm = nn.BatchNorm2d(num_features=out_c)
        pool = nn.MaxPool2d(2)
        self.pre_conv = nn.Sequential(
            OrderedDict([('conv', conv), ('pool', pool), ('norm', norm), ('act', nn.CELU(0.075, inplace=False))]))
        self.res1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False, groups=groups)), ('bn', nn.BatchNorm2d(out_c)),
            ('act', nn.CELU(0.075, inplace=False))]))
        self.res2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False, groups=groups)), ('bn', nn.BatchNorm2d(out_c)),
            ('act', nn.CELU(0.075, inplace=False))]))

    def forward(self, x):
        x = self.pre_conv(x)
        out = self.res1(x)
        out = self.res2(out)
        out = out + x
        return out


class ResNet9(nn.Module):

    def __init__(self, in_channels, num_classes=10, **kwargs):
        super(ResNet9, self).__init__()  # resnet18
        channels = [64, 128, 256, 512]
        group = 1
        self.in_channels = in_channels
        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )
        conv1 = nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          bias=False)
        norm1 = nn.BatchNorm2d(num_features=channels[0])
        act = nn.CELU(0.075, inplace=False)
        pool = nn.MaxPool2d(2)
        self.prep = nn.Sequential(OrderedDict([('conv', conv1), ('bn', norm1), ('act', act)]))
        self.layer1 = ResLayer(channels[0], channels[1], groups=group)
        conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1),
                          bias=False, groups=group)
        norm2 = nn.BatchNorm2d(num_features=channels[2])
        self.layer2 = nn.Sequential(OrderedDict([('conv', conv2), ('pool', pool), ('bn', norm2), ('act', act)]))
        self.layer3 = ResLayer(channels[2], channels[3], groups=group)
        self.pool4 = nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Linear(channels[3], num_classes, bias=False)
        self._half = False
        self._class_normalize = True

    def init(self, model_dir=None, gain=1.):
        self.model_dir = model_dir if model_dir is not None else self.model_dir
        sd = model_zoo.load_url(
            'https://github.com/DeepWisdom/AutoDL/releases/download/opensource/r9-70e4b5c2.pth.tar',
            model_dir=self.model_dir)
        new_sd = copy.deepcopy(sd['state_dict'])
        for key, value in sd['state_dict'].items():
            new_sd[key[7:]] = sd['state_dict'][key]
        self.load_state_dict(new_sd, strict=False)

    def forward_origin(self, x, targets):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):  # pylint: disable=arguments-differ
        inputs = self.stem(inputs)
        logits = self.forward_origin(inputs, targets)
        logits /= tau

        if targets is None:
            return logits
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(self.loss_fn, (
                torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum(dim=0)
            nneg = neg.sum(dim=0)

            positive_ratio = torch.clamp((npos) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])
            negative_ratio = torch.clamp((nneg) / (npos + nneg), min=0.03, max=0.97).view(1, loss.shape[1])

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss
        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()
        return logits, loss

    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue

            if not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self
