from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    """
    modified version of VGG16 model to access activations from 

    middle layers for neural style transfer.
    """

    def __init__(self, requires_grad=True):
        super().__init__()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # NOTE: change False to TRUE before actual training on colab
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
