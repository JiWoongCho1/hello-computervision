import torchvision.models as models
import torch
import torch.nn as nn
from mlp_head import MLPHead

class Resnet18(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained = False)

        if kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained = False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], x.shape[1])
        return self.projeciton(x)