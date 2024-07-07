import torch.nn.functional as F
import torch
import torch.nn as nn


from model.convnext_tiny import convnext_tiny_backbone
from model.test.LANMSFF import PWFS
from visualization.visualize_featuremap import vis_featuremap
from model.test import backbone_convnext_timm
from functools import partial
from timm.layers import LayerNorm2d, trunc_normal_
from timm.layers import NormMlpClassifierHead
from timm.models import named_apply


class TestModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = backbone_convnext_timm.convnext_tiny(opt)
        # (B, 96, 56, 56) (B, 192, 28, 28) (B, 384, 14, 14) (B, 768, 7, 7) 

        # self.pwfs1 = PWFS()
        # self.pwfs2 = PWFS()
        # self.pwfs3 = PWFS()

        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc = nn.Linear(in_features=992, out_features=opt.num_classes)

        # named_apply(partial(_init_weights, head_init_scale=1.), self)
    
    def forward(self, x):
        x, res = self.backbone(x)
        # convnext_tiny (B, 96, 56, 56) (B, 192, 28, 28) (B, 384, 14, 14) (B, 768, 7, 7) 
        # x1, x2, x3, x4 = res

        # x1 = self.global_avg_pool(self.pwfs1(x1))
        # x2 = self.global_avg_pool(self.pwfs2(x2))
        # x3 = self.global_avg_pool(self.pwfs3(x3))
        # x4 = self.global_avg_pool(x4)

        # x = torch.cat([x1, x2, x3, x4], dim=1)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)