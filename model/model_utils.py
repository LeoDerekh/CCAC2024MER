import torch

from model import convnext_tiny, convnext, convnext_timm, resnet, swin_transformer
from model.test import dualmodel
from model.test.testmodel import TestModel

def init_net(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net


def get_model(opt):
    if opt.model == 'convnext_tiny':
        net = convnext_timm.convnext_tiny(opt)
    elif opt.model == 'convnext_small':
        net = convnext_timm.convnext_small(opt)
    elif opt.model == 'testmodel':
        net = TestModel(opt)
    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    return init_net(net, opt.gpu_ids)
