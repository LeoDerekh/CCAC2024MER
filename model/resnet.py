# 解读pytorch官方对resnet的实现
import collections
import os
import sys
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)
sys.path.append(current_working_directory)
from option.option import Options
__all__ = ['ResNet', 'resnet18', 'resnet50']



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


# 封装一个3*3的卷积函数
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# 封装一个1*1的卷积函数
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 在这里定义了最重要的残差模块，这里是基础版适用于18和34层的，由两个3*3卷积组成
class BasicBlock(nn.Module):
    """定义BasicBlock残差块类

    参数：
        inplanes (int): 输入的Feature Map的通道数
        planes (int): 第一个卷积层输出的Feature Map的通道数
        stride (int, optional): 第一个卷积层的步长
        downsample (nn.Sequential, optional): 旁路下采样的操作
    注意：
        残差块输出的Feature Map的通道数是planes*expansion
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
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


# 定义了50 101 152 等深层resnet的残差模块，由1*1，3*3 ，1*1的卷积堆叠而成
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
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


class ResNet(nn.Module):
    """定义ResNet网络的结构

   参数：
       block (BasicBlock / Bottleneck): 残差块类型
       layers (list): 每一个stage的残差块的数目，长度为4
       num_classes (int): 类别数目
       zero_init_residual (bool): 若为True则将每个残差块的最后一个BN层初始化为零，
           这样残差分支从零开始每一个残差分支，每一个残差块表现的就像一个恒等映射，根据
           https://arxiv.org/abs/1706.02677这可以将模型的性能提升0.2~0.3%
   """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # 第一个残差块的输入通道数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 64, layers[0])  # stride默认为1
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """定义ResNet的一个Stage的结构

        参数：
            block (BasicBlock / Bottleneck): 残差块结构
            plane (int): 残差块中第一个卷积层的输出通道数
            blocks (int): 当前Stage中的残差块的数目
            stride (int): 残差块中第一个卷积层的步长
        """

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:  # 当残差块的输入和输出的尺寸不一致或者通道数不一致的时候就会需要下采样结构
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion  # 上一层的输出通道planes * block.expansion作为下一层的输入通道inplanes
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(opt, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=opt.num_classes, **kwargs)
    pretrained = opt.pretrained
    print(f'[!] initializing model with "{pretrained}" weights ...')
    if pretrained == 'imagenet':
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
    elif pretrained == 'miex':
        if opt.input_type == 'flow':
            state_dict = torch.load("", map_location=opt.device)
        elif opt.input_type == 'apex':
            state_dict = torch.load("", map_location=opt.device)
    else:
        raise NotImplementedError('wrong pretrained model!')
    load_pretrained_weights(model, state_dict)
    return model


def resnet50(opt, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=opt.num_classes, **kwargs)
    pretrained = opt.pretrained
    print(f'[!] initializing model with "{pretrained}" weights ...')
    if pretrained == 'imagenet':
        state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
    elif pretrained == 'miex':
        if opt.input_type == 'flow':
            state_dict = torch.load("", map_location=opt.device)
        elif opt.input_type == 'apex':
            state_dict = torch.load("", map_location=opt.device)
    else:
        raise NotImplementedError('wrong pretrained model!')
    load_pretrained_weights(model, state_dict)
    return model


def load_pretrained_weights(model, pretrained_state_dict):
    print('Loading pretrained weights...')

    def remove_module_prefix(state_dict):
        """Remove 'module.' prefix from state_dict keys."""
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it exists
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    # Remove 'module.' prefix from pretrained state dict if present
    pretrained_state_dict = remove_module_prefix(pretrained_state_dict)

    # Retrieve the model's state dictionary
    model_state_dict = model.state_dict()

    # Create a new ordered dictionary to hold the matched parameters
    new_state_dict = collections.OrderedDict()

    # Lists to keep track of matched and discarded layers
    matched_layers, discarded_layers = [], []

    # Iterate over the pretrained state dictionary
    for k, v in pretrained_state_dict.items():
        if k in model_state_dict and model_state_dict[k].size() == v.size():
            # If the key and size match, add to new state dictionary
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            # Otherwise, add to discarded layers list
            discarded_layers.append(k)

    # Print the number of parameters in the pretrained model
    print('Total params num (pretrained):', len(pretrained_state_dict))
    # Print the number of parameters in the model's state dictionary
    print('Model state dict params:', len(model_state_dict))

    # Update the model's state dictionary with the new matched parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    # Print the updated number of parameters in the model's state dictionary
    print('Updated model state dict params:', len(model_state_dict))
    # Print the number of loaded and discarded parameters
    print('Loaded params num (from pretrained):', len(matched_layers))
    print('Discarded pretrained params num:', len(discarded_layers))
    # Print the keys of discarded parameters
    print('Discarded pretrained keys:', discarded_layers)


if __name__ == '__main__':
    opt = Options().parse()
    opt.pretrained = 'imagenet'
    model = resnet50(opt).cuda()
    from torchsummary import summary
    summary(model, (3, 224, 224))
    # resnet18 (64, 112, 112) (64, 56, 56) (128, 28, 28) (256, 14, 14) (512, 7, 7)
    # resnet50 (64, 112, 112) (256, 56, 56) (512, 28, 28) (1024, 14, 14) (2048, 7, 7)
    x = torch.rand(64, 3, 224, 224).cuda()
    output = model(x)

    for name, para in model.named_parameters():
        print(name, para.shape)

    print('output', output.shape)



    import timm
    model = timm.create_model(
            model_name="resnet18",
            num_classes=opt.num_classes,
            pretrained=False
        ).cuda()
    summary(model, (3, 224, 224))

    model = timm.create_model(
            model_name="resnet18",
            features_only=True,
            pretrained=False,
            output_stride=32,
            out_indices=(0, 1, 2, 3, 4),
        ).cuda()
    output = model(x)
    for o in output:
        print(o.shape)
    # summary(model, (3, 224, 224))