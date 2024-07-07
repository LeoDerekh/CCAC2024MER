import os
import torch
import timm
import sys
sys.path.append(os.getcwd())
from model import convnext
from model.resnet import load_pretrained_weights
import torchvision.models as models

def convnext_tiny(opt):
    if opt.pretrained == "miex":
        if opt.input_type == "flow":
            # file = "/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240530_193701.pth"
            # file = "/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240610_190618.pth"
            file = "/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240611_001220.pth"
        elif opt.input_type == "apex":
            file = "/NAS/xiaohang/checkpoints/8_convnext_tiny_miex_apex_240530_182951.pth"
    elif opt.pretrained == "imagenet":
        file = "/NAS/xiaohang/checkpoints/convnext_tiny.in12k_ft_in1k.bin"
    elif opt.pretrained == "affectnet":
        file = "/NAS/xiaohang/checkpoints/8_convnext_tiny_affectnet_apex_240601_121453.pth"


    convnext_tiny = timm.create_model(
        model_name="convnext_tiny.in12k_ft_in1k",
        num_classes=opt.num_classes,
        pretrained=True,
        pretrained_cfg_overlay=dict(file=file),
    )

    return convnext_tiny


def convnext_tiny_backbone(opt):
    if opt.pretrained == "miex":
        if opt.input_type == "flow":
            # file = "/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240530_193701.pth"
            # file = "/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240610_190618.pth"
            file = "/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240611_001220.pth"
        elif opt.input_type == "apex":
            file = "/NAS/xiaohang/checkpoints/8_convnext_tiny_miex_apex_240530_182951.pth"

    elif opt.pretrained == "imagenet":
        file = "/NAS/xiaohang/checkpoints/convnext_tiny.in12k_ft_in1k.bin"

    convnext_tiny_backbone = timm.create_model(
        model_name="convnext_tiny.in12k_ft_in1k",
        features_only=True,
        pretrained=True,
        output_stride=32,
        out_indices=(0, 1, 2, 3),
        pretrained_cfg_overlay=dict(file=file),
    )


    return convnext_tiny_backbone


# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    model = timm.create_model(
        model_name="convnext_tiny.in12k_ft_in1k",
        num_classes=7,
        pretrained=True,
        pretrained_cfg_overlay=dict(
            file="/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240530_193701.pth"
        ),
    )
    model = model.cuda()
    x = torch.rand(64, 3, 224, 224).cuda()
    output = model(x)
    from torchsummary import summary
    summary(model, (3, 224, 224))

    print("***************named_parameters***************")
    for name, para in model.named_parameters():
        print(name, para.shape)

    print(output.shape)


    print("***************timm***************")
    feature_extractor = timm.create_model(
        model_name="convnext_tiny",
        features_only=True,
        pretrained=False,
        out_indices=(0, 1, 3),
    ).cuda()
    # convnext_tiny (B, 96, 56, 56) (B, 192, 28, 28) (B, 384, 14, 14) (B, 768, 7, 7) 
    output = feature_extractor(x)
    for item in feature_extractor:
        print(item)
    print(len(output))

