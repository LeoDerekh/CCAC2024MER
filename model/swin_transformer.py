import os
import torch
import timm



def swin_t(opt):
    if opt.pretrained != "n":
        if opt.pretrained == "miex":
            if opt.input_type == "flow":
                file = "/NAS/xiaohang/checkpoints/16_swin_t_miex_flow_240611_181538.pth"
            elif opt.input_type == "apex":
                file = ""
        elif opt.pretrained == "imagenet":
            file = "/NAS/xiaohang/checkpoints/swin_tiny_patch4_window7_224.ms_in1k.bin"


        swin_t = timm.create_model(
            model_name="swin_tiny_patch4_window7_224.ms_in1k",
            num_classes=opt.num_classes,
            pretrained=opt.pretrained,
            pretrained_cfg_overlay=dict(file=file),
        )
    else:
        swin_t = timm.create_model(
            model_name="swin_tiny_patch4_window7_224.ms_in1k",
            num_classes=opt.num_classes,
            pretrained=False,)
    return swin_t


def swin_b(opt):
    if opt.pretrained != "n":
        if opt.pretrained == "miex":
            if opt.input_type == "flow":
                file = "/NAS/xiaohang/checkpoints/11_swin_b_miex_flow_240610_221220.pth"
            elif opt.input_type == "apex":
                file = ""
        elif opt.pretrained == "imagenet":
            file = "/NAS/xiaohang/checkpoints/swin_base_patch4_window7_224.ms_in22k_ft_in1k.bin"


        swin_b = timm.create_model(
            model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            num_classes=opt.num_classes,
            pretrained=opt.pretrained,
            pretrained_cfg_overlay=dict(file=file),
        )
    else:
        swin_b = timm.create_model(
            model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            num_classes=opt.num_classes,
            pretrained=False,)
    return swin_b


def swin_b_backbone(opt):
    if opt.pretrained == "miex":
            if opt.input_type == "flow":
                file = "/NAS/xiaohang/checkpoints/11_swin_b_miex_flow_240610_221220.pth"
            elif opt.input_type == "apex":
                file = ""
    elif opt.pretrained == "imagenet":
        file = "/NAS/xiaohang/checkpoints/swin_base_patch4_window7_224.ms_in22k_ft_in1k.bin"

    swin_b_backbone = timm.create_model(
        model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        features_only=True,
        pretrained=True,
        output_stride=32,
        out_indices=(0, 1, 2, 3),
        pretrained_cfg_overlay=dict(file=file),
    )
    return swin_b_backbone


# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    model = timm.create_model(
        model_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        num_classes=7,
        pretrained=True,
        pretrained_cfg_overlay=dict(
            file="/NAS/xiaohang/checkpoints/7_convnext_tiny_miex_flow_240530_193701.pth"
        ),
    )

    x = torch.rand(64, 3, 224, 224)
    output = model(x)
    for name, para in model.named_parameters():
        print(name, para.shape)

    print(output.shape)
