import argparse
import os
from datetime import datetime
import time
import torch
import random
import numpy as np


class Options(object):
    def __init__(self):
        super(Options, self).__init__()

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         description='CCAC2024MER')

        # General settings
        parser.add_argument('--pretrained', type=str, default='miex', help='Pretrained on [miex|imagenet|affectnet]')
        parser.add_argument('--input_type', type=str, default='flow', help='[flow|apex|apex_flow|]')
        parser.add_argument('--gpu_ids', type=str, default='7', help='GPU IDs to use, e.g., "0,1,2" or "-1" for CPU.')
        parser.add_argument('--lucky_seed', type=int, default=42, help='Seed for random initialization, 0 to use current time.')

        # Model settings
        parser.add_argument('--model', type=str, default='convnext_tiny', help='Model to use. [convnext_tiny|resnet18]')
        parser.add_argument('--num_classes', type=int, default=7, help='Number of classes.')

        # Training settings
        parser.add_argument('--batch_size', type=int, default=32, help='Input batch size.')
        parser.add_argument('--n_workers', type=int, default=8, help='Number of workers to load data.')
        parser.add_argument('--epochs', type=int, default=50, help='Number of total epochs to run.')

        # Optimization settings
        parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
        parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use. [adam|adamw]')
        parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='Weight decay term of optimizer.')
        parser.add_argument('--lr_policy', type=str, default='exponential', help='Learning rate policy. [lambda|step|plateau|cosine|exponential]')
        parser.add_argument('--use_ema', type=bool, default=False, help='use EMA.')
        parser.add_argument('--ema_decay', type=float, default=0.9, help='EMA decay.')

        # Data settings
        parser.add_argument('--data_root', default="/data/xiaohang/ME_DATA/", help='Path to the data set.')
        parser.add_argument('--data_path', default="dataset/dfme_apex_optical_flow.csv", help='Path to the data file.')
        parser.add_argument('--data_n_frames_path', default="dataset/dfme_4_frames_optical_flow.csv", help='Path to the four frames data file.')
        parser.add_argument('-a', '--testA', action='store_true')
        parser.add_argument('--testA_data_path', type=str, default="dataset/dfme_testA_apex_optical_flow.csv")
        parser.add_argument('--testB_data_path', type=str, default="dataset/dfme_testB_apex_optical_flow.csv")
        parser.add_argument('--scale_factor', type=float, default=1.0)

        # Checkpoints and results
        parser.add_argument('--ckpt_dir', type=str, default="/NAS/xiaohang/CCAC2024MER/checkpoints", help='Directory to save checkpoints.')
        parser.add_argument('--results', type=str, default='results', help='Directory to save results.')
        parser.add_argument('--opt_file', type=str, default="opt.txt", help='Options file name.')
        parser.add_argument('--log_dir', type=str, default="logs", help='Directory to save tensorboard logs.')

        return parser

    def parse(self):
        parser = self.initialize()
        parser.set_defaults(name=datetime.now().strftime("%y%m%d_%H%M%S"))
        opt = parser.parse_args()

        if opt.model == "dualmodel":
            opt.input_type = "apex_flow"

        # update checkpoint and results dir
        opt.ckpt_dir = os.path.join(opt.ckpt_dir, opt.model, opt.name)
        if not os.path.exists(opt.ckpt_dir):
            os.makedirs(opt.ckpt_dir)

        if not os.path.exists(opt.results):
            os.makedirs(opt.results)

        # set gpu device
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                opt.gpu_ids.append(cur_id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        opt.device = torch.device('cuda:%d' % opt.gpu_ids[0] if opt.gpu_ids else 'cpu')
        print('device:', opt.device)

        # set seed
        if opt.lucky_seed == 0:
            opt.lucky_seed = int(time.time())
        random.seed(a=opt.lucky_seed)
        np.random.seed(seed=opt.lucky_seed)
        torch.manual_seed(opt.lucky_seed)
        if len(opt.gpu_ids) > 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(opt.lucky_seed)
            torch.cuda.manual_seed_all(opt.lucky_seed)

        # print and write options file
        msg = ''
        msg += '------------------- [%s]Options --------------------\n' % opt.name
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_v = parser.get_default(k)
            if v != default_v:
                comment = '\t[default: %s]' % str(default_v)
            msg += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        msg += '--------------------- [%s]End ----------------------\n' % (
            opt.name)
        print(msg)
        with open(os.path.join(opt.ckpt_dir, "opt.txt"), 'a+') as f:
            f.write(msg + '\n\n')

        return opt


if __name__ == '__main__':
    opt = Options().parse()
    print(type(vars(opt)))
    print(vars(opt))
