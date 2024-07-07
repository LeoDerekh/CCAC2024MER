import sys
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from torchvision import transforms

from dataset.dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle

from ema import EMA


def load_me_data(opt):
    df_train, df_val = data_split(opt.data_path)
    # train oversampling
    df_n_frames = pd.read_csv(opt.data_n_frames_path)
    df_train = sample_data(df_train, df_n_frames, scale_factor=opt.scale_factor)
    df_train = shuffle(df_train)

    if opt.input_type == 'flow':
        train_paths, train_labels = get_optical_flow_data(df_train)
        val_paths, val_labels = get_optical_flow_data(df_val)
    elif opt.input_type == 'magnify':
        train_paths, train_labels = get_magnify_data(df_train)
        val_paths, val_labels = get_magnify_data(df_val)
    elif opt.input_type == 'apex':
        train_paths, train_labels = get_apex_data(df_train)
        val_paths, val_labels = get_apex_data(df_val)
    elif opt.input_type == 'apex_flow':
        train_paths, train_labels = get_apex_optical_flow_data(df_train)
        val_paths, val_labels = get_apex_optical_flow_data(df_val)

    train_transforms = transforms.Compose([transforms.Resize((240, 240), interpolation=InterpolationMode.BICUBIC),
                                           transforms.RandomRotation(degrees=(-8, 8)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                  saturation=0.2, hue=0.2),
                                           transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                           ])

    val_transforms = transforms.Compose([transforms.Resize((240, 240), interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                         ])

    train_dataset = Dataset(root=opt.data_root,
                            img_paths=train_paths,
                            img_labels=train_labels,
                            transform=train_transforms)
    print('Train set size:', train_dataset.__len__())
    print('The Train dataset distribute:', train_dataset.__distribute__())

    val_dataset = Dataset(root=opt.data_root,
                          img_paths=val_paths,
                          img_labels=val_labels,
                          transform=val_transforms)
    print('Validation set size:', val_dataset.__len__())
    print('The Validation dataset distribute:', val_dataset.__distribute__())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batch_size,
                                               num_workers=opt.n_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=opt.batch_size,
                                             num_workers=opt.n_workers,
                                             shuffle=False,
                                             pin_memory=True)
    return train_loader, val_loader


def get_test_loader(opt):
    if opt.testA:
        df_test = pd.read_csv(opt.testA_data_path)
    else:
        df_test = pd.read_csv(opt.testB_data_path)
    if opt.input_type == 'flow':
        test_paths = get_optical_flow_test_data(df_test)
    elif opt.input_type == 'apex':
        test_paths = get_apex_test_data(df_test)
    elif opt.input_type == 'apex_flow':
        test_paths = get_apex_optical_flow_test_data(df_test)

    test_transforms = transforms.Compose([transforms.Resize((240, 240), interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                         ])
    test_dataset = TestDataset(root=opt.data_root,
                               img_paths=test_paths,
                               transform=test_transforms)
    print('Test set size:', test_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=opt.batch_size,
                                              num_workers=opt.n_workers,
                                              shuffle=False,
                                              pin_memory=True)
    return test_loader


def train_one_epoch(opt, model, criterion, optimizer, data_loader, device, epoch, ema):
    model.train()
    y_true = []
    y_pred = []
    losses = AverageMeter()
    bar = tqdm(data_loader, file=sys.stdout)
    for batch_idx, (inputs, labels) in enumerate(bar):
        if opt.input_type == "apex_flow":
            apex = inputs[0].to(device)
            optical_flow = inputs[1].to(device)
            outputs = model(apex, optical_flow)
        else:
            inputs = inputs.to(device)
            outputs = model(inputs)

        labels = labels.to(device)

        optimizer.zero_grad()

        
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        if opt.use_ema:
            ema.update()

        losses.update(loss.data.item(), outputs.size(0))

        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

        bar.desc = 'Train Epoch:{:0>3}/{:0>3} loss:{:.4f} lr:{:.8f}'.format(epoch, opt.epochs, loss,
                                                                            optimizer.param_groups[0]['lr'])
    UF1, UAR, ACC, class_accuracies = calculate_metrics(y_true, y_pred, opt.num_classes)
    return losses.avg, UF1, UAR, ACC, list(class_accuracies)


def evaluate(opt, model, criterion, data_loader, device, epoch):
    model.eval()
    y_true = []
    y_pred = []
    losses = AverageMeter()
    bar = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(bar):
            if opt.input_type == "apex_flow":
                apex = inputs[0].to(device)
                optical_flow = inputs[1].to(device)
                outputs = model(apex, optical_flow)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)

            labels = labels.to(device)

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            losses.update(loss.data.item(), outputs.size(0))

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

            bar.desc = 'Validation Epoch:{:0>3}/{:0>3} loss:{:.4f}'.format(epoch, opt.epochs, loss)

    UF1, UAR, ACC, class_accuracies = calculate_metrics(y_true, y_pred, opt.num_classes)
    return losses.avg, UF1, UAR, ACC, list(class_accuracies)


def predict(opt, model, data_loader, device, epoch):
    model.eval()
    y_pred = []
    bar = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for batch_idx, inputs in enumerate(bar):
            inputs = inputs.to(device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            y_pred.extend(preds.tolist())

            bar.desc = 'Test Epoch:{:0>3}/{:0>3}'.format(epoch, opt.epochs)

    return y_pred


def save_info_append(path, info):
    columns = [k for k in info.keys()]
    if not os.path.exists(path):
        df = pd.DataFrame(data=None, columns=columns)
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    new_row = pd.DataFrame(info, index=[0])
    df = pd.concat([df, new_row], ignore_index=True, axis=0)
    df.to_csv(path, index=False)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(opt, parameters):
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    elif opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('optimizer [%s] is not implemented', opt.optimizer)
    return optimizer


def get_scheduler(opt, optimizer):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.987)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def calculate_metrics(y_true, y_pred, num_classes):
    """
    计算UF1, UAR, ACC指标以及每个类别的准确率.

    参数:
    y_true (list or np.array): 真实标签.
    y_pred (list or np.array): 预测标签.
    num_classes (int): 类别总数.

    返回:
    tuple: 包含UF1, UAR, ACC和每个类别准确率的元组.
    """
    # 初始化计数器
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)
    N = np.zeros(num_classes)

    # 计算TP, FP, FN, N
    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        N[true_class] += 1
        if true_class == pred_class:
            TP[true_class] += 1
        else:
            FP[pred_class] += 1
            FN[true_class] += 1

    # 计算UF1
    F1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        if (2 * TP[i] + FP[i] + FN[i]) > 0:
            F1_scores[i] = (2 * TP[i]) / (2 * TP[i] + FP[i] + FN[i])
    UF1 = np.mean(F1_scores)

    # 计算UAR
    recall_scores = np.zeros(num_classes)
    for i in range(num_classes):
        if N[i] > 0:
            recall_scores[i] = TP[i] / N[i]
    UAR = np.mean(recall_scores)

    # 计算ACC
    ACC = np.sum(TP) / np.sum(N)

    # 计算每个类别的准确率
    class_accuracies = np.zeros(num_classes)
    for i in range(num_classes):
        if N[i] > 0:
            class_accuracies[i] = TP[i] / N[i]

    return UF1, UAR, ACC, class_accuracies


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues,
                          save_path=None):
    """
    绘制并保存混淆矩阵.

    参数:
    y_true (list or np.array): 真实标签.
    y_pred (list or np.array): 预测标签.
    classes (list): 类别名称列表.
    normalize (bool): 是否进行归一化.
    title (str): 图表标题.
    cmap (matplotlib.colors.Colormap): 颜色映射.
    save_path (str): 保存图片的路径.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

# # 示例使用
# y_true = [0, 1, 2, 2, 0, 1, 2, 0, 1, 1]
# y_pred = [0, 2, 2, 2, 0, 0, 2, 0, 1, 1]
# class_names = ['Class 0', 'Class 1', 'Class 2']
#
# # 指定保存路径
# save_path = 'confusion_matrix.png'
# plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True, title='Normalized Confusion Matrix', save_path=save_path)
