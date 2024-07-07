import os

import numpy as np
import pandas as pd
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, img_paths, img_labels, transform=None, get_aux=False, aux=None):
        self.root = root
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform
        self.get_aux = get_aux
        self.aux = aux

    def __getitem__(self, idx):
        img_paths = self.img_paths[idx]
        label = self.img_labels[idx]

        if isinstance(img_paths, str):
            img_paths = [img_paths]

        imgs = []
        for img_path in img_paths:
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        if len(imgs) == 1:
            imgs = imgs[0]

        if self.get_aux:
            return imgs, label, self.aux[idx]
        else:
            return imgs, label

    def __len__(self):
        return len(self.img_paths)

    def __distribute__(self):
        distribute_ = np.array(self.img_labels)
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_paths, transform=None, get_aux=False, aux=None):
        self.root = root
        self.img_paths = img_paths
        self.transform = transform
        self.get_aux = get_aux
        self.aux = aux

    def __getitem__(self, idx):
        img_paths = self.img_paths[idx]

        if isinstance(img_paths, str):
            img_paths = [img_paths]

        imgs = []
        for img_path in img_paths:
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        if len(imgs) == 1:
            imgs = imgs[0]

        if self.get_aux:
            return imgs, self.aux[idx]
        else:
            return imgs

    def __len__(self):
        return len(self.img_paths)


def data_split(data_path):
    # 读取 csv 文件
    df = pd.read_csv(data_path)

    # 读取 train_path.txt 和 val_path.txt 文件
    with open('dataset/train_path.txt', 'r') as train_file:
        train_filenames = train_file.read().splitlines()

    with open('dataset/val_path.txt', 'r') as val_file:
        val_filenames = val_file.read().splitlines()

    # 根据文件名划分数据
    df_train = df[df['filename'].isin(train_filenames)]
    df_val = df[df['filename'].isin(val_filenames)]

    return df_train, df_val


def get_apex_data(df):
    paths = list(df.apex_frame_path)
    labels = list(df.label)
    return paths, labels


def get_optical_flow_data(df):
    paths = list(df.optical_flow_path)
    labels = list(df.label)
    return paths, labels


def get_magnify_data(df):
    paths = list(df.magnify_path)
    labels = list(df.label)
    return paths, labels


def get_apex_optical_flow_data(df):
    apex_paths = list(df.apex_frame_path)
    optical_flow_paths = list(df.optical_flow_path)
    paths = [(apex, optical_flow) for (apex, optical_flow) in zip(apex_paths, optical_flow_paths)]
    labels = list(df.label)
    return paths, labels

def get_apex_test_data(df):
    paths = list(df.apex_frame_path)
    return paths

def get_optical_flow_test_data(df):
    paths = list(df.optical_flow_path)
    return paths

def get_apex_optical_flow_test_data(df):
    apex_paths = list(df.apex_frame_path)
    optical_flow_paths = list(df.optical_flow_path)
    paths = [(apex, optical_flow) for (apex, optical_flow) in zip(apex_paths, optical_flow_paths)]
    return paths


def get_triple_meta_data(df):
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    optical_flow_paths = list(df.optical_flow_path)
    paths = [(on, apex, optical_flow) for (on, apex, optical_flow) in zip(on_paths, apex_paths, optical_flow_paths)]
    labels = list(df.label)
    return paths, labels


def get_four_meta_data(df):
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    off_paths = list(df.offset_frame_path)
    optical_flow_paths = list(df.optical_flow_path)
    paths = [(on, apex, off, optical_flow) for (on, apex, off, optical_flow) in
             zip(on_paths, apex_paths, off_paths, optical_flow_paths)]
    labels = list(df.label)
    return paths, labels


def upsample_subdata(df, df_four, number=4):
    result = df.copy()
    for i in range(df.shape[0]):
        quotient = int(number)  # 确保是整数
        remainder = number % 1
        remainder = 1 if np.random.rand() < remainder else 0
        value = quotient + remainder

        tmp = df_four[(df_four['subject'] == df.iloc[i]['subject']) & (df_four['filename'] == df.iloc[i]['filename'])]
        value = min(value, tmp.shape[0])
        tmp = tmp.sample(int(value))
        result = pd.concat([result, tmp])
    return result


def sample_data(df, df_four, scale_factor=1):
    # 分离7个类别的数据
    df_list = [df[df.label == i] for i in range(7)]

    # 选择一个基准类别（通常选择数量最多的类别）
    base_class_size = max([sub_df.shape[0] for sub_df in df_list]) * scale_factor

    # 计算每个类别的上采样比例
    upsample_ratios = [(base_class_size / sub_df.shape[0]) - 1 if sub_df.shape[0] < base_class_size else 0 for sub_df in
                       df_list]

    # 上采样每个类别的数据
    for i in range(7):
        df_list[i] = upsample_subdata(df_list[i], df_four, upsample_ratios[i])
        print(f'df_class_{i}', df_list[i].shape)

    # 合并所有上采样后的类别数据
    df_balanced = pd.concat(df_list)
    return df_balanced
