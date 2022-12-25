"""
@ File Name     :   loaddata.py
@ Time          :   2022/12/13
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   load data from file -> dataloader or list
@ Function List :   load_train_data() -- load train dataset -> dataloader
                    load_test_data() -- load test dataset -> dataloader
                    load_raw_test_data() -- load test dataset for get final result -> list
"""


import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from data.dataset import WDmCDataset, WDmCNeckDataset


def load_train_data(train_scale, data_path, transform, batch_size, is_neck):
    """load train dataset -> dataloader

    Args:
        train_scale (float): train - valid scale
        data_path (str): dataset file path
        transform (transform): transform to process data before using
        batch_size (int): deep learning trian batch size
        is_neck (bool): is neck model or not

    Returns:
        train_dataloader, valid_dataloader: two dataloaders
    """
    print("load train data from <- {}".format(data_path))
    if is_neck:
        training_dataset = WDmCNeckDataset(data_path, transform, None, "train")
    else:
        training_dataset = WDmCDataset(data_path, transform, None, "train")
    train_num = int(len(training_dataset) * train_scale)
    valid_num = len(training_dataset) - train_num
    training_data, validing_data = random_split(
        training_dataset,
        [train_num, valid_num],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        validing_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader


def load_test_data(data_path, transform, is_neck):
    """load test dataset -> dataloader

    Args:
        data_path (str): dataset file path
        transform (transform): transform to process data before using
        is_neck (bool): is neck model or not

    Returns:
        test_dataloader: test dataloader
    """
    print("load test data from <- {}".format(data_path))
    if is_neck:
        test_dataset = WDmCNeckDataset(data_path, transform, None, "test_self")
    else:
        test_dataset = WDmCDataset(data_path, transform, None, "test_self")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return test_dataloader


def load_raw_test_data(data_path, transform, is_neck):
    """get dataset for getting the final result

    Args:
        data_path (_type_): _description_
        transform (_type_): _description_
        is_neck (bool): _description_


    if is_neck is true:
        we need a list, every element in it is two tensors like:
                [[1x52x52, 1x224x224], [1x52x52, 1x224x224], ...]
            1x52x52 is for our model based on VGG16
            1x224x224 is for our model based on ResNet50 and Vit

    if is_neck is false:
        we only need 1x52x52, others are the same as above
    """
    print("load raw test data from <- {}".format(data_path))
    raw_data = np.load(data_path)
    test_data = raw_data["test"]
    final_dataloader = []

    if is_neck:
        for img in test_data:
            final_dataloader.append(
                {
                    "raw": transform["raw"](img).unsqueeze(0),
                    "resize": transform["resize"](img).unsqueeze(0),
                }
            )
    else:
        for img in test_data:
            final_dataloader.append(transform(img).unsqueeze(0))
    return final_dataloader
