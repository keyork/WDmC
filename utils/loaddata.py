import torch
from torch.utils.data import random_split, DataLoader
from data.dataset import WDMCDataset


def load_train_data(train_scale, data_path, transform, batch_size):
    training_dataset = WDMCDataset(data_path, transform, None, "train")
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


def load_test_data(data_path, transform):

    test_dataset = WDMCDataset(data_path, transform, None, "test_self")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return test_dataloader
