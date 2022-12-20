import numpy as np
from torch.utils.data import Dataset


class WDmCDataset(Dataset):
    def __init__(
        self,
        data_path,
        transform=None,
        target_transform=None,
        data_type="train",
    ):
        self.raw_data = np.load(data_path)
        self.data_type = data_type
        if self.data_type == "test_self":
            self.data = self.raw_data["test"]
            self.label = self.raw_data["label_test"]
        elif self.data_type == "train":
            self.data = self.raw_data["train"]
            self.label = self.raw_data["label_train"]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.data_type == "test_self":
            label = self.label[idx]
        elif self.data_type == "train":
            label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class WDmCNeckDataset(Dataset):
    def __init__(
        self,
        data_path,
        transform=None,
        target_transform=None,
        data_type="train",
    ):
        self.raw_data = np.load(data_path)
        self.data_type = data_type
        if self.data_type == "test_self":
            self.data = self.raw_data["test"]
            self.label = self.raw_data["label_test"]
        elif self.data_type == "train":
            self.data = self.raw_data["train"]
            self.label = self.raw_data["label_train"]
        self.transform_raw = transform["raw"]
        self.transform_resize = transform["resize"]
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.data_type == "test_self":
            label = self.label[idx]
        elif self.data_type == "train":
            label = self.label[idx]
        if self.transform_raw:
            image_raw = self.transform_raw(image)
            image_resize = self.transform_resize(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image_raw, image_resize, label
