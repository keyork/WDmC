"""
@ File Name     :   cfg.py
@ Time          :   2022/12/13
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   transforms config
"""

from torchvision import transforms

train_transform_vgg = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(45),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

test_transform_vgg = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

train_transform_vit = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(45),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

test_transform_vit = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)


train_transform_resnet = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(45),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

test_transform_resnet = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)
