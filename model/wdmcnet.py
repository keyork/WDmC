"""
@ File Name     :   wdmcnet.py
@ Time          :   2022/12/13
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   deep learning model
@ Class List    :   WDmCNetVGG -- our model based on vgg16
                    WDmCNetResNet -- our model based on resnet50
                    WDmCNetTransformer -- our model based on vit (transformer)
                    WDmCNetNeck -- our neck model
"""

import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange
from .resnetbox import Block
from .transformerbox import pair


class WDmCNetVGG(nn.Module):
    """model based on vgg16"""

    def __init__(self, num_classes: int = 8):
        super(WDmCNetVGG, self).__init__()
        print("Create Model based on VGG16")
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 512x6x6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


Layers = [3, 4, 6, 3]


class WDmCNetResNet(nn.Module):
    """model based on resnet50"""

    def __init__(self, num_classes: int = 8):
        super(WDmCNetResNet, self).__init__()
        print("Create Model based on ResNet50")
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2
            self._make_layer(64, (64, 64, 256), Layers[0]),
            self._make_layer(256, (128, 128, 512), Layers[1], 2),
            self._make_layer(512, (256, 256, 1024), Layers[2], 2),
            self._make_layer(1024, (512, 512, 2048), Layers[3], 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layer(self, in_channels, filter_group, blocks, stride=1):
        layers = []
        base_block = Block(in_channels, filter_group, stride=stride, is_11conv=True)
        layers.append(base_block)
        for i in range(1, blocks):
            layers.append(
                Block(filter_group[2], filter_group, stride=1, is_11conv=False)
            )
        return nn.Sequential(*layers)


class WDmCNetTransformer(nn.Module):
    """model based on transformer (vit)"""

    def __init__(
        self,
        *,
        image_size=224,
        patch_size=32,
        num_classes=8,
        dim=128,
        transformer,
        pool="cls",
        channels=1
    ):
        super().__init__()
        print("Create Model based on Transformer")
        image_size_h, image_size_w = pair(image_size)
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        patch_dim = channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class WDmCNetNeck(nn.Module):
    """neck model

    we have already got three model based on vgg16, resnet50 and vit
    the hypothesis is:
        different model learn different knowledge from the dataset

    we want to combine these knowledge, and we make this neck model
    """

    def __init__(self, base_models, num_classes: int = 8):
        super(WDmCNetNeck, self).__init__()
        print("Create Neck-Model")
        self.base_models = base_models
        self.base2neckxvgg16 = nn.Sequential(
            nn.Linear(64, 1024), nn.LeakyReLU(inplace=True)
        )
        self.base2neckxresnet50 = nn.Sequential(
            nn.Linear(64, 1024), nn.LeakyReLU(inplace=True)
        )
        self.base2neckxvit = nn.Sequential(
            nn.Linear(64, 1024), nn.LeakyReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            x_vgg16 = self.base_models["xvgg16"](x["raw"])
            x_resnet50 = self.base_models["xresnet50"](x["resize"])
            x_vit = self.base_models["xvit"](x["resize"])
        neck_vgg16 = self.base2neckxvgg16(x_vgg16)
        neck_resnet50 = self.base2neckxresnet50(x_resnet50)
        neck_vit = self.base2neckxvit(x_vit)
        neck = neck_vgg16 + neck_resnet50 + neck_vit
        result = self.classifier(neck)
        return result
