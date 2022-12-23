"""
@ File Name     :   resnetbox.py
@ Time          :   2022/12/18
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""


import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, filter_group, stride=1, is_11conv=False):
        super(Block, self).__init__()

        filter1, filter2, filter3 = filter_group
        self.is_11conv = is_11conv
        self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU(),
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU(),
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter3),
        )

    def forward(self, x):

        shortcut = x
        x = self.features(x)
        if self.is_11conv:
            shortcut = self.shortcut(shortcut)
        x += shortcut
        x = self.relu(x)
        return x
