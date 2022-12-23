'''
@ File Name     :   initweights.py
@ Time          :   2022/12/13
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
'''

import torch.nn as nn
import torch


def init_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.constant_(layer.bias, 0)
    return model


def load_weights(model, path):
    model.load_state_dict(torch.load(path))
    return model
