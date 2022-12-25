"""
@ File Name     :   initweights.py
@ Time          :   2022/12/13
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   init or load the weights of model
@ Function List :   init_weights() -- init model
                    load_weights() -- load model
"""

import torch.nn as nn
import torch


def init_weights(model):
    """init model weights

    Args:
        model (nn.Module): raw deep learning model

    Returns:
        model (nn.Module): deep learning model after init

    kaiming normal init method -> conv layers weight
    not init the bias and others layers (like fc layers ...)
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.constant_(layer.bias, 0)
    return model


def load_weights(model, path):
    """load model weights

    Args:
        model (nn.Module): raw deep learning model
        path (str): the path to deep learning model weights

    Returns:
        model (nn.Module): deep learning model after load weights
    """
    print("load weights from <- {}".format(path))
    model.load_state_dict(torch.load(path))
    return model
