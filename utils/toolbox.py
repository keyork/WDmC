"""
@ File Name     :   toolbox.py
@ Time          :   2022/10/11
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   log_color() -- color
                    str2bool() -- args
@ Details       :   None
"""

import argparse
import logging

import colorlog


def log_color():
    """color log

    Returns:
        color_logger (log): logger
    """
    color_logger = logging.getLogger("ROOT")
    color_logger.setLevel(logging.DEBUG)

    log_handler = logging.StreamHandler()
    log_handler.setLevel(logging.DEBUG)

    fmt_str = "%(log_color)s[%(levelname)s]: %(message)s (%(asctime)s)"

    log_colors = {
        "DEBUG": "bg_blue",
        "INFO": "bg_green",
        "WARNING": "bg_yellow",
        "ERROR": "bg_red",
        "CRITICAL": "bg_purple",
    }

    the_format = colorlog.ColoredFormatter(fmt_str, log_colors=log_colors)
    log_handler.setFormatter(the_format)
    color_logger.addHandler(log_handler)

    return color_logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


LOGGER = log_color()
