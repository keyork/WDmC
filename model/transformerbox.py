"""
@ File Name     :   transformerbox.py
@ Time          :   2022/12/18
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   transformer (vit) tools
@ Function List :   pair()
"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
