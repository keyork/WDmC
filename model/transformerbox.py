"""
@ File Name     :   transformerbox.py
@ Time          :   2022/12/18
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
