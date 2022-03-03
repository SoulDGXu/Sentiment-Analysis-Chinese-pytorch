# !usr/bin/env python
# -*- coding:utf-8 -*-

''' 
@Author: HsuDan
@Date: 2022-02-22 11:45:07
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-22 12:22:19
@Description: file content
@FilePath: /Sentiment/Sentiment-Analysis-Chinese-pytorch/TestCode.py
''' 

from __init__ import *
import os

curPath = os.path.abspath(os.path.dirname(__file__))
with open(curPath + '/word2vec/train.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
print(len(data), data[0])
                