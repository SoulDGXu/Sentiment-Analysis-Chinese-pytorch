# !usr/bin/env python
# -*- coding:utf-8 -*-

''' 
@Author: HsuDan
@Date: 2022-02-22 11:45:07
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-22 14:16:40
@Description: file content
@FilePath: /Sentiment-Analysis-Chinese-pytorch/TestCode.py
''' 

with open( './word2vec/train.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
print(len(data), data[0])
                