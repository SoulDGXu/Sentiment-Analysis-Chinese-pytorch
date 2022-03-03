# !usr/bin/env python
# -*- coding:utf-8 -*-

''' 
@Author: HsuDan
@Date: 2022-02-22 11:45:07
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-24 17:14:10
@Description: file content
@FilePath: /Sentiment-Analysis-Chinese-pytorch/TestCode.py
''' 
import pandas as pd

def data_preview(file):
    with open(file, 'r', encoding='utf-8') as f: # './word2vec/train.txt'
        data = []
        for line in f.readlines():
            sall = line.strip().split()
            data.append([int(sall[0]), sall[1:]])
    print('data length: ', len(data))
    print('data preview: ', data[:5])



print(len(data), data[0])
                