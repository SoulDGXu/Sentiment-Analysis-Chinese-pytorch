# !usr/bin/env python
# -*- coding:utf-8 -*-

''' 
@Author: HsuDan
@Date: 2022-02-22 11:45:07
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-24 17:09:38
@Description: file content
@FilePath: /Sentiment-Analysis-Chinese-pytorch/TestCode.py
''' 

with open( './word2vec/train.txt', 'r', encoding='utf-8') as f:
    train = []
   for line in f.readlines():
       sall = line.strip().split()
       train.append([int(sall[0]), ])

print(len(data), data[0])
                