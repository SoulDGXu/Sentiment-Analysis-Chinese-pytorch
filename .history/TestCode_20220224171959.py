# !usr/bin/env python
# -*- coding:utf-8 -*-

''' 
@Author: HsuDan
@Date: 2022-02-22 11:45:07
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-24 17:19:59
@Description: file content
@FilePath: /Sentiment-Analysis-Chinese-pytorch/TestCode.py
''' 
import pandas as pd

def data_preview(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: 
        data = []
        for line in f.readlines():
            sall = line.strip().split()
            data.append([int(sall[0]), sall[1:]])
    print('data length: ', len(data))
    print('data preview: ', data[:1])
    df = pd.DataFrame(data, columns=['label', 'text'])
    return df



file = './data/train.txt'
df = data_preview(file)
print(df.head())
df.info()
print(df['label'].unique())
                