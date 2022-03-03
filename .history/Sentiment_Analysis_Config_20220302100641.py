# !usr/bin/env python
# -*- coding:utf-8 -*-

""" 
@Author: HsuDan
@Date: 2022-02-18 19:12:58
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-03-01 19:05:17
@Description: file content
@FilePath: /Sentiment-Analysis-Chinese-pytorch/Sentiment_Analysis_Config.py
"""


class Config:
    update_w2v = True  # 是否在训练中更新w2v
    vocab_size = 54848  # 词汇量，与word2id中的词汇量一致
    n_class = 2  # 分类数：分别为pos和neg
    max_sen_len = 75  # 句子最大长度
    embedding_dim = 50  # 词向量维度
    batch_size = 64  # 批处理尺寸
    hidden_dim = 128  # 隐藏层节点数
    n_epoch = 5  # 训练迭代周期，即遍历整个训练样本的次数
    lr = 0.0001  # 学习率；若opt=‘adadelta'，则不需要定义学习率
    drop_keep_prob = 0.2  # dropout层，参数keep的比例
    num_layers = 2  # LSTM层数
    bidirectional = True  # 是否使用双向LSTM
    stopword_path = "./data/stopword.txt"
    train_path = "./data/train.txt"
    val_path = "./data/validation.txt"
    test_path = "./data/test.txt"
    pre_path = "./data/pre.txt"
    word2id_path = "./word2vec/word2id.txt"
    pre_word2vec_path = "./word2vec/wiki_word2vec_50.bin"
    corpus_word2vec_path = "./word2vec/word_vec.txt"
    model_state_dict_path = "./model"  # 训练模型保存的地址
