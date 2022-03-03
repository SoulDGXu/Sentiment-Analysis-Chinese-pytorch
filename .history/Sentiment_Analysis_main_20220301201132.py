# !usr/bin/env python
# -*- coding:utf-8 -*-

""" 
@Author: HsuDan
@Date: 2022-02-18 19:12:58
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-25 11:18:32
@Description: main 
@FilePath: /Sentiment-Analysis-Chinese-pytorch/Sentiment_Analysis_main.py
"""
from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
from Sentiment_Analysis_DataProcess import (
    prepare_data,
    build_word2id,
    build_id2word,
    build_word2vec,
    Data_set,
)
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import os
from Sentiment_model import LSTMModel, LSTM_attention
from Sentiment_Analysis_Config import Config
from Sentiment_Analysis_eval import val_accuary


def train(train_dataloader, model, device, epoches, lr):

    model.train()
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)  # 学习率调整
    best_acc = 0.85
    for epoch in range(epoches):  # 一个epoch可以认为是一次训练循环
        train_loss = 0.0
        correct = 0
        total = 0

        train_dataloader = tqdm.tqdm(train_dataloader)
        # train_dataloader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epoches, 'lr:', scheduler.get_last_lr()[0]))
        for i, data_ in enumerate(train_dataloader):

            optimizer.zero_grad()
            input_, target = data_[0], data_[1]
            input_ = input_.type(torch.LongTensor)
            target = target.type(torch.LongTensor)
            input_ = input_.to(device)
            target = target.to(device)
            output = model(input_)
            # 经过模型对象就产生了输出
            target = target.squeeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            # print(predicted.shape)
            total += target.size(0)  # 此处的size()类似numpy的shape: np.shape(train_images)[0]
            # print(target.shape)
            correct += (predicted == target).sum().item()
            F1 = f1_score(target.cpu(), predicted.cpu(), average="weighted")
            Recall = recall_score(target.cpu(), predicted.cpu(), average="micro")
            # CM=confusion_matrix(target.cpu(),predicted.cpu())
            postfix = {
                "train_loss: {:.5f},train_acc:{:.3f}%"
                ",F1: {:.3f}%,Recall:{:.3f}%".format(
                    train_loss / (i + 1), 100 * correct / total, 100 * F1, 100 * Recall
                )
            }
            train_dataloader.set_postfix(log=postfix)

        acc = val_accuary(model, val_dataloader, device, criterion)
        if acc > best_acc:
            best_acc = acc
            if os.path.exists(Config.model_state_dict_path) == False:
                os.mkdir(Config.model_state_dict_path)
            torch.save(model, "./word2vec/sen_model_best.pkl")


if __name__ == "__main__":
    # 建立word2id
    word2id = build_word2id("./word2vec/word2id.txt")

    # 建立id2word
    id2word = build_id2word(word2id)

    # 建立word2vec
    w2vec = build_word2vec(
        Config.pre_word2vec_path, word2id, Config.corpus_word2vec_path
    )

    # 得到句子id表示和标签
    (
        train_array,
        train_label,
        val_array,
        val_label,
        test_array,
        test_label,
    ) = prepare_data(
        word2id,
        train_path=Config.train_path,
        val_path=Config.val_path,
        test_path=Config.test_path,
        seq_lenth=Config.max_sen_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = Data_set(train_array, train_label)
    train_dataloader = DataLoader(
        train_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0
    )  # 用了workers反而变慢了

    val_loader = Data_set(val_array, val_label)
    val_dataloader = DataLoader(
        val_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0
    )

    test_loader = Data_set(test_array, test_label)
    test_dataloader = DataLoader(
        test_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0
    )
    # 生成word2vec
    w2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
    w2vec = torch.from_numpy(w2vec)
    w2vec = w2vec.float()  # CUDA接受float32，不接受float64
    model = LSTM_attention(
        Config.vocab_size,
        Config.embedding_dim,
        w2vec,
        Config.update_w2v,
        Config.hidden_dim,
        Config.num_layers,
        Config.drop_keep_prob,
        Config.n_class,
        Config.bidirectional,
    )

    # 训练
    train(
        train_dataloader,
        model=model,
        device=device,
        epoches=Config.n_epoch,
        lr=Config.lr,
    )

    # 保存模型
    if os.path.exists(Config.model_state_dict_path) == False:
        os.mkdir(Config.model_state_dict_path)
    torch.save(model, Config.model_state_dict_path)
