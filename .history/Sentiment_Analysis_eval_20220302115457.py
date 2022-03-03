# !usr/bin/env python
# -*- coding:utf-8 -*-

""" 
@Author: HsuDan
@Date: 2022-02-18 19:12:58
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-25 11:19:09
@Description: model evaluation 
@FilePath: /Sentiment-Analysis-Chinese-pytorch/Sentiment_Analysis_eval.py
"""
from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import os
from Sentiment_model import LSTMModel, LSTM_attention
from Sentiment_Analysis_Config import Config
from Sentiment_Analysis_DataProcess import (
    build_word2id,
    build_id2word,
    build_word2vec,
    prepare_data,
    build_word2vec,
    text_to_array_nolabel,
    Data_set,
)


def val_accuary(model, val_dataloader, device, criterion):
    """
    @description: get the accuracy of validation set
    @param {*}
    - model: Obeject, the Sentiment Analysis model (Seq2Seq).
    - val_dataloader: DataLoader, validation DataLoader
    - device: str, choose torch device {"cpu", "cuda"}
    - criterion: Object, criterion for calculating the loss
    @return {*}
    val_accuracy: the accuracy of total validation set
    """
    model = model.to(device)
    with torch.no_grad():
        correct = 0  # 预测的和实际的label相同的样本个数
        total = 0  # 累计validation样本个数
        val_loss = 0.0
        for j, val_data in enumerate(val_dataloader, 0):
            input_val, target_val = val_data[0], val_data[1]
            input_val = input_val.type(torch.LongTensor)
            target_val = target_val.type(torch.LongTensor)
            target_val = target_val.squeeze(1)
            input_val = input_val.to(device)
            target_val = target_val.to(device)
            output_val = model(input_val)
            loss = criterion(output_val, target_val)
            val_loss += loss.item()
            _, pred_val = torch.max(output_val, 1)
            total += target_val.size(0)
            correct += (pred_val == target_val).sum().item()
            F1 = f1_score(target_val.cpu(), pred_val.cpu(), average="weighted")
            Recall = recall_score(target_val.cpu(), pred_val.cpu(), average="micro")
            CM = confusion_matrix(target_val.cpu(), pred_val.cpu())
        print(
            "\nVal accuracy : {:.3f}%,val_loss:{:.3f}, F1_score：{:.3f}%, Recall：{:.3f}%".format(
                100 * correct / total, val_loss, 100 * F1, 100 * Recall
            )
        )
        return 100 * correct / total


def test_accuary(model, test_dataloader, device):
    """
    @description: get the accuracy of test set
    @param {*}
    - model: Obeject, the Sentiment Analysis model (Seq2Seq).
    - test_dataloader: DataLoader, test DataLoader
    - device: str, choose torch device {"cpu", "cuda"}
    @return {*}
    test_accuracy: the accuracy of total test set
    """
    model = model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for k, data_test in enumerate(test_dataloader, 0):
            input_test, target_test = data_test[0], data_test[1]
            input_test = input_test.type(torch.LongTensor)
            target_test = target_test.type(torch.LongTensor)
            target_test = target_.squeeze(1)
            input_test = input_test.to(device)
            target_ = target_.to(device)
            output2 = model(input_test)
            _, predicted_test = torch.max(output2, 1)
            total += target_.size(
                0
            )  # 此处的size()类似numpy的shape: np.shape(train_images)[0]
            correct += (predicted_test == target_).sum().item()
            F1 = f1_score(target_.cpu(), predicted_test.cpu(), average="weighted")
            Recall = recall_score(target_.cpu(), predicted_test.cpu(), average="micro")
            CM = confusion_matrix(target_.cpu(), predicted_test.cpu())
        print(
            "test accuracy : {:.3f}%, F1_score：{:.3f}%, Recall：{:.3f}%,Confusion_matrix：{}".format(
                100 * correct / total, 100 * F1, 100 * Recall, CM
            )
        )


def pre(word2id, model, seq_lenth, path):
    model.cpu()
    with torch.no_grad():
        input_array = text_to_array_nolabel(word2id, seq_lenth, path)
        # sen_p = sen_p.type(torch.LongTensor)
        sen_p = torch.from_numpy(input_array)
        sen_p = sen_p.type(torch.LongTensor)
        output_p = model(sen_p)
        _, pred = torch.max(output_p, 1)
        for i in pred:
            print("预测类别为", i.item())


if __name__ == "__main__":
    # 建立word2id
    word2id = build_word2id("./word2vec/word2id.txt")

    # 建立id2word
    id2word = build_id2word(word2id)

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
    # 读取模型
    # model1 = torch.load(Config.model_state_dict_path)
    model = torch.load("./model/sen_model_best.pkl")

    # model.load_state_dict(torch.load(Config.model_state_dict_path)) #仅保存参数
    # 验证
    # val_accuary(model1, val_dataloader, device)
    # 测试
    test_accuary(model, test_dataloader, device)
    # 预测
    pre(word2id, model, Config.max_sen_len, Config.pre_path)
