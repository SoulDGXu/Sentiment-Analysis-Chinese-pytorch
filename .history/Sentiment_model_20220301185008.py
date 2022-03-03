# !usr/bin/env python
# -*- coding:utf-8 -*-

""" 
@Author: HsuDan
@Date: 2022-02-18 19:12:58
@Version: 1.0
@LastEditors: HsuDan
@LastEditTime: 2022-02-25 11:18:51
@Description: build model: LSTM, LSTM+attention 
@FilePath: /Sentiment-Analysis-Chinese-pytorch/Sentiment_model.py
"""
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_weight,
        update_w2v,
        hidden_dim,
        num_layers,
        drop_keep_prob,
        n_class,
        bidirectional,
        **kwargs
    ):
        """
        @description: initial seetings
        @param {*}
        - vocab_size: int, vocabulary size.
        - embedding_dim: int, the embedding layer dim.
        - pretrained_weight: Tensor, FloatTensor containing weights for the Embedding.
        - update_w2v: bool, whether to update word2vec embedding weight.
        - hidden_dim: int, the hidden layer dim.
        - num_layers: int, the number of layers.
        - drop_keep_prob: float, the keep probability of dropout layer.
        - n_class: int, the number of classes (labels).
        - bidirectional: bool, whether to use bidirectional LSTM.
        @return {*}
        None
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=drop_keep_prob,
        )

        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):
        # inputs: (num_embeddings, embedding_dim) => [batch, seq_len, embed_dim]:[64,75,50]
        embeddings = self.embedding(inputs)
        # LSTM input:(seq, batch, input_size) => [seq_len, batch, embed_dim]:[75,64,50]
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        # states: ( (D∗num_layers,N,Hout)
        encoding = torch.cat([states[0], states[-1]], dim=1)  # 张量拼接[32,512]
        outputs = self.decoder1(encoding)
        # outputs = F.softmax(outputs, dim=1)
        outputs = self.decoder2(outputs)
        return outputs


class LSTM_attention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_weight,
        update_w2v,
        hidden_dim,
        num_layers,
        drop_keep_prob,
        n_class,
        bidirectional,
        **kwargs
    ):
        super(LSTM_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=drop_keep_prob,
        )

        # TODO
        # What is nn. Parameter ? Explain
        self.weight_W = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))

        if self.bidirectional:
            # self.decoder1 = nn.Linear(hidden_dim * 2, n_class)
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):
        embeddings = self.embedding(
            inputs
        )  # [batch, seq_len] => [batch, seq_len, embed_dim][64,75,50]

        states, hidden = self.encoder(
            embeddings.permute([0, 1, 2])
        )  # [batch, seq_len, embed_dim]
        # attention

        u = torch.tanh(torch.matmul(states, self.weight_W))
        att = torch.matmul(u, self.weight_proj)

        att_score = F.softmax(att, dim=1)
        scored_x = states * att_score
        encoding = torch.sum(scored_x, dim=1)
        outputs = self.decoder1(encoding)
        outputs = self.decoder2(outputs)
        return outputs
