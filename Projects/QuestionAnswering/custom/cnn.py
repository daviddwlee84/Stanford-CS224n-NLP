#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    """ To combine the character embeddings """

    def __init__(self, char_embed_dim: int, # e_char
                       word_embed_dim: int, # e_word (set filter number to be equal to e_word)
                       max_word_length: int=21, # (m_word) max word length
                       kernel_size: int=5): # window size
        super(CNN, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=word_embed_dim, # number of filter, output feature
            kernel_size=kernel_size,
            bias=True)

        # MaxPool simply takes the maximum across the second dimension
        self.maxpool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x):
        # (batch size, char embedding size, max word length)
        x_conv = self.conv1d(x)
        # (batch size, word embedding size, max_word_length - kernel_size + 1)
        x_conv_out = self.maxpool(torch.relu(x_conv)).squeeze()
        # (batch size, word embedding size)

        return x_conv_out

