#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h

class Highway(nn.Module):
    """ Highway Networks6 have a skip-connection controlled by a dynamic gate """

    def __init__(self, embed_dim: int): # word embedding dimension
        super(Highway, self).__init__()
        
        self.conv_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x_conv_out):
        x_proj = torch.relu(self.conv_out_proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))

        x_highway = x_gate * x_conv_out + (1 - x_gate) * x_conv_out

        return x_highway

### END YOUR CODE 

