#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch.nn as nn
from custom.cnn import CNN
from custom.highway import Highway

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, char_vectors, embed_size, drop_prob=0.2,
                       char_embed_size=64, char_limit=16, kernel_size=5):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab_size (int): Vocabulary size (i.e. Character tokens numbers).
        """
        super(ModelEmbeddings, self).__init__()

        self.embed_size = embed_size

        self.max_word_len = char_limit
        self.dropout_rate = drop_prob
        self.kernel_size = kernel_size

        self.char_embedding = nn.Embedding.from_pretrained(char_vectors)
        self.char_embed_size = self.char_embedding.embedding_dim

        self.cnn = CNN(
            char_embed_dim=self.char_embed_size,
            word_embed_dim=self.embed_size,
            max_word_length=self.max_word_len,
            kernel_size=self.kernel_size
        )

        self.highway = Highway(
            embed_dim=self.embed_size
        )

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param x: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        # (sentence_length, batch_size, max_word_length)
        x_emb = self.char_embedding(x) # look up char embedding
        sentence_length, batch_size, max_word_length, char_embed_size = x_emb.size()
        # (sentence_length, batch_size, max_word_length, char_embed_size)
        x_reshaped = x_emb.view(sentence_length*batch_size, max_word_length, char_embed_size).permute(0, 2, 1)
        # (sentence_length * batch_size, char_embed_size, max_word_length)
        x_conv = self.cnn(x_reshaped)
        # (sentence_length * batch_size, word_embed_size)
        x_highway = self.highway(x_conv)
        # (sentence_length * batch_size, word_embed_size)
        x_word_emb = self.dropout(x_highway)
        # (sentence_length * batch_size, word_embed_size)
        output = x_word_emb.view(sentence_length, batch_size, -1)
        # (sentence_length, batch_size, word_embed_size)

        return output

