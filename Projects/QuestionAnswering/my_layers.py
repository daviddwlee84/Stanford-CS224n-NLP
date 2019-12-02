"""Assortment of layers for use in my_models.py.

Author:
    David Lee (dwlee@pku.edu.cn)
"""

from layers import Embedding as WordEmbedding
from custom.model_embeddings import ModelEmbeddings as CharEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingWithChar(nn.Module):
    """Embedding layer used by BiDAF, with the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Initial char vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(EmbeddingWithChar, self).__init__()
        self.word_embed = WordEmbedding(word_vectors, hidden_size//2, drop_prob)
        self.char_embed = CharEmbedding(char_vectors, hidden_size//2, drop_prob)

    def forward(self, w_idxs, c_idxs):
        word_emb = self.word_embed(w_idxs)   # (batch_size, seq_len, hidden_size//2)
        char_emb = self.char_embed(c_idxs)   # (batch_size, seq_len, hidden_size//2)

        emb = torch.cat([word_emb, char_emb], dim=2)

        return emb
