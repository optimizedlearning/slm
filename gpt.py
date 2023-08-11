"""
GPT model, based on the papers:
[GPT1] Improving Language Understanding by Generative Pre-Training
[GPT2] Language Models are Unsupervised Multitask Learners
[GPT3] Language Models are Few-Shot Learners
[Att] Attention is All You Need
"""


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import lightning.pytorch as pl

from einops import rearrange


class CausalSelfAttention(nn.Module):
    """
    simple attention class.
    """

    def __init__(self, config):
        super().__init__()

        self.dim = config.dim
        self.num_heads = config.num_heads

        dim = self.dim
        num_heads = self.num_heads

        self.key_linear = nn.Linear(dim, dim, bias=config.bias)
        self.query_linear = nn.Linear(dim, dim, bias=config.bias)
        self.value_linear = nn.Linear(dim, dim, bias=config.bias)

    def forward(self, data):
        """
        data is [B, L, D]
        """

        keys = self.key_linear(data)  # [B, L, D] -> [B, L, D]
        queries = self.query_linear(data)  # [B, L, D] -> [B, L, D]
        values = self.value_linear(data)  # [B, L, D] -> [B, L, D]

        keys = rearrange(keys, "B L (H D) -> B H L D", H=self.num_heads)
        queries = rearrange(queries, "B L (H D) -> B H L D", H=self.num_heads)
        values = rearrange(values, "B L (H D) -> B H L D", H=self.num_heads)

        out = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True)

        out = rearrange(out, "B H N D -> B N (H D)")

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.fc_dim = config.get("fc_dim", 4 * self.dim)
        if self.fc_dim is None:
            self.fc_dim = 4 * self.dim

        self.attn = CausalSelfAttention(config)

        self.ln1 = nn.LayerNorm(self.dim)
        self.ln2 = nn.LayerNorm(self.dim)

        self.expand_fc = nn.Linear(self.dim, self.fc_dim, bias=config.bias)
        self.reduce_fc = nn.Linear(self.fc_dim, self.dim, bias=config.bias)

        #
        # [GPT2] says:
        # "
        # A modified initialization which accounts
        # for the accumulation on the residual path with model depth
        # is used. We scale the weights of residual layers at
        # initialization by a factor of 1/\sqrt{N} where N is the number
        # of residual layers.
        # "
        # This is pretty vague: what exactly is a residual layer? I will
        # choose to interpret this as only the last layer of the MLP of the
        # transformer block, but it could just as well also mean the value
        # matrices in the attention layers, or even just all weights in the
        # transformer block.
        #
        with torch.no_grad():
            self.reduce_fc.weight *= 1.0 / np.sqrt(config.num_blocks)

    def forward(self, data):
        """
        data is [..., N, D]
        """
        # Order of these operations described in section 2.3 of [GPT2].
        # That text unfortunately does not actually describe the model,
        # but provides a "diff" from the model described  in Fig 1 of [GPT2]
        # The best reference I can find for the MLP using 4x the input dim as
        # the intermediate layer is [Att]
        #
        out = self.ln1(data)
        out = self.attn(data)
        post_attn = data + out
        out = self.ln2(post_attn)
        out = self.expand_fc(out)
        out = F.gelu(out)
        out = self.reduce_fc(out)
        out = post_attn + out

        return out


class GPT(pl.LightningModule):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        # config = config.model
        self.vocab_size = vocab_size
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_blocks)]
        )

        self.token_embedding = nn.Embedding(vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.context_length, config.dim)

        self.ln = nn.LayerNorm(config.dim)

        self.head = nn.Linear(config.dim, vocab_size, bias=config.bias)

    def forward(self, token_indices):

        device = token_indices.device

        B, L = token_indices.size()

        

        # From Equation (2) of [GPT1], we only do tokens + positions here and
        # then no other position encoding to get GPT-1. The descriptions of
        # GPT-2 and GPT-3 do not say to alter this part.
        # No idea if current GPT-4 type stuff does more fancy things, but even
        # the results in [GPT3] looked amazing.

        tokens = self.token_embedding(token_indices)
        positions = self.position_embedding(
            torch.arange(L, dtype=torch.long, device=device)
        )
        data = tokens + positions

        out = self.blocks(data)

        # [GPT2] says to add an extra layer normalization after the transformer
        # blocks. [GPT3] doesn't really say to change anything - it just makes
        # the model bigger.
        out = self.ln(out)

        logits = self.head(out)

        return logits

        