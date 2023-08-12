"""
Trains on a language modeling corpus.
"""


from typing import Dict, Union, Callable, Any, List

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

import lightning.pytorch as pl

import contextlib

def get_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100) -> torch.Tensor:
    mask = (targets != ignore_index)
    correct = torch.sum(mask * (torch.argmax(logits, dim=-1) == targets))
    return correct / torch.sum(mask)

def get_loss(
        model: Union[Callable, torch.nn.Module],
        data: Any,
        ignore_index: int=-100) -> Dict:
    input = data['input_ids']    # [B, L]
    targets = data['targets'] # [B, L]

    logits = model(input)    # [B, L, C]

    targets = rearrange(targets, 'B L -> (B L)')
    logits = rearrange(logits, 'B L C -> (B L) C')

    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)

    accuracy = get_accuracy(logits, targets, ignore_index)

    result = {
        'logits': logits,
        'loss': loss,
        'accuracy': accuracy,
    }

    if 'chunked_bytes' in data:
        # If we were to compress the input using the predicted logits
        # and an arithmetic coder, then the total compressed length of a sequence 
        # (in bits) is simply the cross entropy loss (total loss, not average loss).
        # So, here we record the compression ratio in terms of number of average bits
        # to compress a byte of the original sequence. This metric has the advantage that
        # it is more "tokenizer independent", unlike loss and accuracy metrics.
        total_bytes = torch.sum(data['chunked_bytes'])
        result['bits_per_byte'] = (loss * torch.sum(targets != ignore_index)) / total_bytes
        


    return result



