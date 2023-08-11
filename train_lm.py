"""
Trains on a language modeling corpus.
"""


from typing import Dict, Union, Callable, Any, List

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

from util import NoOpContext

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


def train_step(
        model: Union[Callable, torch.nn.Module],
        data: Any,
        optimizer_fn: Callable,
        train_contexts: List=[NoOpContext],
        ignore_index: int=-100) -> None:

    # This is some fancy stuff to allow for changing the number of contexts in future
    # i.e. stacking a mixed precision context with some  other context.
    with contextlib.ExitStack() as stack:
        for context in train_contexts:
            stack.enter_context(context)

        model.zero_grad()

        loss_data = get_loss(model, data, ignore_index)

        loss_data['loss'].backward()

        # We allow the optimizer function to use a loss closure here.
        # if you explicitly don't want to do so, just wrap optimizer.step
        # in a function that ignores its first argument before passing
        # it to train_step like so:
        # optimizer_fn = lambda _: optimizer.step()
        # train_step(model, data, optimizer_fn)
        #
        optimizer_fn(lambda: get_loss(model, data, ignore_index)['loss'])



