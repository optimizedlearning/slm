from typing import Dict, Union, Callable, Any, List

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

# Tell torch.compile that rearrange should not break the graph
# https://github.com/pytorch/pytorch/issues/93905
# if you upgrade versions enough this is probably unnecessary: https://github.com/arogozhnikov/einops/issues/250
from torch._dynamo import allow_in_graph
allow_in_graph(rearrange)

import lightning.pytorch as pl

def get_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100) -> torch.Tensor:
    mask = (targets != ignore_index)
    correct = torch.sum(mask * (torch.argmax(logits, dim=-1) == targets))
    return correct / torch.sum(mask)

def get_loss_data(
        model: Union[Callable, torch.nn.Module],
        batch: Any,
        ignore_index: int=-100) -> Dict:
    logits = get_output_from_batch(model, batch)
    return get_loss_data_from_logits(logits, batch, ignore_index)



def get_output_from_batch(
        model: Union[Callable, torch.nn.Module],
        batch: Any,
        ignore_index: int=-100) -> Dict:
    input = batch['input_ids']    # [B, L]
    targets = batch['targets'] # [B, L]

    logits = model(input)    # [B, L, C]
    return logits


def get_only_loss_from_logits(
        logits: torch.Tensor,
        batch: Any,
        ignore_index: int=-100) -> Dict:

    targets = batch['targets'] # [B, L]
    # logits should be [B, L, C]
    
    targets = rearrange(targets, 'B L -> (B L)')
    logits = rearrange(logits, 'B L C -> (B L) C')


    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    return loss


def get_loss_data_from_logits(
        logits: torch.Tensor,
        batch: Any,
        ignore_index: int=-100) -> Dict:

    targets = batch['targets'] # [B, L]
    # logits should be [B, L, C]
    
    targets = rearrange(targets, 'B L -> (B L)')
    logits = rearrange(logits, 'B L C -> (B L) C')


    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)


    accuracy = get_accuracy(logits, targets, ignore_index)

    result = {
        'logits': logits,
        'loss': loss,
        'accuracy': accuracy,
    }

    if 'tokenized_bytes' in batch:
        # If we were to compress the input using the predicted logits
        # and an arithmetic coder, then the total compressed length of a sequence 
        # (in bits) is simply the cross entropy loss (total loss, not average loss).
        # So, here we record the compression ratio in terms of number of average bits
        # to compress a byte of the original sequence. This metric has the advantage that
        # it is more "tokenizer independent", unlike loss and accuracy metrics.
        total_bytes = torch.sum(batch['tokenized_bytes'])
        result['bits_per_byte'] = (loss * torch.sum(targets != ignore_index)) / total_bytes
        


    return result



