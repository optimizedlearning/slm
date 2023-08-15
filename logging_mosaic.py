

from composer import Callback
import wandb

from torchmetrics import Metric
import torch

# this is likely not the optimal way to do this...

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, model, batch, output):
        ignore_index = model.ignore_index
        loss, logits = output
        preds = torch.argmax(logits, dim=-1)

        targets = batch['targets']
        mask = (targets != ignore_index)
        
        self.correct += torch.sum((preds == targets) * mask)
        self.total += torch.sum(mask)

    def compute(self):
        return self.correct.float() / self.total


class BitsPerByte(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("bits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bytes", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, model, batch, output):
        # bits, bytes = self._input_format(preds, target)
        loss, logits = output
        ignore_index = model.ignore_index
        targets = batch['targets']
        num_examples = torch.sum(targets != ignore_index)

        self.bits += torch.sum(loss * num_examples)
        self.bytes += torch.sum(batch['tokenized_bytes'])

    def compute(self):
        return self.bits / self.bytes


class Loss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, model, batch, output):
        ignore_index = model.ignore_index
        loss, logits = output

        targets = batch['targets']
        mask = (targets != ignore_index)
        num_examples = torch.sum(mask)
        self.loss += loss * num_examples.float()
        self.total += num_examples

    def compute(self):
        return self.loss / self.total
