# see https://torchmetrics.readthedocs.io/en/latest/
# for more info on torchmetrics.
# At a high level, a metric must implement `update`,
# which can update some state, and `compute`, which actually
# returns the metric.

from composer import Callback
import wandb

from torchmetrics import Metric
import torch

# this is likely not the optimal way to do this...


class Accuracy(Metric):
    # this metric just records the simple accuracy.
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, model, batch, output):
        # arguments are the model, batch, and output from a step of the trainer.
        ignore_index = model.ignore_index
        loss, logits = output
        preds = torch.argmax(logits, dim=-1)

        targets = batch["targets"]
        mask = targets != ignore_index

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
        # arguments are the model, batch, and output from a step of the trainer.
        loss, logits = output
        ignore_index = model.ignore_index
        targets = batch["targets"]
        num_examples = torch.sum(targets != ignore_index)

        self.bits += torch.sum(loss * num_examples)
        self.bytes += torch.sum(batch["tokenized_bytes"])

    def compute(self):
        return self.bits / self.bytes


class Loss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, model, batch, output):
        # arguments are the model, batch, and output from a step of the trainer.
        ignore_index = model.ignore_index
        loss, logits = output

        targets = batch["targets"]
        mask = targets != ignore_index
        num_examples = torch.sum(mask)
        self.loss += loss * num_examples.float()
        self.total += num_examples

    def compute(self):
        return self.loss / self.total
