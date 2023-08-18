import os
import time
import logging


import numpy as np
import hydra
import torch
from omegaconf import OmegaConf, DictConfig

import composer
from composer import Trainer
from composer.models import ComposerModel
from composer.algorithms import GradientClipping
from composer.callbacks import (
    SpeedMonitor,
    LRMonitor,
    OptimizerMonitor,
    MemoryMonitor,
    RuntimeEstimator,
)

from transformers import GPT2TokenizerFast

# our code imports
from logging_composer import Accuracy, BitsPerByte, Loss
from loss_utils import get_only_loss_from_logits
from gpt import GPT
from load_text_streaming import get_dataloaders
from wandb_logger_autoresume import WandBLoggerWithAutoResume
from progress_bar_autoresume import ProgressBarWithAutoResume


class Model(ComposerModel):
    def __init__(self, config, tokenizer, model_class=GPT):
        super().__init__()

        # load "actual" model - this class is basically a wrapper around
        # this inner model.
        self.model = model_class(vocab_size=len(tokenizer), config=config.model)
        self.config = config

        # in pytorch's cross_entropy loss, the field 'ignore_index' is a value that
        # we use to tell pytorch to not compute the loss for certain items.
        # Specifically, if the target id for any classification item is equal to
        # ignore_index, then we will not compute the loss for this target.
        self.ignore_index = tokenizer(tokenizer.pad_token)["input_ids"][0]

    def forward(self, batch):
        # batch is the output of the dataloader, so we need to process it
        # to get the token indices to provde to self.model

        # This function is allowed to return basically anything: its output
        # will be provided as the first input to self.loss.
        token_indices = batch["input_ids"]  # [B, L]
        logits = self.model(token_indices)
        loss = get_only_loss_from_logits(logits, batch, self.ignore_index)

        return loss, logits

    # This torchmetrics stuff seems overly complicated, but apparently it might help
    # with doing metrics properly in distributed settings, so we'll try to make it work.
    def get_metrics(self, is_train):
        """
        outputs a dictionary of TorchMetrics objects to be logged in train and eval
        """
        # see https://docs.mosaicml.com/projects/composer/en/stable/composer_model.html#metrics
        metrics = {"accuracy": Accuracy(), "loss": Loss()}

        if self.config.train.log_bits_per_byte:
            metrics["bits_per_byte"] = BitsPerByte()

        return metrics

    def update_metric(self, batch, outputs, metric):
        # here we need to call the update function of the given metric
        # Unfortunately, we do not actually know the name of the metric,
        # so the simplest thing is to just have all metrics take the same
        # arguments in the update function and have them sort out themselves
        # what to do with those arguments. It means that this function is
        # basically a no-op, and some metrics will get more information
        # than they really need.
        metric.update(self, batch, outputs)

    def loss(self, loss_logits, batch):
        """
        compute the loss.

        First argument is the output of self.forward.
        Second output is the batch (also the input to self.forward).
        """
        loss, logits = loss_logits
        return loss


def get_scheduler(config, optimizer):
    def lr_schedule(step):
        """
        a scale factor to scale the lr as a function of the step count.
        """
        scale = 1.0
        warmup_ratio = config.train.lr_warmup / config.train.max_steps
        current_ratio = step / config.train.max_steps
        if config.train.lr_warmup > 0:
            scale *= min(1.0, current_ratio / warmup_ratio)

        decay_type = config.train.lr_decay
        if decay_type == "linear":
            scale *= 1.0 - current_ratio
        if decay_type == "cosine":
            scale *= 0.5 * (1.0 + np.cos(np.pi * current_ratio))
        return scale

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    return scheduler


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt")
def train(config: DictConfig) -> None:
    # hydra will load the config argument from conf/config_gpt.
    # see https://hydra.cc/docs/tutorials/basic/your_first_app/
    # you can override config values from the commandline like so:
    # python gpt_pile.py train.max_steps=100000 model.num_blocks=6

    # make the composer framework print out all the logs
    logging.getLogger("composer").setLevel("DEBUG")

    logging.info(OmegaConf.to_yaml(config))

    logging.debug(OmegaConf.to_yaml(dict(os.environ)))

    # Load GPT tokenizer.
    # We need to specify a padding token, so we will use '<|pad|>'
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # define dataloaders
    train_loader, eval_loader = get_dataloaders(
        "/projectnb/aclab/datasets/pile/mds", config, tokenizer
    )

    # define lightning module
    model = Model(config=config, tokenizer=tokenizer)

    # wandb logger, wrapped as a composer LoggerDestination. This will make
    # the metrics in model.get_metrics be logged automatically to wandb.
    # it will also take care of initializing and tearing down the wandb logging process.
    wandb_logger = WandBLoggerWithAutoResume(
        project=config.wandb.project,
        resume=config.run_name is not None,
    )

    # The default progress bar implementation doesn't work properly when resuming
    # from a mid-epoch checkpoint. This mildly customized version does.
    # Note that the trainer will complain at you for using, but it should
    # be a drop-in replacement.
    progress_bar = ProgressBarWithAutoResume()

    # start setting up callbacks. These are basically sets of functions that the
    # Trainer will call for us at appropriate times.
    callbacks = [
        LRMonitor(),
        SpeedMonitor(),
        OptimizerMonitor(),
        MemoryMonitor(),
        RuntimeEstimator(),
    ]

    if config.train.compile:
        # we use the default compile mode.
        # See compile mode descriptions here: https://pytorch.org/get-started/pytorch-2.0/#user-experience
        # Using 'max-autotune' seems to  increase memory usage, especially on multiple GPUs.
        # In my experiments, 1 V100 GPU could run GPT2 with a batch size of 10,
        # but 2 GPUs would OOM with a batch size of 8 per GPU if mode='max-autotune'
        compile_config = {"mode": "default"}
    else:
        compile_config = None

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    scheduler = get_scheduler(config, optimizer)

    algorithms = []
    algorithms.append(
        GradientClipping(
            clipping_type=config.train.gradient_clip_algorithm,  # 'norm', 'adaptive', or 'value'
            clipping_threshold=config.train.gradient_clip_val,
        )
    )

    # define the trainer object. See
    # https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.Trainer.html
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        max_duration=f"{config.train.max_steps}ba",
        eval_dataloader=eval_loader,
        loggers=[progress_bar],
        optimizers=optimizer,
        schedulers=scheduler,
        step_schedulers_every_batch=True,
        eval_interval=f"{config.train.val_check_interval}ba",
        eval_subset_num_batches=config.train.val_batches,
        precision=config.train.precision,
        algorithms=algorithms,
        callbacks=callbacks,
        compile_config=compile_config,
        run_name=config.run_name,
        save_folder="checkpoints/{run_name}",
        save_interval=f"{config.checkpoint.frequency_batches}ba",
        save_num_checkpoints_to_keep=config.checkpoint.num_to_keep,
        save_filename=config.checkpoint.name_format
        or "ep{epoch}-ba{batch}-rank{rank}.pt",
        autoresume=config.run_name is not None,
        console_log_interval="10ba",
    )

    # upload the config
    trainer.logger.log_hyperparameters(OmegaConf.to_container(config))

    # now we can train!
    # this should take care of scaling to multiple gpus if available as well.
    trainer.fit()


def patch_dist_init():
    # HACK: There appears to be some kind of race condition when using
    # multiprocessing. With 2 processes, the rank 1 process will frequently
    # error with the error:
    # RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
    # when hitting a barrier synchronization.
    # It's unclear what exactly causes this, but if we make sure that the
    # non-rank 0 processes get a "delayed start" so that the rank 0 one
    # hits the barrier first, then things seem to be ok.
    # NOTE: This has only been tested in 2 processes.

    # It sometimes also has error: RuntimeError: Broken pipe
    # occuring in _store_based_barrier when initializing torch.distributed.
    # This race seems less common, so it is as yet unlear if this hack also
    # fixes it or not.
    import torch

    old_init_process_group_fn = torch.distributed.init_process_group

    def new_init_process_group_fn(*args, **kwargs):
        result = old_init_process_group_fn(*args, **kwargs)
        if composer.utils.dist.get_local_rank() != 0:
            time.sleep(1)
        return result

    torch.distributed.init_process_group = new_init_process_group_fn


if __name__ == "__main__":
    patch_dist_init()
    train()
