import hydra
import torch
from omegaconf import OmegaConf, DictConfig
import numpy as np
import os

import logging

from composer import Trainer
from composer.models import ComposerModel
from composer.loggers import WandBLogger
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
from logging_mosaic import Accuracy, BitsPerByte, Loss
from train_lm import get_only_loss_from_logits
from gpt import GPT
from load_pile import get_dataloaders
from wandb_logger_autoresume import WandBLoggerWithAutoResume


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
        """
        batch is the output of the dataloader, so we need to process it
        to get the token indices to provde to self.model
        """
        token_indices = batch["input_ids"]  # [B, L]
        logits = self.model(token_indices)
        loss = get_only_loss_from_logits(logits, batch, self.ignore_index)

        return loss, logits

    # This torchmetrics stuff seems overly complicated, but apparently it might help
    # with doing metrics properly in distributed settings, so we'll try to make it work.
    def get_metrics(self, is_train):
        # see https://docs.mosaicml.com/projects/composer/en/stable/composer_model.html#metrics
        metrics = {"accuracy": Accuracy(), "loss": Loss()}

        if self.config.train.log_bits_per_byte:
            metrics["bits_per_byte"] = BitsPerByte()

        return metrics

    def update_metric(self, batch, outputs, metric):
        metric.update(self, batch, outputs)

    def loss(self, loss_logits, batch):
        loss, logits = loss_logits
        return loss

def get_scheduler(config, optimizer):
    def lr_schedule(step):
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
    logging.getLogger("composer").setLevel("DEBUG")

    # hydra will load the config argument from conf/config_gpt.
    # see https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/
    # you can override config values from the commandline like so:
    # python gpt_pile.py train.max_steps=100000 model.num_blocks=6

    logging.info(OmegaConf.to_yaml(config))

    logging.debug(OmegaConf.to_yaml(dict(os.environ)))

    # Load GPT tokenizer.
    # We need to specify a padding token, so we will use '<|pad|>'
    # In any event, I think ignore_index loss masking will make the padding token
    # actually irrelevant.
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # define dataloaders
    train_loader, valid_loader = get_dataloaders(
        "/projectnb/aclab/datasets/pile/mds", config, tokenizer
    )
    # train_loader, valid_loader = get_dataloaders(config, tokenizer)

    # define lightning module
    model = Model(config=config, tokenizer=tokenizer)

    # wandb logger, as wrapped by pytorch lightning. This will make the wand logging
    # object accessible as self.log in the pytorch module.
    # it will also take care of initializing and tearing down the wandb logging process.
    # TODO: currently if the run is resumed, the wandb_logger will NOT resume from the
    # same run id. This is mildly annoying to fix with the "autoresume" option to
    # Trainer because the checkpoint file is identified and loaded AFTER the wandb logger
    # is initialized. However, the wandb run id is indeed stored in the checkpoint, so if we
    # manually lookup the checkpoint file at this point, we would be able to load.
    wandb_logger = WandBLoggerWithAutoResume(
        project=config.wandb.project,
        resume=config.run_name is not None,
    )

    # start setting up callbacks. These are basically sets of functions that the
    # Trainer will call for us at appropriate times.
    callbacks = [LRMonitor(), SpeedMonitor(), OptimizerMonitor(), MemoryMonitor(), RuntimeEstimator()]

    # if config.train.max_time_hours is not None:
    #     # this callback will stop the training after the specified number of hours.
    #     callbacks.append(Timer({'hours': config.train.max_time_hours}))

    if config.train.compile:
        # we use the default compile mode.
        # See compile mode descriptions here: https://pytorch.org/get-started/pytorch-2.0/#user-experience
        # Using 'max-autotune' seems to  increase memory usage, especially on multiple GPUs.
        # In my experiments, 1 V100 GPU could run GPT2 with a batch size of 8,
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
        eval_dataloader=valid_loader,
        loggers=[wandb_logger],
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
        console_log_interval='10ba',
    )

    # upload the config
    trainer.logger.log_hyperparameters(OmegaConf.to_container(config))

    # now we can train!
    # this should take care of scaling to multiple gpus if available as well.
    trainer.fit()


if __name__ == "__main__":
    train()
