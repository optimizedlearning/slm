"""
trains a GPT model on "the pile" dataset

Note: this script only works for single-gpu training
lightning does not automatically shard iterable datasets
across many gpus, so if you try with multiple gpus, it will silently
replicate data across the gpus rather than giving different data
to each gpu, resulting in no change vs just one gpu.
"""

# library imports
import hydra
import torch
from omegaconf import OmegaConf, DictConfig
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from transformers import GPT2TokenizerFast
import numpy as np
import datetime

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer

# our code imports
from logging_lightning import LogTrainMetrics
import loss_utils
from gpt import GPT
from load_pile import get_dataloaders


# This is the pytorch lightning module. In an effort to isolate lightning code,
# the main model class "GPT" is as ordinary pytorch.nn.Module.
class Model(pl.LightningModule):
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

        # this will automatically upload the argument to wandb.
        # wandb does not know how to properly display the hydra config object,
        # in the web gui, so we convert it into a series of nested dicts/lists first.
        self.save_hyperparameters(OmegaConf.to_container(config))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # For straightforward training loops like this one, pytorch lightning will
        # handle all of the standard steps (running loss.backward, zeroing gradients,
        # taking an optimizer step).
        # In this case, `training_step` only needs to return either the loss, or
        # a dictionary with 'loss' as a key where 'loss' is the training loss. We opt
        # for the latter.
        # If you do want to manually control the loss.backward, optimizer.step etc, then
        # you can set self.automatic_optimizer=False in __init__() and put the training
        # logic in this training_step function, as described here:
        # https://lightning.ai/docs/pytorch/latest/model/manual_optimization.html
        loss_data = loss_utils.get_loss_data(
            self.model, batch, ignore_index=self.ignore_index
        )
        # we don't need the logits, and they stick around in between
        # batches and cause OOM errors.
        del loss_data["logits"]

        return loss_data

    def validation_step(self, batch, batch_idx):
        loss_data = loss_utils.get_loss_data(
            self.model, batch, ignore_index=self.ignore_index
        )
        # we don't need the logits, and they stick around in between
        # batches and cause OOM errors.
        del loss_data["logits"]
        return loss_data

    def configure_optimizers(self):
        # this function tells pytorch lightning which optimizer to use.
        # the optimizer returned from this function
        # can later be accessed via self.optimizers().optimizer.
        # See https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        # for description of the return values.
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        def lr_schedule(step):
            scale = 1.0
            warmup_ratio = self.config.train.lr_warmup / self.config.train.max_steps
            current_ratio = step / self.config.train.max_steps
            if self.config.train.lr_warmup > 0:
                scale *= min(1.0, current_ratio / warmup_ratio)

            decay_type = self.config.train.lr_decay
            if decay_type == "linear":
                scale *= 1.0 - current_ratio
            if decay_type == "cosine":
                scale *= 0.5 * (1.0 + np.cos(np.pi * current_ratio))
            return scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update the scheduler every step rather than every epoch.
                "frequency": 1,
            },
        }


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt")
def train(config: DictConfig) -> None:
    # hydra will load the config argument from conf/config_gpt.
    # see https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/
    # you can override config values from the commandline like so:
    # python gpt_pile.py train.max_steps=100000 model.num_blocks=6

    print(OmegaConf.to_yaml(config))

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

    # define lightning module
    model = Model(config=config, tokenizer=tokenizer)

    # wandb logger, as wrapped by pytorch lightning. This will make the wand logging
    # object accessible as self.log in the pytorch module.
    # it will also take care of initializing and tearing down the wandb logging process.
    wandb_logger = WandbLogger(project=config.wandb.project)

    # start setting up callbacks. These are basically sets of functions that pytorch lightning will
    # call for us at appropriate times.
    callbacks = []

    # this callback will automatically save checkpoints for us.
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.path,
        train_time_interval=datetime.timedelta(
            minutes=config.checkpoint.frequency_mins
        ),
        save_top_k=config.checkpoint.num_to_keep,
        filename=config.checkpoint.name_format,
    )
    callbacks.append(checkpoint_callback)

    # this callback will cause logging of the learning rate from our scheduler
    callbacks.append(LearningRateMonitor())

    # This is a custom callback implemented in util.py.
    # it takes care of logging loss, iteration speed, and other metrics we
    # might want to track.
    callbacks.append(LogTrainMetrics())

    if config.train.max_time_hours is not None:
        # this callback will stop the training after the specified number of hours.
        callbacks.append(Timer({"hours": config.train.max_time_hours}))

    if config.train.compile:
        # we use the default compile mode.
        # See compile mode descriptions here: https://pytorch.org/get-started/pytorch-2.0/#user-experience
        # Using 'max-autotune' seems to  increase memory usage, especially on multiple GPUs.
        # In my experiments, 1 V100 GPU could run GPT2 with a batch size of 8,
        # but 2 GPUs would OOM with a batch size of 8 per GPU if mode='max-autotune'
        model = torch.compile(model, mode="default")

    # define the trainer object. See
    # https://lightning.ai/docs/pytorch/latest/common/trainer.html#trainer
    trainer = pl.Trainer(
        max_steps=config.train.max_steps,
        precision=config.train.precision,
        gradient_clip_val=config.train.gradient_clip_val,
        gradient_clip_algorithm=config.train.gradient_clip_algorithm,
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=config.train.val_check_interval,
        limit_val_batches=config.train.val_batches,
    )

    # now we can train!
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )


if __name__ == "__main__":
    train()
