"""
trains a GPT model on "the pile" dataset
"""

# library imports
import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from datasets import load_dataset
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from load_text import load_next_token_prediction
from transformers import GPT2TokenizerFast
import numpy as np
import datetime

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer

# our code imports
from util import LogTrainMetrics
import train_lm
from gpt import GPT

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
        return  self.model(*args, **kwargs)


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
        loss_data = train_lm.get_loss(self.model, batch, ignore_index=self.ignore_index)
        # we don't need the logits, and they stick around in between
        # batches and cause OOM errors.
        del loss_data['logits']

        return loss_data

    def validation_step(self, batch, batch_idx):
        loss_data = train_lm.get_loss(self.model, batch, ignore_index=self.ignore_index)
        # we don't need the logits, and they stick around in between
        # batches and cause OOM errors.
        del loss_data['logits']
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
            warmup_ratio = self.config.train.lr_warmup/self.config.train.max_steps
            current_ratio = step/self.config.train.max_steps
            if self.config.train.lr_warmup > 0:
                scale *= min(1.0, current_ratio/warmup_ratio)

            decay_type = self.config.train.lr_decay
            if decay_type == "linear":
                scale *= 1.0 - current_ratio
            if decay_type == "cosine":
                scale *= 0.5 * (
                    1.0 + np.cos(np.pi * current_ratio)
                )
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


def get_dataloaders(config: DictConfig, tokenizer) -> (DataLoader, DataLoader):
    # load train and validation datasets.
    # TODO: maybe consider https://github.com/mosaicml/streaming instead?
    # seems like it would be better when implementing resuming interrupted
    # training, but may require the data to be actually stored somewhere more
    # easily.
    train_dataset = load_next_token_prediction(
        config.dataset.path,
        config.dataset.name,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
        split="train",
        text_key="text",
        # we may not want to do the whole validation set, so
        # let's just do the first few.
        # it would be better to use a different chunk for each
        # vaidation run, but this is easy and probably good enough.
        map_batch_size=config.train.tokenizer_batch_size,
        preserve_non_numerical_keys=config.train.log_bits_per_byte,
    )
    if config.train.log_bits_per_byte and config.train.compile:
        # remove non-tensor columns from dataset:
        # we we do not do this then torch.compile will recompile the training
        # step every iteration because it does not know that the training step
        # ignores these string values.
        train_dataset = train_dataset.remove_columns(['chunked_meta', 'chunked_text'])

    valid_dataset = load_next_token_prediction(
        config.dataset.path,
        config.dataset.name,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
        split="validation",
        text_key="text",
        map_batch_size=config.train.tokenizer_batch_size,
        # this option usually makes data loading slow...
        # probably could be fixed.
        preserve_non_numerical_keys=True,
    )
    if config.train.compile:
        valid_dataset = valid_dataset.remove_columns(['chunked_meta', 'chunked_text'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.per_device_batch_size,
        num_workers=config.train.dataloader_workers,
        prefetch_factor=config.train.prefetch_factor)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.train.per_device_batch_size)

    return train_loader, valid_loader

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
    train_loader, valid_loader = get_dataloaders(config, tokenizer)

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
        train_time_interval=datetime.timedelta(minutes=config.checkpoint.frequency_mins),
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
        callbacks.append(Timer({'hours': config.train.max_time_hours}))

    if config.train.compile:
        # max-autotune mode will take longer to compile, but maybe it will
        # be faster overall?
        model = torch.compile(model, mode='max-autotune')

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
    # this should take care of scaling to multiple gpus if available as well.
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        )


if __name__ == "__main__":
    train()
