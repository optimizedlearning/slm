import hydra

import torch

from omegaconf import OmegaConf, DictConfig

from torch.utils.data import DataLoader

from datasets import load_dataset

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from load_text import load_next_token_prediction

from transformers import GPT2Tokenizer

from gpt import GPT
import train_lm
import numpy as np

from lightning.pytorch.callbacks import LearningRateMonitor

from util import LogTrainMetrics


class Model(GPT):
    def __init__(self, config, tokenizer):
        super().__init__(vocab_size=tokenizer.vocab_size, config=config.model)
        self._full_config = config
        self.train_config = config.train
        self.tokenizer = tokenizer
        self.ignore_index = tokenizer(tokenizer.pad_token)["input_ids"][0]

        self.save_hyperparameters(OmegaConf.to_container(config))



    def training_step(self, batch, batch_idx):

        
        loss_data = train_lm.get_loss(self, batch, ignore_index=self.ignore_index)


        return loss_data

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )

        def lr_schedule(step):
            scale = 1.0
            if self.train_config.lr_warmup > 0:
                scale *= min(1.0, step / self.train_config.lr_warmup)
            if self.train_config.lr_decay == "linear":
                scale *= 1.0 - step / self.train_config.max_batches_per_epoch
            if self.train_config.lr_decay == "cosine":
                scale *= 0.5 * (
                    1.0 + np.cos(np.pi * step / self.train_config.max_batches_per_epoch)
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Load GPT tokenizer.
    # We need to specify a padding token, so we will use '<|pad|>'
    # In any event, I think ignore_index loss masking will make the padding token
    # actually irrelevant.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    dataset = load_next_token_prediction(
        config.dataset.path,
        config.dataset.name,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
        split="train",
        text_key="text",
        map_batch_size=config.train.tokenizer_batch_size,
        # this option usually makes data loading slow...
        # probably could be fixed.
        preserve_non_numerical_keys=config.train.log_bits_per_byte,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.dataloader_workers,
        prefetch_factor=config.train.prefetch_factor)

    model = Model(config=config, tokenizer=tokenizer)
    wandb_logger = WandbLogger(project=config.wandb_project)
    # wandb_logger.watch(model)
    trainer = pl.Trainer(
        limit_train_batches=config.train.max_batches_per_epoch,
        max_epochs=config.train.max_epochs,
        precision=config.train.precision,
        gradient_clip_val=config.train.gradient_clip_val,
        gradient_clip_algorithm=config.train.gradient_clip_algorithm,
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(), LogTrainMetrics()],
    )
    # pl_model = Model(model=model, optimizer=optimizer, tokenizer=tokenizer)
    if config.train.compile:
        # In my tests, this actually somewhat slows things down.
        # It slows things down a LOT if you try to log bits/byte, I believe
        # because the data now contains strings that change every iteration.
        model = torch.compile(model)
    trainer.fit(model=model, train_dataloaders=loader)


if __name__ == "__main__":
    train()
