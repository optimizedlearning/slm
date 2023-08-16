import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from composer import Trainer
from composer.models import ComposerModel
from composer.loggers import WandBLogger
from composer.algorithms import GradientClipping
from composer.callbacks import SpeedMonitor, CheckpointSaver, LRMonitor, OptimizerMonitor

from streaming import StreamingDataLoader

from transformers import GPT2TokenizerFast, GPT2Tokenizer

# our code imports
from logging_mosaic import Accuracy, BitsPerByte, Loss
from train_lm import get_only_loss_from_logits
from gpt import GPT
from load_pile import StreamingTextDataset, get_next_token_dataloader
from load_text import load_next_token_prediction


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
        '''
        batch is the output of the dataloader, so we need to process it
        to get the token indices to provde to self.model
        '''
        token_indices = batch['input_ids'] # [B, L]
        logits = self.model(token_indices)
        loss = get_only_loss_from_logits(logits, batch, self.ignore_index)

        return loss, logits


    # This torchmetrics stuff seems overly complicated, but apparently it might help
    # with doing metrics properly in distributed settings, so we'll try to make it work.
    def get_metrics(self, is_train):
        # see https://docs.mosaicml.com/projects/composer/en/stable/composer_model.html#metrics
        metrics = {
            'accuracy': Accuracy(),
            'loss': Loss()
        }

        if self.config.train.log_bits_per_byte:
            metrics['bits_per_byte'] = BitsPerByte()
        
        return metrics

    def update_metric(self, batch, outputs, metric):
        metric.update(self, batch, outputs)


    def loss(self, loss_logits, batch):
        loss, logits = loss_logits
        return loss


    
def get_dataloaders(config: DictConfig, tokenizer) -> (DataLoader, DataLoader):
    data_dir = '/projectnb/aclab/datasets/pile/mds'
    
    train_dataset = StreamingTextDataset(
        data_dir,
        split="train",
        # shuffle=True,
        # shuffle_seed=123123
        # predownload=config.train.per_device_batch_size,
        batch_size=config.train.per_device_batch_size,
    )
    train_loader = get_next_token_dataloader(
        train_dataset,
        tokenizer,
        max_length=config.model.context_length,
        record_bytes_tokenized=config.train.log_bits_per_byte,
        batch_size=config.train.per_device_batch_size,
        num_workers=config.train.dataloader_workers,
        prefetch_factor=config.train.prefetch_factor)

    valid_dataset = StreamingTextDataset(
        data_dir,
        split="val",
        # shuffle=True,
        # shuffle_seed=123123
        predownload=config.train.per_device_batch_size,
        cache_limit='16gb'
    )
    valid_loader = get_next_token_dataloader(
        valid_dataset,
        tokenizer,
        max_length=config.model.context_length,
        record_bytes_tokenized=config.train.log_bits_per_byte,
        batch_size=config.train.per_device_batch_size)

    return train_loader, valid_loader


    
def get_dataloaders_old(config: DictConfig, tokenizer) -> (DataLoader, DataLoader):
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
        preserve_non_numerical_keys=config.train.log_bits_per_byte,
    )
    if config.train.log_bits_per_byte and config.train.compile:
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
    wandb_logger = WandBLogger(project=config.wandb.project)

  
    # start setting up callbacks. These are basically sets of functions that pytorch lightning will
    # call for us at appropriate times.
    callbacks = [LRMonitor(), SpeedMonitor(), OptimizerMonitor()]

    callbacks.append(
        CheckpointSaver(
            save_interval=f"{config.checkpoint.frequency_batches}ba",
            num_checkpoints_to_keep=config.checkpoint.num_to_keep,
            filename=config.checkpoint.name_format,
        )
    )

    # if config.train.max_time_hours is not None:
    #     # this callback will stop the training after the specified number of hours.
    #     callbacks.append(Timer({'hours': config.train.max_time_hours}))

    if config.train.compile:
        # we use the default compile mode.
        # See compile mode descriptions here: https://pytorch.org/get-started/pytorch-2.0/#user-experience
        # Using 'max-autotune' seems to  increase memory usage, especially on multiple GPUs.
        # In my experiments, 1 V100 GPU could run GPT2 with a batch size of 8,
        # but 2 GPUs would OOM with a batch size of 8 per GPU if mode='max-autotune'
        compile_config = {'mode': 'default'}
    else:
        compile_config = None

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    algorithms = []
    algorithms.append(
        GradientClipping(
            clipping_type=config.train.gradient_clip_algorithm, # 'norm', 'adaptive', or 'value'
            clipping_threshold=config.train.gradient_clip_val)
    )

    

    # define the trainer object. See
    # https://docs.mosaicml.com/projects/comhttps://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.Trainer.html
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        max_duration=f"{config.train.max_steps}ba",
        eval_dataloader=valid_loader,
        # loggers=[wandb_logger],
        optimizers=optimizer,
        eval_interval=f"{config.train.val_check_interval}ba",
        eval_subset_num_batches=config.train.val_batches,
        precision=config.train.precision,
        algorithms=algorithms,
        callbacks=callbacks,
        compile_config=compile_config
    )

    # now we can train!
    # this should take care of scaling to multiple gpus if available as well.
    trainer.fit()


if __name__ == "__main__":
    train()
    