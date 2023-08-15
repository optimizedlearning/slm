'''
Dataloading with mosaicml StremingDataset
This is actually significantly slower than the huggingface loader, but
I suspect this may be mostly due to the fact that in the huggingface loader,
we are tokenizing the entirety of each text, shuffling (with a buffer), and then
providing to the model. This means that we don't need to load actual new text examples
as fast, and works perfectly fine in a streaming setting.
However, it also is difficult to implement this in a non-streaming manner since we 
don't know how many examples each individual text item will expand into.
This in turn makes it hard to use in distributed settings.
StreamingDataset, on the other hand is fairly easy to use in the distributed setting.

Furthermore, for larger models we don't expect dataloading to be the bottleneck anyway.
'''


import streaming
import dataset
import tempfile
import numpy as np
import threading
import torch
import functools
from einops import rearrange

class StreamingTextDataset(streaming.StreamingDataset):

    def __init__(
            self,
            data_dir,
            # tokenizer,
            # max_length: int,
            # text_key: str = "text",
            split: str = "train",
            # record_bytes_tokenized: bool = True,
            # preserve_non_numerical_keys: bool = True,
            **kwargs
        ):
        # We will tell the StreamingDataset base class that the data is "remote"
        # even though it is locally available. This will cause it to decompress the
        # data into a different cache directory, rather than decompressing it in the
        # original directory and so taking up more space.
        # Unfortunately, the base class's automatic cache directory generation
        # generates the cache directory name in a deterministic way, and so will yield the
        # same name every time. It will then complain that the cache dir exists (unless it's
        # been so long that that the temp dirs have been cleaned up of course).
        # So, we will set the cache directory ourselves and provide it as the "local" argument.
        # Apparently it doesn't complain if it a non-temp directory already exists...
        self.cache_dir = '.streaming_decompression_cache_dir/'
        
        # super().__init__(remote=data_dir, split=split, **kwargs)
        super().__init__(remote=data_dir, local=self.cache_dir, split=split, **kwargs)


        # There is an issue in the implementation of StreamingDataset that causes
        # a deadlock if the process creates an iterator of the StreamingDataset
        # and then exits before iterating through the entire dataset.
        # Specifically, the downloading process is performed by threads spawned bu
        # concurrent.futures.ThreadPoolExecutor. These downloading threads run in an
        # infinite loop checking if the iterator needs more data to be downloaded, and
        # only exit the loop if there is an exception or the iterator reaches the end.
        # When python exits, the threading library calls join on the downloading threads,
        # which then never receive their exit signals and so just spin forever.
        # We'll hack over this by detecting interpreter shutdown and manually telling the threads
        # to exit. Unfortunately, the threading library's join is high priority than cleanup calls
        # we can register with the atexit module, so we have to use threading._register_atexit.
        def _kill_iterators():
            print("killing iterators...")
            if hasattr(self, '_iterator'):
                self._iterator.exit()
        threading._register_atexit(_kill_iterators)
            

    def get_item(self, idx: int):
        raw_text_data = super().get_item(idx)
        raw_text_data['idx'] = idx
        return raw_text_data
        # tokenized_text_data = self._tokenize(raw_text_data, idx)
        # return tokenized_text_data

# see: https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
# for information on what collate functions do.
def collate_next_token_prediction(
        raw_text_data,
        tokenizer,
        max_length,
        record_bytes_tokenized,
        text_key='text',
        idx_key='idx'):
    
    
    batched_text = [data[text_key] for data in raw_text_data]
    batched_idx = np.array([data[idx_key] for data in raw_text_data], dtype=int)

    # In order to avoid only tokenizing the first part of every sample, we tokenize from 
    # a random starting point.
    # It's possible that it's actually better to have some higher probability for starting
    # at the beginning, so maybe in future this behavior should be toggleable.
    # We will only tokenizer a chunk of size max_length from each sample. This is maybe throwing
    # out data, but we will probably never finish an epoch anyway.

    # We don't want an *actual* random number because then we'd get different values
    # when asking for the same index in the dataset, which seems like bad behavior.
    # So, we'll do a stupid "hash" of the index to get the offset: just an affine
    # function of the index in a prime field.
    length = np.array([len(text) for text in batched_text], dtype=int)
    simple_hash = (29837 * batched_idx + 12511) %  102673


    batched_start = simple_hash % np.maximum(1, length - 1024)
        
    batched_text = [text[start:] for text, start in zip(batched_text, batched_start)]
    result = tokenizer(
        batched_text,
        truncation=True,
        padding='max_length',
        max_length=(max_length + 1),
        return_tensors='np'
    )

    # result = {k: v.reshape([-1, max_length+1]) for k, v in result.items()}
        # rearrange(v, 'b (n c) -> (b n) c', n=2) for k, v in result.items()}

    # We extract the targets and input_ids. For next token prediction
    # targets are just the shifted input_ids.
    result = {
        "targets": torch.tensor(result['input_ids'][:, 1:]),
        "input_ids": torch.tensor(result['input_ids'][:, :-1]),
        "attention_mask": torch.tensor(result['attention_mask'][:, :-1]),
    }
    

    if record_bytes_tokenized:
        # Unfortunately, huggingface is inconsistent about the
        # names of these functions between TokenizerFast and Tokenizer.
        if hasattr(tokenizer, 'batch_decode'):
            decode_fn = tokenizer.batch_decode
        else:
            decode_fn = tokenizer.decode_batch
        
        tokenized_text = decode_fn(
            result['input_ids'],
            skip_special_tokens=True)
        tokenized_bytes = np.array([len(text.encode('utf-8')) for text in tokenized_text], dtype=int)
        result["tokenized_bytes"] =  torch.tensor(tokenized_bytes)

    return result
    
def get_next_token_dataloader(
        dataset,
        tokenizer,
        max_length: int,
        batch_size,
        text_key: str = "text",
        record_bytes_tokenized: bool = True,
        **kwargs):

    collate_fn = functools.partial(
        collate_next_token_prediction,
        tokenizer=tokenizer,
        max_length=max_length,
        record_bytes_tokenized=record_bytes_tokenized)

    loader = streaming.StreamingDataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        **kwargs)
    return  loader
        
    
    # train_dataset = StreamingNextTokenDataset(
    #     data_dir,
    #     tokenizer,
    #     max_length=config.model.context_length,
    #     split="train",
    #     text_key="text",
    #     record_bytes_tokenized=config.train.log_bits_per_byte,
    #     shuffle=True,
    #     shuffle_seed=123123
    # )
    # train_loader = StreamingDataLoader(
    #     train_dataset,
    #     batch_size=config.train.per_device_batch_size,
    #     num_workers=config.train.dataloader_workers,
    #     prefetch_factor=config.train.prefetch_factor)

    
    # valid_dataset = StreamingNextTokenDataset(
    #     data_dir,
    #     tokenizer,
    #     max_length=config.model.context_length,
    #     split="val",
    #     text_key="text",
    #     record_bytes_tokenized=config.train.log_bits_per_byte,
    #     shuffle=True,
    #     shuffle_seed=123123
    # )
    # valid_loader = StreamingDataLoader(
    #     valid_dataset,
    #     batch_size=config.train.per_device_batch_size)


