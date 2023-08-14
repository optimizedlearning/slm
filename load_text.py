from datasets import load_dataset
import datasets
import numpy as np

from einops import rearrange
from functools import reduce


def tokenize_for_next_token_prediction(
    examples,
    text_key,
    tokenizer,
    max_length,
    preserve_text=True,
    preserve_other_keys=True,
):
    text = examples[text_key]
    tokens = tokenizer(
        text, padding=True, pad_to_multiple_of=max_length + 1, return_tensors="np"
    )

    num_chunks = tokens["input_ids"].shape[1] // (max_length + 1)
    chunks = rearrange(tokens["input_ids"], "B (N C) -> (B N) C", C=max_length + 1)
    attention_chunks = rearrange(
        tokens["attention_mask"], "B (N C) -> (B N) C", C=max_length + 1
    )

    result = {
        "targets": chunks[:, 1:],
        "input_ids": chunks[:, :-1],
        "attention_mask": attention_chunks[:, :-1],
    }
    if preserve_text:
        # this is super-hacky... ideally we would already know these strings
        # after encoding.
        pad_index = tokenizer(tokenizer.pad_token)["input_ids"][0]

        padding = np.count_nonzero(chunks == pad_index, axis=-1)
        
        
        recovered_text = [
            tokenizer.decode(chunk[:max_length+1-pad]) for chunk, pad in zip(chunks, padding)
        ]
        result["chunked_" + text_key] = recovered_text
        result["chunked_bytes"] = [len(r_t.encode('utf-8')) for r_t in recovered_text]

    if preserve_other_keys:
        for key in examples:
            if key != text_key:
                chunked_key = "chunked_" + key
                result[chunked_key] = []
                for value in examples[key]:
                    for i in range(num_chunks):
                        result[chunked_key].append(value)

    return result


def load_next_token_prediction(
    path: str,
    name: str,
    tokenizer,
    max_length: int,
    text_key: str = "text",
    split: str = "train",
    max_text_length: int = 2**31,
    min_text_length: int = 0,
    preserve_non_numerical_keys: bool = True,
    map_batch_size: int = 1000,
) -> datasets.iterable_dataset.IterableDataset:
    """
    loads a textual dataset in streaming mode from
    huggingface and tokenizes the input for next token prediction
    (aka language modeling) task.
    """
    dataset = load_dataset(path, name, split=split, streaming=True)

    columns = dataset.column_names

    dataset = dataset.filter(
        lambda example: len(example[text_key]) >= min_text_length
        and len(example[text_key]) <= max_text_length
    )

    dataset = dataset.map(
        lambda examples: tokenize_for_next_token_prediction(
            examples,
            text_key,
            tokenizer,
            max_length,
            preserve_text=preserve_non_numerical_keys,
            preserve_other_keys=preserve_non_numerical_keys,
        ),
        batched=True,
        batch_size=map_batch_size,
        remove_columns=columns,
    )

    dataset = dataset.filter(lambda example: np.any(example["attention_mask"]))

    dataset = dataset.shuffle(seed=22, buffer_size=100)

    return dataset
