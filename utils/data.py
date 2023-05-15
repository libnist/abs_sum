# Imports of datasets
from datasets import load_dataset
import datasets

# Typing imports
from .tokenizer import CustomTokenizer
from typing import List, Dict

# torch dataset import
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Drawing imports
import matplotlib.pyplot as plt
import numpy as np


def get_data(path: str,
             split: str,
             *args,
             version: str = "3.0.0",
             **kwargs) -> datasets.arrow_dataset.Dataset:
    dataset = load_dataset(path=path,
                           split=split,
                           *args,
                           version=version,
                           **kwargs)
    return dataset


def get_count_data(dataset: datasets.arrow_dataset.Dataset,
                   tokenizers: Dict[str, CustomTokenizer]):

    def count_tokens(example):
        for key, tokenizer in tokenizers.items():
            example[key] = len(tokenizer.encode(example[key])[0])
        return example
    return dataset.map(count_tokens)


class DocSumDataset(Dataset):

    def __init__(self,
                 src: list,
                 tgt: list,
                 src_tokenizer: CustomTokenizer,
                 tgt_tokenizer: CustomTokenizer,
                 src_max_token: int,
                 tgt_max_token: int):

        if len(src) != len(tgt):
            raise ValueError(f"Length of src is not equal to length of tgt")

        self.src = src
        self.tgt = tgt

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        self.src_max_token = src_max_token
        self.tgt_max_token = tgt_max_token

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        src = self.src_tokenizer.encode(self.src[i])[0]

        tgt = self.tgt_tokenizer.encode(self.tgt[i])[0]

        return (src[:self.src_max_token],
                tgt[:self.tgt_max_token][:-1],
                tgt[:self.tgt_max_token][1:])
        
def max_of_lens(lists):
    return max(map(lambda x: len(x), lists))

def to_max(input_ids, max_tokens, pad_id):
    if len(input_ids) > max_tokens:
        return input_ids[:max_tokens]
    else:
        difference = max_tokens - len(input_ids)
        return input_ids + ([pad_id] * difference)
        
def get_collate(src_pad_id=0, tgt_pad_id=0):
    
    def collate_fn(batch):
        src = []
        tgt_in = []
        tgt_out = []
        
        for i in range(len(batch)):
            src.append(batch[i][0])
            tgt_in.append(batch[i][1])
            tgt_out.append(batch[i][2])
        
        src_max_len = max_of_lens(src)
        tgt_in_max_len = max_of_lens(tgt_in)
        tgt_out_max_len = max_of_lens(tgt_out)
        
        src = [to_max(item, src_max_len, src_pad_id) for item in src]
        tgt_in = [to_max(item, tgt_in_max_len, tgt_pad_id) for item in tgt_in]
        tgt_out = [to_max(item, tgt_out_max_len, tgt_pad_id) for item in tgt_out]
        
        return (
            torch.LongTensor(src),
            torch.LongTensor(tgt_in),
            torch.LongTensor(tgt_out)
        )
    return collate_fn


def get_dataloader(
    src: list,
    tgt: list,
    src_tokenizer: CustomTokenizer,
    tgt_tokenizer: CustomTokenizer,
    src_max_token: int,
    tgt_max_token: int,
    batch_size: int,
    num_workers: int = os.cpu_count(),
    shuffle: bool = False
):
    
    src_pad_id = src_tokenizer.token_to_id("<pad>")
    tgt_pad_id = tgt_tokenizer.token_to_id("<pad>")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dataset = DocSumDataset(
        src=src,
        tgt=tgt,
        src_max_token=src_max_token,
        tgt_max_token=tgt_max_token,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer
    )
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=get_collate(src_pad_id, tgt_pad_id))
    return dataloader


def draw_hist(input_,
              title,
              x_lim,
              x_label,
              y_label="Frequency of Docs",
              bins=700):

    data = np.array(input_)

    mean = data.mean()
    median = np.median(data)

    plt.figure(figsize=(10, 7))
    plt.title(f"{title}\n" +
              f"Mean: {mean:.0f} | Median: {median:.0f}")
    plt.xlim(x_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.hist(data, bins=bins)
