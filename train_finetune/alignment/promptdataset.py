import random
import torch
import os
from torch.utils.data import Dataset

from torch.distributed import get_rank, get_world_size
from tqdm import tqdm
import json
import numpy as np


class PromptDataset(Dataset):
    def __init__(self, tokenizer, data_path=None, num=-1):
        super().__init__()
        # self.tokenizer = tokenizer

        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.max_length = 2048

        self.data, self.origin_data = self.load_data_json(data_path)
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print(f"Num PPO instances: {len(self.data)}")
            
    def __len__(self):
        return len(self.data)

    def load_data_json(self, data_path):
        with open(data_path) as f:
            data_origin = json.load(f)
        data = [self.process_raw_sample(raw_sample) for raw_sample in data_origin]
        return data, data_origin

    def process_raw_sample(self, raw_sample):
        text = "".join([conv["from"] + ':' + conv["value"] + '\n' for conv in raw_sample["items"]])
        return text

    def __getitem__(self, index: int):
        data = self.data[index]
        tokenize = self.tokenizer(text=data, padding='max_length', max_length=self.max_length, truncation=True)

        input_ids = tokenize['input_ids']
        attention_mask = tokenize['attention_mask']
        idx = input_ids.index(self.pad_id) if (self.pad_id in input_ids) else -1
        input_ids[idx] = self.eos_id
        idx = attention_mask.index(0) if (0 in attention_mask) else -1
        attention_mask[idx] = 1

        input_ids = tokenize['input_ids'][:self.max_length]
        attention_mask = tokenize['attention_mask'][:self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def collate(self, samples):
        bs = len(samples)
        # print("ckpt 1")
        model_batch = {
            "input_ids": torch.ones(bs, self.max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length, dtype=torch.long)
        }
        # print("ckpt 2")
        for i, sample in enumerate(samples):
            # left padding
            model_batch["input_ids"][i] = torch.tensor(sample["input_ids"], dtype=torch.long)
            model_batch["attention_mask"][i] = torch.tensor(sample["attention_mask"], dtype=torch.long)
        
        return model_batch
