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

        # self.args = args
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_id


        self.data, self.origin_data = self.load_data_json(data_path)
                    
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print(f"Num PPO instances: {len(self.data)}")
            
    def __len__(self):
        return len(self.data)

    def load_data_json(self, data_path):
        with open(data_path) as f:
            datas = json.load(f)
        data_origin = datas
        data = []
        print("Loading Data")
        for d in data_origin:
            items = d["items"]
            for item in items:
                from_ = item["from"]
                value_ = item["value"]
                sentence = from_ + ":" + value_
                data.append(sentence)
        print("Load End")
        return data, data_origin



    def __getitem__(self, index: int):
        data = self.data[index]
        tokenize = self.tokenizer(text=data, padding='max_length', max_length = 37286, truncation=True)
        input_ids = tokenize['input_ids']
        attention_mask = tokenize['attention_mask']
        idx = input_ids.index(1) if (1 in input_ids) else -1
        input_ids[idx] = 2
        idx = attention_mask.index(0) if (0 in attention_mask) else -1
        attention_mask[idx] = 1
        # output_ids = data["output_ids"]
        # data = data["prompt_ids"]
        
        # prompt_length = self.max_prompt_length

        # prompt = data[:prompt_length]
        # rest = data[prompt_length:]  
        # if self.args.json_data:
        #     if output_ids is not None:
        #         rest = output_ids  
        # input_ids = self.tokenizer(data)
        # index
        dic = {
            "index": index,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        return dic
    
    def collate(self, samples):
        bs = len(samples)
        
        max_prompt_length = self.max_prompt_length
        max_rest_length = max([len(samp[2]) for samp in samples])
        
        model_batch = {
            "input_ids": torch.ones(bs, self.max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        
        # no_model_batch = {
        #     "idx": torch.zeros(bs, dtype=torch.long),
        #     "rest_ids": torch.ones(bs, max_rest_length, dtype=torch.long) * self.pad_id
        # }
        
        for i, dic in enumerate(samples):
            idx = dic["index"]
            input_id = dic["input_ids"]
            atten_mask = dic["attention_mask"]
            # left padding
            model_batch["input_ids"][i] = torch.tensor(input_id, dtype=torch.long)
            model_batch["attention_mask"][i] = torch.tensor(atten_mask, dtype=torch.long)
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            # no_model_batch["idx"][i] = idx
            # no_model_batch["rest_ids"][i][:len(rest)] = torch.tensor(rest, dtype=torch.long)
        
        return model_batch
