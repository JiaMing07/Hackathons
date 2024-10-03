import os
import pickle
import json
import numpy as np
import jieba



val_data = np.fromfile(os.path.join(os.path.dirname(__file__), 'train.bin'), dtype=np.uint32)
with open(os.path.join(os.path.dirname(__file__), 'words.json'), 'r', encoding='utf-8') as f:
    stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
print(val_data[:200])
print(decode(val_data[:200]))
