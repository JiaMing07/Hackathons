import os
import pickle
import requests
import numpy as np
import json
import jieba

if os.path.exists(os.path.join(os.path.dirname(__file__), 'all_lyrics.txt')):
    with open(os.path.join(os.path.dirname(__file__), 'all_lyrics.txt'), 'r', encoding='utf-8') as f:
        data = f.read()
else:
    lyrics = []
    for i in range(1, 2):
        input_file_path = os.path.join(os.path.dirname(__file__), f'lyrics{i}.json')
        with open(input_file_path, 'r') as f:
            datas = json.load(f)
        # print(f"length of dataset in characters: {len(data):,}, {'\n'.join(data[0]['lyric'])}")
        for data in datas:
            lyrics.append('\n'.join(data['lyric']))
    lyric = '\n\n'.join(lyrics)
    with open(os.path.join(os.path.dirname(__file__), 'all_lyrics.txt'), 'w', encoding='utf-8') as f:
        f.write(lyric)
    with open(os.path.join(os.path.dirname(__file__), 'all_lyrics.txt'), 'r', encoding='utf-8') as f:
        data = f.read()

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
print('type:', type(data))
print('train data:', len(train_data), train_data[:100])
print('val data:', len(val_data), val_data[:100])

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)