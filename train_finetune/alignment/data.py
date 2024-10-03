from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import tqdm
from torch.distributed import get_rank

model_id = "facebook/opt-350m"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
data_path = "./sharegpt_gpt4.json"
# data_path = "/data/shenjiaming/alignment/sample.json"
with open(data_path) as f:
    datas = json.load(f)
items = []
for d in datas:
    items.extend(d["items"])
max_length = 0
sentences = []
cnt = 0
for item in items:
    sen = item["from"] + " : " + item["value"]
    sentences.append(sen)
    length = len(sen)
    if length > 2048:
        cnt+=1
    # if "what" in sen:
    #     print(sen)
print(cnt)
print(len(sentences))
for i in range(300,400):
    print(items[i])
output_path = "./shared_gpt4.json"
with open(output_path, "w") as f:
    json.dump(items, f)
# data = load_dataset("json", data_files="/data/shenjiaming/alignment/quotes.jsonl")
# data = load_dataset("json", data_files="/data/shenjiaming/alignment/shared_gpt4.json")
# print(data)
# data = data.map(lambda samples: tokenizer(samples["value"]), batched=True)
# print(data)

data_path = "./sharegpt_gpt4.json"
with open(data_path) as f:
    datas = json.load(f)
# print(type(datas))
# padding = 500
data = sentences[0]
# print(data)
# print(type(data))
# # data = list(data)
tokenize = tokenizer(text=data+tokenizer.eos_token, padding='max_length', max_length = 200, truncation=True)
# # input_ids = np.array(data)
print(tokenizer.convert_ids_to_tokens(tokenize['input_ids']))
print(tokenize)
print(tokenizer.bos_token)
# input_ids = tokenize['input_ids']
# attention_mask = tokenize['attention_mask']
# idx = input_ids.index(1) if (1 in input_ids) else -1
# input_ids[idx] = 2
# print(input_ids)
# idx = attention_mask.index(0) if (0 in attention_mask) else -1
# attention_mask[idx] = 1
# print(attention_mask)
# # print(tokenize['input_ids'], tokenize['attention_mask'])
# # attention_mask = np.ones(len(data))
# # input_ids = np.pad(input_ids, ((0, padding)), "constant", constant_values=0)
# # attention_mask = np.pad(attention_mask, ((0, padding)), "constant", constant_values=0)
# def load_data_json(data_path):
#     with open(data_path) as f:
#         datas = json.load(f)
#     data_origin = datas
#     data = []
#     print("Loading Data")
#     for d in data_origin:
#         items = d["items"]
#         for item in items:
#             from_ = item["from"]
#             value_ = item["value"]
#             sentence = from_ + ":" + value_
#             data.append(sentence)
#     print("Load End")
#     return data, data_origin
# print(tokenizer.convert_ids_to_tokens(2))
# # data, data_origin = load_data_json(data_path)
# # print(len(data))
# # print(len(data_origin))
# # print(data[0])