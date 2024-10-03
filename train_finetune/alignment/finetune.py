import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPT2LMHeadModel
import transformers

from promptdataset import PromptDataset

from datasets import load_dataset

model_name = "opt-350m"
model_id = "./" + model_name
save_path = "./"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# data = load_dataset("/workspace/xixuan/data/english_quotes")
# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
data = PromptDataset(tokenizer=tokenizer, data_path="./sharegpt_gpt4.json")

# needed for gpt-neo-x tokenizer
# tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        max_steps=len(data),
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_hf"
        # optim="adamw_torch"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["query_key_value"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# model = get_peft_model(model, config)

trainer.train()

model.save_pretrained(os.path.join(save_path, model_name + "-sft2"))
