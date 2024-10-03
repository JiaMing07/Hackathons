from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, LoraConfig, PeftModel
peft_model_id = "/home/v-jiamshen/hackathon/train_finetune/alignment/opt-350m-qlora"

model = AutoModelForCausalLM.from_pretrained("/home/v-jiamshen/hackathon/train_finetune/alignment/opt-350m")
model = PeftModel.from_pretrained(model, peft_model_id)


# config = LoraConfig(
#     r=8, 
#     lora_alpha=32, 
#     # target_modules=["k_proj", "q_proj", "v_proj"], # "k_proj", "q_proj", "v_proj", "query_key_value", "c_attn"
#     lora_dropout=0.05, 
#     bias="none", 
#     task_type="CAUSAL_LM"
# )

# model.add_adapter("lora")
# model = AutoModelForCausalLM.from_pretrained("/workspace/xixuan/model/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("/home/v-jiamshen/hackathon/train_finetune/alignment/opt-350m")
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=2048, use_cache=True, device="cuda:0", eos_token_id=2)

context = ""

while True:
    human_input = input("human: ")
    context += "human:" + human_input + "\n" + "gpt:"
    machine_output = generator(human_input)[0]['generated_text']
    print("machine: ", machine_output)
    context += machine_output + "\n"
