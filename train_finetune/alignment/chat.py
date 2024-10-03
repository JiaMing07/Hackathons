from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./opt-350m")
# model = AutoModelForCausalLM.from_pretrained("/workspace/xixuan/model/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("./opt-350m")
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=2048, use_cache=True, device="cuda:0", eos_token_id=2)

context = ""

while True:
    human_input = input("human: ")
    context += "human:" + human_input + "\n" + "gpt:"
    machine_output = generator(human_input)[0]['generated_text']
    print("machine: ", machine_output)
    context += machine_output + "\n"

# write a + b problem in python.
