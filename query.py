import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import BitsAndBytesConfig

# Base model setup
base_model = "llama-2-7b-chat-hf-sandia"

# Set up the compute dtype for faster training (optional, for better performance)
compute_dtype = getattr(torch, "float16")

# Load quantization config for 4-bit inference (as per your setup)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load the model with your quantization config
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Create the pipeline with your configurations
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)

print("Model used: " + base_model)

while True:
    # Take the query from the user
    prompt = input("Enter your query (or type 'exit' to stop): ")

    if prompt.lower() == "exit":
        print("Exiting...")
        break
    
    # Format the input prompt with the instructions
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"

    # Get the model's response
    result = pipe(formatted_prompt)

    # Print the result
    print("Model Response: ", result[0]['generated_text'])
