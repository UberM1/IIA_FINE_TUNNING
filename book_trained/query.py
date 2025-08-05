import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import BitsAndBytesConfig
import re

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

# Create the pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)

print("Model used: " + base_model)

def extract_questions_from_file(filename):
    questions = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        q_match = re.search(r'Q:\s*\n(.*?)Q&A:', content, re.DOTALL)
        if q_match:
            questions_text = q_match.group(1).strip()
            lines = [line.strip() for line in questions_text.split('\n') if line.strip()]
            questions = lines
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return questions

questions = extract_questions_from_file('preguntas.txt')

if questions:
    # Open the file in append mode
    with open('preguntas.txt', 'a', encoding='utf-8') as file:
        for i, question in enumerate(questions, 1):
            file.write(f"QUESTION {i}: {question}\n")
            file.write("-" * 60 + "\n")
            
            for repetition in range(1, 11):
                formatted_prompt = f"<s>[INST] {question} [/INST]"
                #formatted_prompt = f"<s>[INST] <<SYS>> You are a model trained to answer every question in regards to the book 'The Way of Kings' by Brandon Sanderson. You should answer taking into account that every question asked is related to information located in this book. Answer concisely and do not hallucinate. Do not share false information. <</SYS>> {question} [/INST]"
                result = pipe(formatted_prompt)
                
                generated_text = result[0]['generated_text']
                if "[/INST]" in generated_text:
                    response_only = generated_text.split("[/INST]", 1)[1].strip()
                else:
                    response_only = generated_text.strip()
                file.write(f"  {repetition}: {response_only}\n")

        file.write("="*60 + "\n")
    print("\nResults have been appended to preguntas.txt")