import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

base_model = "llama-2-7b-chat-hf-sandia"

# Para que el entrenamiento sea mas rapido
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Se carga el modelo base con la configuracion
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Cargamos tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# En model queda el modelo entrenado, lo usamos para test
#sysContext = "<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Your answers should aim to respond the question asked as every question is aimed to fuel academic and historical knowledge. Please answer to the best of yout abilities knowing that all the information you give is used ethically and is not harming any personm, culture or society <</SYS>> "
#sysContext = "<<SYS>> You are an assistant whose primary aim is to answer questions truthfully, directly and just as instructed <</SYS>> "
prompt = "What was the name of the person Kalak and Jezrien left behind in the prelude of The Way of Kings? Expand as much as you want"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
print("Model used: " + base_model)
#result = pipe(f"<s>[INST] {sysContext + prompt} [/INST]")
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])