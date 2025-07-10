import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig


#pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
#messages = [
#    {"role": "user", "content": "Hello, how are you?"},
#]
#output = pipe(messages)

base_model = "meta-llama/Llama-2-7b-chat-hf"

# Se cargan datos para entrenar
dataset = load_dataset("text", data_files="data.txt", split="train")
#print(dataset[0])

new_model = "llama-2-7b-chat-hf-sandia"

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

# Parametros de PEFT
# PEFT = Parameter-Efficient Fine-Tuning
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.2,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Parametros para entrenamiento
training_params = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=1e-5,
    weight_decay=0.002,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field='text',
    max_seq_length=None,
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    args=training_params,
    processing_class=tokenizer
)

trainer.train()

# Guardamos el modelo entrenado
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)

# Esto todacia no anda
# print("Fusionando modelo base con adapter LoRA...")
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     quantization_config=quant_config,
#     device_map="auto"
# )
# model = PeftModel.from_pretrained(model, new_model)

# # Fusionamos y descargamos el modelo completo
# model = model.merge_and_unload()
# model.save_pretrained("sandia_merged")
# tokenizer.save_pretrained("sandia_merged")

# print("Modelo fusionado guardado en: sandia_merged")
