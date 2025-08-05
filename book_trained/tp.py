import os
import torch
import gc
import json
from datetime import datetime
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

TRAINING_CONFIG = {
    # Modelos y datos
    "base_model": "meta-llama/Llama-2-7b-chat-hf",
    "new_model": "llama-2-7b-chat-hf-sandia",
    "merged_model": "sandia_merged",
    "data_file": "data.txt",
    
    # Configuración de cuantización
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "compute_dtype": "float16",
    "bnb_4bit_use_double_quant": False,
    
    # Parámetros LoRA
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "lora_r": 8,
    "lora_bias": "none",
    "lora_target_modules": ["q_proj", "v_proj"],
    
    # Parámetros de entrenamiento
    "output_dir": "./results",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "save_steps": 50,
    "logging_steps": 5,
    "learning_rate": 3e-05,
    "weight_decay": 0.003,
    "fp16": True,
    "bf16": False,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.1,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "report_to": "tensorboard",
    "dataset_text_field": "text",
    "max_seq_length": None,
    "packing": False,
    "dataloader_num_workers": 0,
    "remove_unused_columns": False,
    
    # Configuración adicional
    "enable_merge": True,
    "print_settings": True,
    "config_file": "training_config.json"
}

def write_training_config(config):
    if not config["print_settings"]:
        return
    
    config_with_metadata = config.copy()
    config_with_metadata["training_timestamp"] = datetime.now().isoformat()

    try:
        with open('preguntas.txt', 'a', encoding='utf-8') as f:
            f.write("\nCONFIGURACIÓN:\n")
            f.write(f"🏷️ Modelo base: {config['base_model']}\n")
            f.write(f"  load_in_4bit: {config['load_in_4bit']}\n")
            f.write(f"  bnb_4bit_quant_type: {config['bnb_4bit_quant_type']}\n")
            f.write(f"  compute_dtype: {config['compute_dtype']}\n")
            f.write(f"  bnb_4bit_use_double_quant: {config['bnb_4bit_use_double_quant']}\n")
            f.write(f"🏷️ weight_decay: {config['weight_decay']}\n")
            f.write(f"🎓 Learning rate: {config['learning_rate']}\n")
            f.write(f"🔄 Épocas: {config['num_train_epochs']}\n")
            f.write(f"📊 Batch size: {config['per_device_train_batch_size']}\n")
            f.write(f"🔧 LoRA r: {config['lora_r']}\n")
            f.write(f"🔧 LoRA alpha: {config['lora_alpha']}\n")
            f.write(f"🔧 LoRA dropout: {config['lora_dropout']}\n")
            f.write(f"🔧 LoRA target_modules: {config['lora_target_modules']}\n")
                
    except Exception as e:
        print(f"⚠️ Error guardando configuración: {e}")

def load_training_config(config_file):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return TRAINING_CONFIG
    except Exception as e:
        return TRAINING_CONFIG

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Limpiar memoria GPU
torch.cuda.empty_cache()
gc.collect()

print("🚀 Iniciando entrenamiento optimizado para RTX 3080...")

config = TRAINING_CONFIG

write_training_config(config)

dataset = load_dataset("text", data_files=config["data_file"], split="train")
print(f"📊 Dataset cargado: {len(dataset)} ejemplos")

compute_dtype = getattr(torch, config["compute_dtype"])

quant_config = BitsAndBytesConfig(
    load_in_4bit=config["load_in_4bit"],
    bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
)

model = AutoModelForCausalLM.from_pretrained(
    config["base_model"],
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Cargamos tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    r=config["lora_r"],                 
    bias=config["lora_bias"],
    task_type="CAUSAL_LM",
    target_modules=config["lora_target_modules"]
)

training_params = SFTConfig(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],                    
    per_device_train_batch_size=config["per_device_train_batch_size"],         
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    gradient_checkpointing=config["gradient_checkpointing"],
    optim=config["optim"],
    save_steps=config["save_steps"],
    logging_steps=config["logging_steps"],               
    learning_rate=config["learning_rate"],            
    weight_decay=config["weight_decay"],            
    fp16=config["fp16"],
    bf16=config["bf16"],
    max_grad_norm=config["max_grad_norm"],
    max_steps=config["max_steps"],
    warmup_ratio=config["warmup_ratio"],
    group_by_length=config["group_by_length"],
    lr_scheduler_type=config["lr_scheduler_type"],
    report_to=config["report_to"],
    dataset_text_field=config["dataset_text_field"],
    max_seq_length=config["max_seq_length"],           
    packing=config["packing"],                 
    dataloader_num_workers=config["dataloader_num_workers"],      
    remove_unused_columns=config["remove_unused_columns"],   
)

print("🔧 Configurando trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    args=training_params,
    processing_class=tokenizer
)

print("🎯 Iniciando entrenamiento...")
print("📈 Monitorea el progreso en: tensorboard --logdir=./results/runs")

trainer.train()

print("💾 Guardando modelo entrenado...")
trainer.model.save_pretrained(config["new_model"])
trainer.processing_class.save_pretrained(config["new_model"])

del trainer
del model
torch.cuda.empty_cache()
gc.collect()

print(f"✅ Entrenamiento completo! Modelo guardado en: {config['new_model']}")

if config["enable_merge"]:
    print("\n🔄 Fusionando modelo base con adapter LoRA...")
    try:
        base_model_for_merge = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=quant_config,
            device_map="auto"
        )
        
        model_with_lora = PeftModel.from_pretrained(base_model_for_merge, config["new_model"])
        
        merged_model = model_with_lora.merge_and_unload()
        merged_model.save_pretrained(config["merged_model"])
        tokenizer.save_pretrained(config["merged_model"])
        
        print(f"Modelo fusionado guardado en: {config['merged_model']}")
        
    except Exception as e:
        print(f"Error en fusión: {e}")
