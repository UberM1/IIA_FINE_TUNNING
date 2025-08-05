import os
import torch
import gc
import json
from datetime import datetime
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
    EarlyStoppingCallback
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

TRAINING_CONFIG = {
    "base_model": "meta-llama/Llama-2-7b-chat-hf",
    "new_model": "llama-2-7b-chat-hf-sandia",
    "merged_model": "sandia_merged",
    "train_file": "train_dataset.json",
    "validation_file": "validation_dataset.json",
    
    # Configuración de cuantización
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "compute_dtype": "float16",
    "bnb_4bit_use_double_quant": False,
    
    # Parámetros LoRA
    "lora_alpha": 8,
    "lora_dropout": 0.15,
    "lora_r": 8,
    "lora_bias": "none",
    "lora_target_modules": ["q_proj", "v_proj"],
    
    # Parámetros de entrenamiento
    "output_dir": "./results",
    "num_train_epochs": 35,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "save_steps": 50,
    "eval_steps": 50,
    "logging_steps": 5,
    "learning_rate": 4e-5,
    "weight_decay": 0.003,
    "fp16": True,
    "bf16": False,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.1,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "report_to": "tensorboard",
    
    # Early Stopping
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "early_stopping_patience": 5,
    
    # SFT
    "max_length": 1024,
    "packing": False,
    "dataset_text_field": "text",
    
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
            f.write(f"📈 Early stopping patience: {config['early_stopping_patience']}\n")
                
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

def format_text(example):
    return {
        "text": f"<s>[INST] {example['question']} [/INST] {example['answer']} </s>"
    }

def load_datasets(config):
    """Carga los datasets de entrenamiento y validación"""
    
    print("📊 Cargando datasets...")
    
    # dataset de entrenamiento
    with open(config["train_file"], "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_dataset = Dataset.from_list(train_data["data"])
    train_dataset = train_dataset.map(format_text)
    
    # dataset de validación
    with open(config["validation_file"], "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_dataset = Dataset.from_list(val_data["data"])
    val_dataset = val_dataset.map(format_text)
    
    print(f"📊 Dataset de entrenamiento: {len(train_dataset)} ejemplos")
    print(f"📊 Dataset de validación: {len(val_dataset)} ejemplos")
    
    return train_dataset, val_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()
gc.collect()

print("🚀 Iniciando entrenamiento optimizado para RTX 3080...")

config = TRAINING_CONFIG

write_training_config(config)

train_dataset, val_dataset = load_datasets(config)

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
    per_device_eval_batch_size=config["per_device_eval_batch_size"],        
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    gradient_checkpointing=config["gradient_checkpointing"],
    optim=config["optim"],
    save_steps=config["save_steps"],
    eval_steps=config["eval_steps"],
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
    
    # Early Stopping config
    eval_strategy=config["evaluation_strategy"],
    save_strategy=config["save_strategy"],
    load_best_model_at_end=config["load_best_model_at_end"],
    metric_for_best_model=config["metric_for_best_model"],
    greater_is_better=config["greater_is_better"],
    
    # SFT parameters
    dataset_text_field=config["dataset_text_field"],
    max_length=config["max_length"],
    packing=config["packing"],
    
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

print("🔧 Configurando trainer con validación y early stopping...")

# FIXED: Simplified SFTTrainer initialization
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_params,
    args=training_params,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
)

print("🎯 Iniciando entrenamiento con validación...")
print("📈 Monitorea el progreso en: tensorboard --logdir=./results/runs")
print("🔍 El entrenamiento se detendrá automáticamente si no mejora durante 3 evaluaciones")

trainer.train()

print("💾 Guardando modelo entrenado...")
trainer.model.save_pretrained(config["new_model"])
trainer.tokenizer.save_pretrained(config["new_model"])

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
        
        print(f"✅ Modelo fusionado guardado en: {config['merged_model']}")
        
    except Exception as e:
        print(f"❌ Error en fusión: {e}")
        print("💡 El modelo LoRA se guardó correctamente y puede usarse sin fusionar")

print("\n🎉 ¡Proceso completado exitosamente!")