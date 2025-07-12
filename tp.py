import os
import torch
import gc
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

# ===== CONFIGURACIÓN CRÍTICA PARA RTX 3080 =====
# Configurar variables de entorno ANTES de cargar el modelo
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Limpiar memoria GPU
torch.cuda.empty_cache()
gc.collect()

print("🚀 Iniciando entrenamiento optimizado para RTX 3080...")

base_model = "meta-llama/Llama-2-7b-chat-hf"

# Se cargan datos para entrenar
dataset = load_dataset("text", data_files="data.txt", split="train")
print(f"📊 Dataset cargado: {len(dataset)} ejemplos")

new_model = "llama-2-7b-chat-hf-sandia"

# Para que el entrenamiento sea mas rapido
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

print("⚡ Cargando modelo base con cuantización...")
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

peft_params = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,   
    r=8,                 
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]  
)

training_params = SFTConfig(
    output_dir="./results",
    num_train_epochs=5,                    
    per_device_train_batch_size=1,         
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_steps=50,
    logging_steps=5,               
    learning_rate=3e-5,            
    weight_decay=0.003,            
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    dataset_text_field='text',
    max_seq_length=None,           
    packing=False,                 
    dataloader_num_workers=0,      
    remove_unused_columns=False,   
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
# Guardamos el modelo entrenado
trainer.model.save_pretrained(new_model)
trainer.processing_class.save_pretrained(new_model)

del trainer
del model
torch.cuda.empty_cache()
gc.collect()

print(f"✅ Entrenamiento completo! Modelo guardado en: {new_model}")

print("\n🔄 Fusionando modelo base con adapter LoRA...")
try:
    base_model_for_merge = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    model_with_lora = PeftModel.from_pretrained(base_model_for_merge, new_model)
    
    # Fusionar y descargar el modelo completo
    merged_model = model_with_lora.merge_and_unload()
    merged_model.save_pretrained("sandia_merged")
    tokenizer.save_pretrained("sandia_merged")
    
    print("✅ Modelo fusionado guardado en: sandia_merged")
    
except Exception as e:
    print(f"⚠️ Error en fusión: {e}")
    print("💡 Puedes fusionar manualmente después usando el código comentado")

print("\n🎉 ¡Proceso completo terminado!")