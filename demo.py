from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configura el modelo (asegúrate de que 'meta-llama/Llama-2-7b-chat-hf' esté disponible para ti)
model_name = "meta-llama/Llama-2-7b-chat-hf"  # o la ruta local si lo descargaste
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Usar mitad de precisión para ahorrar memoria
    device_map="auto",          # Carga automáticamente en GPU si está disponible
)

# Función para generar respuestas
def ask_llama(prompt, max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,  # Controla la creatividad (menor = más determinista)
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Demo interactiva
print("¡Demo de LLaMA 2.7B! Escribe 'salir' para terminar.")
while True:
    user_input = input("\nTú: ")
    if user_input.lower() == "salir":
        break
    prompt = f"Pregunta: {user_input}\nRespuesta:"
    response = ask_llama(prompt)
    print("\nLLaMA:", response.split("Respuesta:")[1].strip())  # Extrae solo la respuesta