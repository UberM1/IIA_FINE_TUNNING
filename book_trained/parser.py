from datasets import Dataset
import re
MAX_TOKENS = 2048


def parse_book_to_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    paragraphs = re.split(r'\n\s*\n', content.strip()) # saca los vacios
    paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs if p.strip()]
    
    dataset = []
    current_entry = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_word_count = len(para_words)
        
        if para_word_count > MAX_TOKENS:
            words = para.split()
            chunks = [' '.join(words[i:i+MAX_TOKENS]) for i in range(0, len(words), MAX_TOKENS)]
            
            for chunk in chunks:
                dataset.append({'text': chunk})
        else:
            # Verificar si podemos añadir este párrafo a la entrada actual
            if current_word_count + para_word_count <= MAX_TOKENS:
                current_entry.append(para)
                current_word_count += para_word_count
            else:
                # Añadir la entrada actual al dataset y comenzar nueva
                if current_entry:
                    dataset.append({'text': ' '.join(current_entry)})
                current_entry = [para]
                current_word_count = para_word_count
    
    # Añadir la última entrada si queda algo
    if current_entry:
        dataset.append({'text': ' '.join(current_entry)})
    
    return dataset

file_path = 'out.txt'
dataset = parse_book_to_dataset(file_path)

print(f"Total de entradas en el dataset: {len(dataset)}")
print("Primeras entradas:")
for i in range(min(3, len(dataset))):
    print(f"dataset[{i}] = {dataset[i]}")
    print(f"Palabras: {len(dataset[i]['text'].split())}\n")

