import os

BOOK_DIR = "/Users/salvahin/TC3002B-2026/book"
filepath = os.path.join(BOOK_DIR, "ai", "03_generacion_autoregresiva.md")

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# We want to insert this before "## Reflexión y Ejercicios"
target_heading = "## Reflexión y Ejercicios"

if target_heading in content and "Implementando la Generación Paso a Paso" not in content:
    insertion = """
## Implementando la Generación Paso a Paso (PyTorch)

Para quitarle la "magia" a las librerías como HuggingFace, vamos a implementar nosotros mismos el ciclo autoregresivo exacto usando un modelo de lenguaje real (GPT-2).

```{code-cell} ipython3
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# 1. Cargar modelo pequeño y tokenizer
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)
model.eval()

# 2. Entrada inicial (Prompt)
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

print(f"Texto original: '{text}'")
print(f"Tokens de entrada (IDs): {inputs['input_ids'][0].tolist()}")

# 3. Generación Autoregresiva Manual (Paso a paso)
max_new_tokens = 5
current_input_ids = inputs['input_ids']

print("\\nIniciando generación iterativa...")
print("-" * 50)

with torch.no_grad():
    for step in range(max_new_tokens):
        # A) Forward pass: pasar todos los tokens actuales por el transformer
        outputs = model(input_ids=current_input_ids)
        
        # B) Extraer los logits de la ÚLTIMA posición generada
        # outputs.logits tiene forma: [batch_size, sequence_length, vocab_size]
        next_token_logits = outputs.logits[0, -1, :] 
        
        # C) Selección del siguiente token (Greedy Decoding: argmax)
        next_token_id = torch.argmax(next_token_logits)
        
        # D) Decodificar para ver qué palabra eligió
        next_word = tokenizer.decode(next_token_id)
        
        print(f"Paso {step+1} | Token ID: {next_token_id.item():>5} | Palabra: '{next_word}'")
        
        # E) Agregar el nuevo token a la secuencia para el siguiente paso (Autoregresivo!)
        current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)

print("-" * 50)
print(f"Texto final completo: '{tokenizer.decode(current_input_ids[0])}'")
```

Como puedes ver, `model.generate()` simplemente oculta este ciclo `for` detrás de una función fácil de usar. Cada palabra generada obliga al modelo a recalcular todo el contexto anterior.

"""
    new_content = content.replace(target_heading, insertion + target_heading)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Injected step-by-step PyTorch generation into 03_generacion_autoregresiva.md")
else:
    print("Could not find target heading or already injected.")
