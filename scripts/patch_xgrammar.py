import os

BOOK_DIR = "/Users/salvahin/TC3002B-2026/book"
filepath = os.path.join(BOOK_DIR, "project_1", "04_xgrammar_constrained.md")

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# We want to replace the standard generation loop in "Flujo Completo: De Schema a Generación"
# with an ACA framework pedagogical block.

old_loop = """
with torch.no_grad():
    for _ in range(100):  # Máximo 100 tokens
        outputs = model(torch.tensor([input_ids]))
        logits = outputs.logits[0, -1, :]

        # Aplicar restricciones
        logits = processor(input_ids, logits)

        # Muestreo
        token_id = torch.argmax(logits).item()
        input_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break
"""

new_loop = """
with torch.no_grad():
    for step in range(100):  # Máximo 100 tokens
        outputs = model(torch.tensor([input_ids]))
        raw_logits = outputs.logits[0, -1, :].clone()  # Guardamos copia para inspección

        # ACA Framework: Inspeccionemos el PRIMER token antes y después de XGrammar
        if step == 0:
            print("\\n--- ACA INTERCEPT: Token 0 ---")
            top_raw = torch.topk(raw_logits, 3)
            print("Top 3 tokens que el modelo QUERÍA generar (Sin Restricción):")
            for val, idx in zip(top_raw.values, top_raw.indices):
                print(f"  '{tokenizer.decode(idx)}' (Logit: {val.item():.2f})")
        
        # Aplicar restricciones de la Gramática (El FSM entra en acción)
        masked_logits = processor(input_ids, raw_logits)

        if step == 0:
            top_masked = torch.topk(masked_logits, 3)
            print("\\nTop 3 tokens PERMITIDOS por la gramática:")
            for val, idx in zip(top_masked.values, top_masked.indices):
                # Valid tokens retain their logits, invalid ones hit -inf
                print(f"  '{tokenizer.decode(idx)}' (Logit: {val.item():.2f})")
            print("--------------------------------\\n")

        # Seleccionar el token más probable de los Permitidos
        token_id = torch.argmax(masked_logits).item()
        input_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break
"""

if old_loop in content and "ACA INTERCEPT" not in content:
    new_content = content.replace(old_loop, new_loop)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Injected logits inspection loop into 04_xgrammar_constrained.md")
else:
    print("Could not find the target loop or already injected.")
