import json
import os

NOTEBOOKS_DIR = "/Users/salvahin/TC3002B-2026/book/notebooks"

def patch_notebook(filename, old_str, new_str):
    path = os.path.join(NOTEBOOKS_DIR, filename)
    if not os.path.exists(path):
        print(f"File {filename} not found.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Patched {filename} successfully.")
    else:
        print(f"⚠️ Target string not found in {filename}.")

def run_all_patches():
    # 1. Fix Plotly 1-based indexing in 04_triton_completo.ipynb (col=idx -> col=idx+1)
    old_04 = '                             showscale=False), row=1, col=idx)\\n"'
    new_04 = '                             showscale=False), row=1, col=idx+1)\\n"'
    patch_notebook("04_triton_completo.ipynb", old_04, new_04)

    # 2. Fix missing LLM definition in 05-reproducibilidad.ipynb
    # The error says `llm` is not defined. I will comment out the code block since it looks like pseudo-code
    old_05 = '    "# Baseline (determin\u00edstico)\\n",\n    "response = llm.generate(prompt, temperature=0.0, seed=42)\\n",\n    "\\n",\n    "# Con restricciones (determin\u00edstico)\\n",\n    "response = llm.generate(prompt, temperature=0.0, seed=42)"\n   ]'
    new_05 = '    "# Baseline (determin\u00edstico)\\n",\n    "# response = llm.generate(prompt, temperature=0.0, seed=42)\\n",\n    "\\n",\n    "# Con restricciones (determin\u00edstico)\\n",\n    "# response = llm.generate(prompt, temperature=0.0, seed=42)"\n   ]'
    
    # Actually wait, maybe it's just meant to be pseudo code so changing the cell type to markdown is better,
    # but string replacing with # is safest for JSON structural integrity.
    patch_notebook("05-reproducibilidad.ipynb", old_05, new_05)

    # 3. Fix unclosed print string in 08_serving_llms.ipynb
    old_08 = '    "print(\\""'
    new_08 = '    "print(\\"\\")"'
    # This might be too generic. Let me find the exact broken line.
    
    # 4. Fix unclosed print string in 09_token_economics_integracion.ipynb
    # Usually ending print(" is due to a multi-line string that got truncated or syntactically broken by the markdown->ipynb converter.

if __name__ == "__main__":
    run_all_patches()
