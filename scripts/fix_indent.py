import json
import os

NOTEBOOKS_DIR = "/Users/salvahin/TC3002B-2026/book/notebooks"

def clean_trailing_prints(filename):
    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            lines = cell.get('source', [])
            for i, line in enumerate(lines):
                 if line.startswith("    ax.text(0.9, 0.65, 'Salida\\n"):
                      lines[i] = "ax.text(0.9, 0.65, 'Salida\\n(Sigmoid)', ha='center', fontsize=9)\\n"
                      print(f"[{filename}] Fixed indentation of ax.text")
                
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Cleaned {filename}")

if __name__ == "__main__":
    clean_trailing_prints("02_fundamentos_deep_learning.ipynb")
