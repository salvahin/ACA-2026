import json
import os

NOTEBOOKS_DIR = "/Users/salvahin/TC3002B-2026/book/notebooks"

def patch_notebooks():
    nb2_path = os.path.join(NOTEBOOKS_DIR, "02_fundamentos_deep_learning.ipynb")
    
    with open(nb2_path, 'r', encoding='utf-8') as f:
        nb2 = json.load(f)

    for cell in nb2.get('cells', []):
        if cell.get('cell_type') == 'code':
            lines = cell.get('source', [])
            for i, line in enumerate(lines):
                # The line with the issue
                if line.strip() == '"ax.text(0.9, 0.65, \'Salida\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\n"':
                    # It was accidentally saved as string literal containing Python code literally
                    # Let's fix it so it actually is the Python code:
                    lines[i] = "    ax.text(0.9, 0.65, 'Salida\\n(Sigmoid)', ha='center', fontsize=9)\n"
                    print("Patched the array element to be valid Python")

    with open(nb2_path, 'w', encoding='utf-8') as f:
        json.dump(nb2, f, indent=1)

    print("Notebook patches applied successfully.")

if __name__ == "__main__":
    patch_notebooks()
