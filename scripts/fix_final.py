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
                 if "circle = plt.Circle((0.9, 0.5), 0.05, color='lightcoral', ec='black', linewidth=2)" in line:
                     # Delete everything from here to the end of this block and rewrite it precisely
                     try:
                         # We'll just hard-replace the problematic indices 
                         # We know the specific surrounding lines
                         
                         idx1 = i
                         # Assuming there are 3 lines for the final output neuron plotting
                         lines[idx1] = "circle = plt.Circle((0.9, 0.5), 0.05, color='lightcoral', ec='black', linewidth=2)\\n"
                         lines[idx1+1] = "ax.add_patch(circle)\\n"
                         lines[idx1+2] = "ax.text(0.9, 0.5, f'{a2[0]:.3f}', ha='center', va='center', fontsize=10, weight='bold')\\n"
                         lines[idx1+3] = "ax.text(0.9, 0.65, 'Salida\\n(Sigmoid)', ha='center', fontsize=9)\\n"
                         print(f"[{filename}] Performed exact surgery on index {idx1}")
                     except IndexError:
                         pass
                
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Cleaned {filename}")

if __name__ == "__main__":
    clean_trailing_prints("02_fundamentos_deep_learning.ipynb")
