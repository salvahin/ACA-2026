import json
import os

NOTEBOOKS_DIR = "/Users/salvahin/TC3002B-2026/book/notebooks"

def patch_notebooks():
    nb2_path = os.path.join(NOTEBOOKS_DIR, "02_fundamentos_deep_learning.ipynb")
    
    with open(nb2_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # The erroneous string in JSON is:
    erroneous = '"ax.text(0.9, 0.5, f\'{a2[0]:.3f}\', ha=\'center\', va=\'center\', fontsize=10, weight=\'bold\')\\n",\n    "ax.text(0.9, 0.65, \'Salida\\\\\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\\\n"\\n'
    # Wait, it's safer to just replace the broken chunk directly
    
    if '"ax.text(0.9, 0.65, \'Salida\\\\\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\\\n"\\n' in content:
         content = content.replace('"ax.text(0.9, 0.65, \'Salida\\\\\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\\\n"\\n', '"ax.text(0.9, 0.65, \'Salida\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\n"')
    elif '    "ax.text(0.9, 0.65, \'Salida\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\n"\\n' in content:
         content = content.replace('    "ax.text(0.9, 0.65, \'Salida\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\n"\\n', '    "ax.text(0.9, 0.65, \'Salida\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\n"')

    with open(nb2_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Notebook patches applied successfully.")

if __name__ == "__main__":
    patch_notebooks()
