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
            # If the last line is essentially a broken print, clear it
            if lines and 'print("' in lines[-1] and not lines[-1].strip().endswith('")') and not lines[-1].strip().endswith('")\\n'):
                print(f"[{filename}] Found broken print: {lines[-1]!r}")
                # We can just remove the broken print line entirely, it's just a dangling artifact
                lines.pop()
                if lines and lines[-1].endswith("\\n"):
                     lines[-1] = lines[-1].rstrip("\\n")
                
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Cleaned {filename}")

if __name__ == "__main__":
    clean_trailing_prints("08_serving_llms.ipynb")
    clean_trailing_prints("09_token_economics_integracion.ipynb")
