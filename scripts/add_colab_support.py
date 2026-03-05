import json
import os
import glob
import re

NOTEBOOKS_DIR = "/Users/salvahin/TC3002B-2026/book/notebooks"
GITHUB_REPO_URL = "https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/"

# Dependencies categorized by notebook type
GENERAL_DEPS = ["numpy", "pandas", "matplotlib", "seaborn", "scikit-learn"]
ML_DEPS = ["torch", "transformers", "accelerate"]
GPU_DEPS = ["triton"]
GRAMMAR_DEPS = ["xgrammar"]

# Helper to decide what to install
def get_install_command(notebook_name, content_str):
    deps = list(GENERAL_DEPS)
    
    # Check ML/PyTorch needs
    if any(kw in notebook_name.lower() or kw in content_str.lower() for kw in ["ia_", "deep_learning", "transformer", "bert", "gpt", "llm", "fine_tuning", "torch", "pytorch", "model"]):
        deps.extend(ML_DEPS)
        
    # Check GPU/Triton needs
    if any(kw in notebook_name.lower() or kw in content_str.lower() for kw in ["gpu", "cuda", "triton", "kernel", "memoria", "optimizacion"]):
        deps.extend(GPU_DEPS)
        
    # Check XGrammar needs
    if any(kw in notebook_name.lower() or kw in content_str.lower() for kw in ["grammar", "xgrammar", "constrained", "sampling"]):
        deps.extend(GRAMMAR_DEPS)
        
    # Remove duplicates but preserve order somewhat
    deps = list(dict.fromkeys(deps))
    
    return f"!pip install -q {' '.join(deps)}"

def create_setup_cell(install_cmd):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "tags": ["remove-input", "setup"]
        },
        "outputs": [],
        "source": [
            "# Setup Colab Environment\n",
            install_cmd + "\n",
            "print('Dependencies installed!')"
        ]
    }

def create_colab_badge_cell(notebook_filename):
    colab_url = f"{GITHUB_REPO_URL}{notebook_filename}"
    badge_markdown = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})"
    
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "```{admonition} Ejecutar en Google Colab\n",
            ":class: tip\n",
            "\n",
            badge_markdown + "\n",
            "```\n"
        ]
    }

def process_notebook(filepath):
    filename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {filename}. Skipping.")
            return False

    if "cells" not in nb:
        return False
        
    content_str = json.dumps(nb)
    cells = nb["cells"]
    
    # 1. Clean up existing Colab badges or setup cells to avoid duplicates
    cleaned_cells = []
    has_title = False
    title_index = -1
    
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and any("colab-badge.svg" in line for line in cell.get("source", [])):
            continue # Skip old badge cell
            
        if cell["cell_type"] == "code" and any("!pip install" in line for line in cell.get("source", [])) and "setup" in cell.get("metadata", {}).get("tags", []):
            continue # Skip old setup cell
            
        cleaned_cells.append(cell)
        
        # Find the main title (e.g. # Lectura 1...)
        if not has_title and cell["cell_type"] == "markdown":
            source = cell.get("source", [])
            if source and source[0].strip().startswith("# "):
                has_title = True
                title_index = len(cleaned_cells) - 1

    # If no title found, we just append at the beginning
    insert_idx = title_index + 1 if has_title else 0
    
    # 2. Add New Badge Cell
    badge_cell = create_colab_badge_cell(filename)
    cleaned_cells.insert(insert_idx, badge_cell)
    
    # 3. Add New Setup Cell
    install_cmd = get_install_command(filename, content_str)
    setup_cell = create_setup_cell(install_cmd)
    cleaned_cells.insert(insert_idx + 1, setup_cell)
    
    nb["cells"] = cleaned_cells
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    return True

def main():
    notebooks = glob.glob(os.path.join(NOTEBOOKS_DIR, "*.ipynb"))
    print(f"Found {len(notebooks)} notebooks to process.")
    
    success_count = 0
    for nb_path in notebooks:
        if process_notebook(nb_path):
            success_count += 1
            
    print(f"Successfully processed {success_count} notebooks.")

if __name__ == "__main__":
    main()
