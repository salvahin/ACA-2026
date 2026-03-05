import os
import glob

BOOK_DIR = "/Users/salvahin/TC3002B-2026/book"
GITHUB_REPO_URL = "https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/"

# Dependencies categorized by notebook type
GENERAL_DEPS = ["numpy", "pandas", "matplotlib", "seaborn", "scikit-learn"]
ML_DEPS = ["torch", "transformers", "accelerate"]
GPU_DEPS = ["triton"]
GRAMMAR_DEPS = ["xgrammar"]

# Helper to decide what to install
def get_install_command(notebook_name, content_str):
    deps = list(GENERAL_DEPS)
    
    if any(kw in notebook_name.lower() or kw in content_str.lower() for kw in ["ia_", "deep_learning", "transformer", "bert", "gpt", "llm", "fine_tuning", "torch", "pytorch", "model"]):
        deps.extend(ML_DEPS)
        
    if any(kw in notebook_name.lower() or kw in content_str.lower() for kw in ["gpu", "cuda", "triton", "kernel", "memoria", "optimizacion"]):
        deps.extend(GPU_DEPS)
        
    if any(kw in notebook_name.lower() or kw in content_str.lower() for kw in ["grammar", "xgrammar", "constrained", "sampling"]):
        deps.extend(GRAMMAR_DEPS)
        
    deps = list(dict.fromkeys(deps))
    return f"!pip install -q {' '.join(deps)}"

def get_colab_badge_markdown(notebook_filename):
    colab_url = f"{GITHUB_REPO_URL}{notebook_filename}"
    return [
        "\n```{admonition} Ejecutar en Google Colab\n",
        ":class: tip\n\n",
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n",
        "```\n"
    ]

def get_setup_cell_markdown(install_cmd):
    return [
        "\n```{code-cell} ipython3\n",
        ":tags: [remove-input, setup]\n\n",
        "# Setup Colab Environment\n",
        f"{install_cmd}\n",
        "print('Dependencies installed!')\n",
        "```\n"
    ]

def process_md_file(filepath):
    filename = os.path.basename(filepath)
    notebook_filename = filename.replace('.md', '.ipynb')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    content_str = "".join(lines)
    
    # If already processed, skip
    if "https://colab.research.google.com/assets/colab-badge.svg" in content_str:
        return False
        
    # Find the title insertion point
    insert_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('# '):
            insert_idx = i + 1
            break
            
    if insert_idx == -1:
        # If no h1 found, just append after frontmatter or at beginning
        for i, line in enumerate(lines):
            if i > 0 and line.strip() == "---":
                insert_idx = i + 1
                break
        if insert_idx == -1:
            insert_idx = 0
            
    install_cmd = get_install_command(notebook_filename, content_str)
    
    badge_lines = get_colab_badge_markdown(notebook_filename)
    setup_lines = get_setup_cell_markdown(install_cmd)
    
    new_lines = lines[:insert_idx] + badge_lines + setup_lines + lines[insert_idx:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
        
    return True

def main():
    md_files = glob.glob(f"{BOOK_DIR}/**/*.md", recursive=True)
    success_count = 0
    
    for filepath in md_files:
        # Skip index.md and the notebooks directory
        if "index.md" in filepath or "notebooks" in filepath or "_build" in filepath:
            continue
            
        if process_md_file(filepath):
            success_count += 1
            print(f"Injected Colab support into {os.path.basename(filepath)}")
            
    print(f"\nSuccessfully injected Colab support into {success_count} Markdown files.")

if __name__ == "__main__":
    main()
