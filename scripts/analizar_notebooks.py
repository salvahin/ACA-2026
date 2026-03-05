import os
import glob
import json
import re
from pathlib import Path

# Paths
BOOK_DIR = Path("book")

# Regexes for MD files to find code blocks
# Matches ```python ... ``` or ```{code-cell} ipython3 ... ```
MD_CODE_BLOCK_RE = re.compile(r"```(?:python|{code-cell} ipython3)\n(.*?)```", re.DOTALL)

# Patterns to detect
GPU_PATTERNS = [
    r"\.cuda\(\)",
    r"device\s*=\s*[\"']cuda",
    r"cuda:\d+",
    r"import bitsandbytes",
    r"from bitsandbytes",
    r"import triton",
    r"import vllm",
    r"from vllm",
    r"import auto_gptq",
    r"load_in_8bit\s*=\s*True",
    r"load_in_4bit\s*=\s*True"
]

COLAB_PATTERNS = [
    r"!pip install",
    r"%pip install",
    r"from google\.colab import",
    r"import google\.colab"
]

def analyze_code_content(code):
    is_gpu = False
    is_colab = False
    gpu_matches = []
    colab_matches = []

    for pattern in GPU_PATTERNS:
        matches = re.findall(pattern, code)
        if matches:
            is_gpu = True
            gpu_matches.extend(matches)
            
    for pattern in COLAB_PATTERNS:
        matches = re.findall(pattern, code)
        if matches:
            is_colab = True
            colab_matches.extend(matches)
            
    return {
        "is_gpu": is_gpu,
        "is_colab": is_colab,
        "gpu_matches": list(set(gpu_matches)),
        "colab_matches": list(set(colab_matches))
    }

def analyze_ipynb(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        code_cells = [
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        
        full_code = "\n".join(code_cells)
        return analyze_code_content(full_code)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_md(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        code_blocks = MD_CODE_BLOCK_RE.findall(content)
        full_code = "\n".join(code_blocks)
        
        if not full_code.strip():
            # If no python code cells found, it's just a text MD file
            return {"is_gpu": False, "is_colab": False, "gpu_matches": [], "colab_matches": [], "empty": True}
            
        return analyze_code_content(full_code)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    results = []
    
    # Exclude _build
    files = []
    for root, dirs, filenames in os.walk(BOOK_DIR):
        if "_build" in root:
            continue
        for filename in filenames:
            if filename.endswith(".ipynb") or filename.endswith(".md"):
                files.append(os.path.join(root, filename))
                
    files.sort()
    
    for f in files:
        if f.endswith(".ipynb"):
            res = analyze_ipynb(f)
        else:
            res = analyze_md(f)
            if res and res.get("empty"):
                continue # Skip pure text markdown files
                
        if res:
            res["file"] = f
            results.append(res)
            
    # Generate JSON output
    with open("notebook_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Analyzed {len(results)} executable files.")
    print("Results saved to notebook_analysis.json")

if __name__ == "__main__":
    main()
