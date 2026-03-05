import os
import json
import re
from pathlib import Path

# Cargar el reporte de análisis
with open("notebook_analysis.json", "r", encoding="utf-8") as f:
    analysis = json.load(f)

# Código a inyectar
COLAB_SETUP_CODE = """# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido."""

GPU_FALLBACK_CODE = """import torch
import warnings

# Selección dinámica de dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    warnings.warn("No se detectó un acelerador (GPU/MPS). La ejecución será lenta.")

print(f"Usando dispositivo: {device}")"""

GPU_STRICT_CODE = """import torch
import sys

# Validación temprana de entorno para cuadernos estrictamente acoplados a NVidia (Triton/vLLM)
if not torch.cuda.is_available():
    print("❌ ADVERTENCIA: Este notebook requiere una GPU NVidia y arquitectura CUDA para funcionar.")
    print("Por favor, sube este notebook a Google Colab y selecciona Entorno de ejecución -> Cambiar tipo de entorno -> T4 GPU o superior.")
    sys.exit("Entorno incompatible: Sistema sin CUDA detectado.")
else:
    print("✅ Entorno GPU detectado compatible con los requerimientos.")"""

def inject_ipynb(filepath, item_data):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        cells = nb.get("cells", [])
        if not cells: return
        
        # 1. ¿Necesita Setup de Colab?
        if not item_data["is_colab"]:
            # Insertar como primera celda
            setup_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"tags": ["hide-cell"]},
                "outputs": [],
                "source": [line + "\n" for line in COLAB_SETUP_CODE.split("\n")]
            }
            cells.insert(0, setup_cell)
            print(f"  [+] Añadido bloque Colab a {filepath}")
            
        # 2. ¿Necesita Graceful Exit o Fallback dinámico?
        if item_data["is_gpu"]:
            # Revisar si se mencionó Triton, auto_gptq, o vLLM (estricto)
            is_strict = any(x in str(item_data["gpu_matches"]) for x in ["triton", "vllm", "auto_gptq", "bitsandbytes"])
            
            code_to_inject = GPU_STRICT_CODE if is_strict else GPU_FALLBACK_CODE
            
            # Buscar dónde insertarlo (después de la metadata de Colab si existe, o al principio)
            target_idx = 1 if not item_data["is_colab"] else 0
            
            gpu_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_to_inject.split("\n")]
            }
            cells.insert(target_idx, gpu_cell)
            print(f"  [+] Añadido bloque [{'Estricto' if is_strict else 'Fallback'}] de Hardware a {filepath}")
            
        # Guardar cambios
        nb["cells"] = cells
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")


def modify_md_files(filepath, item_data):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Usar advertencias MD blocktypes estándar de ipython
        changed = False
        
        # 1. ¿Necesita Setup de Colab?
        if not item_data["is_colab"]:
            colab_md_block = f"```{{code-cell}} ipython3\n{COLAB_SETUP_CODE}\n```\n\n"
            # Insertar justo después del título H1
            content = re.sub(r'^(# .*?\n)', r'\1\n' + colab_md_block, content, count=1, flags=re.MULTILINE)
            changed = True
            print(f"  [+] Añadido bloque Colab a {filepath}")
            
        # 2. Requerimientos de GPU
        if item_data["is_gpu"]:
            is_strict = any(x in str(item_data["gpu_matches"]) for x in ["triton", "vllm", "auto_gptq", "bitsandbytes"])
            code_to_inject = GPU_STRICT_CODE if is_strict else GPU_FALLBACK_CODE
            gpu_md_block = f"```{{code-cell}} ipython3\n{code_to_inject}\n```\n\n"
            
            # Insertar después del H1 o después del bloque Colab
            content = re.sub(r'^(# .*?\n)', r'\1\n' + gpu_md_block, content, count=1, flags=re.MULTILINE)
            changed = True
            print(f"  [+] Añadido bloque [{'Estricto' if is_strict else 'Fallback'}] de Hardware a {filepath}")

        if changed:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
                
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")

def main():
    print("Iniciando inyección de compatibilidad...")
    for item in analysis:
        # Solo modificar si realmente falta colab o requiere GPU settings
        if not item["is_colab"] or item["is_gpu"]:
            filepath = item["file"]
            if not os.path.exists(filepath):
                continue
                
            if filepath.endswith(".ipynb"):
                inject_ipynb(filepath, item)
            elif filepath.endswith(".md"):
                modify_md_files(filepath, item)

if __name__ == "__main__":
    main()
