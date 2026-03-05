import json
from collections import defaultdict

with open("notebook_analysis.json", "r") as f:
    data = json.load(f)

# Group by category (AI, Compilers, Stats, Projects, etc)
def get_category(filepath):
    parts = filepath.split('/')
    if len(parts) > 2:
        return parts[1]
    return "other"

results_by_cat = defaultdict(list)
for item in data:
    cat = get_category(item['file'])
    results_by_cat[cat].append(item)

md_lines = ["# Reporte de Compatibilidad de Notebooks\n"]
md_lines.append("Este documento resume la compatibilidad de cada ambiente interactivo (`.ipynb` y `.md` de Jupyter Book) con **entornos locales** (generalmente CPU o M1/M2/M3) y **Google Colab**.\n")

total_files = len(data)
gpu_files = sum(1 for x in data if x['is_gpu'])
colab_ready_files = sum(1 for x in data if x['is_colab'])

md_lines.append("## Resumen General")
md_lines.append(f"- **Total de archivos ejecutables analizados:** {total_files}")
md_lines.append(f"- **Archivos que requieren GPU dedicada (Triton, CUDA, vLLM, etc.):** {gpu_files}")
md_lines.append(f"- **Archivos listos para Colab (`!pip install` explícito):** {colab_ready_files}\n")

for cat, items in sorted(results_by_cat.items()):
    cat_name = cat.capitalize() if cat != "ai" else "AI"
    if cat == "project_1": cat_name = "Proyecto 1"
    if cat == "project_2": cat_name = "Proyecto 2"
    if cat == "notebooks": cat_name = "Notebooks Sueltos"
    
    md_lines.append(f"## Módulo: {cat_name}")
    md_lines.append("| Archivo | Soporte Local | Soporte Colab | Observaciones |")
    md_lines.append("|:---|:---:|:---:|:---|")
    
    for item in sorted(items, key=lambda x: x['file']):
        fname = item['file'].replace('book/', '', 1)
        
        # Local Logic
        str_local = "✅ SÍ" if not item['is_gpu'] else "❌ NO (Requiere GPU)"
        
        # Colab Logic
        if item['is_colab']:
            str_colab = "✅ SÍ"
        else:
            str_colab = "⚠️ Depende del entorno"
            
        # Observations
        obs = []
        if item['is_gpu']:
            obs.append(f"Usa GPU ({', '.join(item['gpu_matches'])[:30]}...)")
        if not item['is_colab']:
            obs.append("Falta `!pip install` explícito")
            
        str_obs = ". ".join(obs) if obs else "Todo bien"
        
        md_lines.append(f"| `{fname}` | {str_local} | {str_colab} | {str_obs} |")
        
    md_lines.append("\n")

with open("/Users/salvahin/.gemini/antigravity/brain/b2a9736c-934d-4b83-80b1-9a79e0895351/reporte_compatibilidad.md", "w") as f:
    f.write("\n".join(md_lines))

print("Reporte generado.")
