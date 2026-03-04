#!/usr/bin/env python3
"""
Genera notebooks .ipynb a partir de los archivos MyST .md usando jupytext.
Los notebooks se guardan junto a los archivos .md para que Colab pueda abrirlos.
"""
import subprocess
from pathlib import Path

BOOK_DIR = Path(__file__).parent.parent / "book"

def has_code_cells(filepath: Path) -> bool:
    """Verifica si el archivo tiene code-cells ejecutables."""
    content = filepath.read_text(encoding="utf-8")
    return "```{code-cell}" in content

def convert_to_notebook(md_path: Path) -> bool:
    """Convierte un archivo .md a .ipynb usando jupytext."""
    ipynb_path = md_path.with_suffix(".ipynb")

    # Intentar con path completo o python -m
    jupytext_commands = [
        ["/Users/salvahin/Library/Python/3.9/bin/jupytext", "--to", "notebook", str(md_path), "-o", str(ipynb_path)],
        ["python3", "-m", "jupytext", "--to", "notebook", str(md_path), "-o", str(ipynb_path)],
    ]

    for cmd in jupytext_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            continue

    print("Error: jupytext no está instalado. Ejecuta: pip install jupytext")
    return False

def main():
    print("=" * 50)
    print("Generando notebooks para Google Colab")
    print("=" * 50)

    modules = ["ai", "compilers", "stats", "project_1", "project_2"]
    total_generated = 0

    for module in modules:
        module_dir = BOOK_DIR / module
        if not module_dir.exists():
            continue

        print(f"\n[{module}]")
        for md_file in sorted(module_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue

            if not has_code_cells(md_file):
                continue

            if convert_to_notebook(md_file):
                print(f"  + {md_file.stem}.ipynb")
                total_generated += 1
            else:
                print(f"  ! Error: {md_file.name}")

    print(f"\n{'=' * 50}")
    print(f"Notebooks generados: {total_generated}")
    print("=" * 50)
    print("\nPróximo paso: actualizar badges para apuntar a .ipynb")

if __name__ == "__main__":
    main()
