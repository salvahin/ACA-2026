#!/usr/bin/env python3
"""
Agrega badges de Google Colab a cada lectura del Jupyter Book.
"""
import re
from pathlib import Path

BOOK_DIR = Path(__file__).parent.parent / "book"
REPO = "salvahin/ACA-2026"
BRANCH = "main"

# Plantilla del badge de Colab
COLAB_BADGE = """
```{{admonition}} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{repo}/blob/{branch}/book/{path})
```
"""

def has_code_cells(content: str) -> bool:
    """Verifica si el archivo tiene code-cells ejecutables."""
    return "```{code-cell}" in content

def has_colab_badge(content: str) -> bool:
    """Verifica si ya tiene un badge de Colab."""
    return "colab-badge.svg" in content or "Open In Colab" in content

def add_badge_after_header(content: str, badge: str) -> str:
    """Agrega el badge después del primer header H1."""
    # Buscar el primer header nivel 1
    pattern = r"(^# .+\n)"
    match = re.search(pattern, content, re.MULTILINE)

    if match:
        insert_pos = match.end()
        return content[:insert_pos] + "\n" + badge + "\n" + content[insert_pos:]

    # Si no hay H1, agregar al inicio después del front matter
    if content.startswith("---"):
        # Encontrar el cierre del front matter
        end_fm = content.find("---", 3)
        if end_fm != -1:
            insert_pos = end_fm + 4
            return content[:insert_pos] + "\n" + badge + "\n" + content[insert_pos:]

    return badge + "\n" + content

def process_file(filepath: Path) -> bool:
    """Procesa un archivo y agrega badge si tiene code-cells."""
    content = filepath.read_text(encoding="utf-8")

    # Saltar si no tiene code-cells o ya tiene badge
    if not has_code_cells(content):
        return False

    if has_colab_badge(content):
        return False

    # Calcular path relativo para Colab
    rel_path = filepath.relative_to(BOOK_DIR)
    badge = COLAB_BADGE.format(repo=REPO, branch=BRANCH, path=str(rel_path))

    # Agregar badge
    new_content = add_badge_after_header(content, badge)
    filepath.write_text(new_content, encoding="utf-8")

    return True

def main():
    print("=" * 50)
    print("Agregando badges de Google Colab")
    print("=" * 50)

    modules = ["ai", "compilers", "stats", "project_1", "project_2"]
    total_added = 0

    for module in modules:
        module_dir = BOOK_DIR / module
        if not module_dir.exists():
            continue

        print(f"\n[{module}]")
        for md_file in sorted(module_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue

            if process_file(md_file):
                print(f"  + {md_file.name}")
                total_added += 1

    print(f"\n{'=' * 50}")
    print(f"Badges agregados: {total_added}")
    print("=" * 50)

if __name__ == "__main__":
    main()
