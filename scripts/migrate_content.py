#!/usr/bin/env python3
"""
Script para migrar contenido de lecturas/ a book/ para Jupyter Book.
"""
import os
import re
import shutil
from pathlib import Path

# Configuración
SOURCE = Path(__file__).parent.parent / "lecturas"
DEST = Path(__file__).parent.parent / "book"

# Mapeo de módulos (origen -> destino)
MODULES = {
    "AI": "ai",
    "Compilers": "compilers",
    "Stats": "stats",
    "Research": "research",
    "Project_1": "project_1",
    "Project_2": "project_2",
}


def add_myst_header(content: str, filename: str) -> str:
    """Agrega front matter MyST si no existe."""
    if content.startswith("---"):
        return content

    header = f"""---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

"""
    return header + content


def normalize_filename(name: str) -> str:
    """Normaliza nombre de archivo a minúsculas."""
    if name == "README.md":
        return "index.md"
    return name.lower()


def migrate_module(src_name: str, dest_name: str):
    """Migra un módulo completo."""
    src_path = SOURCE / src_name
    dest_path = DEST / dest_name

    if not src_path.exists():
        print(f"  [SKIP] No existe: {src_path}")
        return

    dest_path.mkdir(parents=True, exist_ok=True)

    # Migrar archivos markdown
    for md_file in sorted(src_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        content = add_myst_header(content, md_file.name)

        dest_filename = normalize_filename(md_file.name)
        dest_file = dest_path / dest_filename

        dest_file.write_text(content, encoding="utf-8")
        print(f"  {md_file.name} -> {dest_filename}")

    # Copiar diagrams/
    diagrams_src = src_path / "diagrams"
    diagrams_dest = dest_path / "diagrams"

    if diagrams_src.exists():
        if diagrams_dest.exists():
            shutil.rmtree(diagrams_dest)
        shutil.copytree(diagrams_src, diagrams_dest)

        num_files = len(list(diagrams_dest.glob("*")))
        print(f"  diagrams/ ({num_files} archivos)")


def main():
    print("=" * 50)
    print("Migrando contenido a Jupyter Book")
    print("=" * 50)
    print(f"Origen:  {SOURCE}")
    print(f"Destino: {DEST}")
    print()

    for src, dest in MODULES.items():
        print(f"\n[{src}] -> book/{dest}/")
        migrate_module(src, dest)

    print("\n" + "=" * 50)
    print("Migración completada!")
    print("=" * 50)
    print("\nPróximos pasos:")
    print("  1. pip install -r requirements.txt")
    print("  2. jupyter-book build book/")
    print("  3. open book/_build/html/index.html")


if __name__ == "__main__":
    main()
