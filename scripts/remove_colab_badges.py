#!/usr/bin/env python3
"""
Remueve badges de Colab de archivos que solo tienen código de setup.
"""
import re
from pathlib import Path

BOOK_DIR = Path(__file__).parent.parent / "book"


def has_executable_code_cells(content: str) -> bool:
    """
    Verifica si el archivo tiene code-cells ejecutables REALES.
    Ignora celdas que son solo setup.
    """
    code_cell_pattern = r'```\{code-cell\}[^\n]*\n(.*?)```'
    matches = re.findall(code_cell_pattern, content, re.DOTALL)

    if not matches:
        return False

    setup_patterns = [
        r'^:tags:\s*\[.*setup.*\]',
        r'^\s*#\s*Setup\s',
        r'^\s*!pip\s+install',
        r"^\s*print\(['\"]Dependencies",
        r"^\s*print\(['\"]Módulo",
        r'^\s*import\s+sys\s*\n\s*if\s+[\'"]google\.colab',
    ]

    for cell_content in matches:
        cell_lines = cell_content.strip()

        is_setup_only = False
        for pattern in setup_patterns:
            if re.search(pattern, cell_lines, re.MULTILINE | re.IGNORECASE):
                is_setup_only = True
                break

        if not is_setup_only:
            non_comment_lines = [
                line for line in cell_lines.split('\n')
                if line.strip() and not line.strip().startswith('#') and not line.strip().startswith(':')
            ]
            if non_comment_lines:
                return True

    return False


def has_colab_badge(content: str) -> bool:
    """Verifica si ya tiene un badge de Colab."""
    return "colab-badge.svg" in content or "Open In Colab" in content


def remove_colab_badge(content: str) -> str:
    """Remueve el bloque de badge de Colab."""
    # Patrón para el admonition completo del badge
    pattern = r'\n*```\{admonition\}\s*Ejecutar en Google Colab\s*\n:class:\s*tip\s*\n\n\[!\[Open In Colab\].*?\]\(.*?\)\s*\n```\n*'
    return re.sub(pattern, '\n', content)


def remove_setup_code_cell(content: str) -> str:
    """Remueve la celda de setup de Colab si existe."""
    # Patrón para la celda de setup de Colab
    pattern = r'\n*```\{code-cell\}\s*ipython3\s*\n#\s*Setup\s+condicional\s+para\s+Google\s+Colab.*?```\n*'
    return re.sub(pattern, '\n', content, flags=re.DOTALL)


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Procesa un archivo y remueve badge si no tiene código ejecutable."""
    content = filepath.read_text(encoding="utf-8")

    # Solo procesar si tiene badge pero no tiene código ejecutable
    if not has_colab_badge(content):
        return False

    if has_executable_code_cells(content):
        return False

    # Remover badge y celda de setup
    new_content = remove_colab_badge(content)
    new_content = remove_setup_code_cell(new_content)

    if new_content != content:
        if not dry_run:
            filepath.write_text(new_content, encoding="utf-8")
        return True

    return False


def main(dry_run: bool = False):
    print("=" * 60)
    print("REMOVIENDO BADGES DE COLAB INNECESARIOS")
    if dry_run:
        print("(MODO SIMULACIÓN - no se modifican archivos)")
    print("=" * 60)

    modules = ["ai", "compilers", "stats", "project_1", "project_2", "research"]
    total_removed = 0

    for module in modules:
        module_dir = BOOK_DIR / module
        if not module_dir.exists():
            continue

        for md_file in sorted(module_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue

            if process_file(md_file, dry_run):
                rel_path = md_file.relative_to(BOOK_DIR)
                print(f"  - {rel_path}")
                total_removed += 1

    print(f"\n{'=' * 60}")
    action = "Se removerían" if dry_run else "Badges removidos:"
    print(f"{action} {total_removed}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    main(dry_run)
