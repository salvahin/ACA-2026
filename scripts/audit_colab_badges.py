#!/usr/bin/env python3
"""
Audita los badges de Colab en las lecturas.
Identifica archivos que tienen badges pero no deberían (solo código de setup).
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


def audit_files():
    """Audita todos los archivos para identificar badges incorrectos."""
    print("=" * 60)
    print("AUDITORÍA DE BADGES DE COLAB")
    print("=" * 60)

    modules = ["ai", "compilers", "stats", "project_1", "project_2", "research"]

    files_with_badge_ok = []
    files_with_badge_should_remove = []
    files_without_badge_ok = []
    files_without_badge_should_add = []

    for module in modules:
        module_dir = BOOK_DIR / module
        if not module_dir.exists():
            continue

        for md_file in sorted(module_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue

            content = md_file.read_text(encoding="utf-8")
            has_badge = has_colab_badge(content)
            has_code = has_executable_code_cells(content)

            rel_path = md_file.relative_to(BOOK_DIR)

            if has_badge and has_code:
                files_with_badge_ok.append(rel_path)
            elif has_badge and not has_code:
                files_with_badge_should_remove.append(rel_path)
            elif not has_badge and has_code:
                files_without_badge_should_add.append(rel_path)
            else:
                files_without_badge_ok.append(rel_path)

    print(f"\n✅ CON BADGE (correcto) - {len(files_with_badge_ok)} archivos:")
    for f in files_with_badge_ok:
        print(f"   {f}")

    print(f"\n❌ CON BADGE (REMOVER) - {len(files_with_badge_should_remove)} archivos:")
    for f in files_with_badge_should_remove:
        print(f"   {f}")

    print(f"\n⚠️  SIN BADGE (agregar) - {len(files_without_badge_should_add)} archivos:")
    for f in files_without_badge_should_add:
        print(f"   {f}")

    print(f"\n✅ SIN BADGE (correcto) - {len(files_without_badge_ok)} archivos:")
    for f in files_without_badge_ok:
        print(f"   {f}")

    print("\n" + "=" * 60)

    return files_with_badge_should_remove


if __name__ == "__main__":
    to_remove = audit_files()
    if to_remove:
        print(f"\nSe recomienda remover badges de {len(to_remove)} archivos.")
        print("Ejecuta: python scripts/remove_colab_badges.py")
