#!/usr/bin/env python3
"""
Corrige la numeración de figuras en las lecturas.
Renumera secuencialmente empezando en 1 para cada archivo.
"""
import re
from pathlib import Path

BOOK_DIR = Path(__file__).parent.parent / "book"


def fix_figure_numbers(filepath: Path, dry_run: bool = False) -> tuple[bool, list]:
    """
    Corrige la numeración de figuras en un archivo.
    Returns: (was_modified, changes_made)
    """
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Encontrar todas las figuras con sus posiciones
    # Patrón: **Figura N:** o **Figura N** o ***Figura N:**
    pattern = r'(\*{2,3}Figura\s+)(\d+(?:\.\d+)?)([\*:])'

    # Encontrar todas las coincidencias
    matches = list(re.finditer(pattern, content))

    if not matches:
        return False, []

    # Crear mapa de números viejos a nuevos
    changes = []
    seen_positions = {}  # Para manejar duplicados (img + caption)

    # Agrupar figuras por posición aproximada (dentro de 10 líneas)
    figure_groups = []
    current_group = []
    last_pos = -1000

    for match in matches:
        # Calcular línea aproximada
        line_num = content[:match.start()].count('\n')
        old_num = match.group(2)

        if line_num - last_pos <= 5 and current_group:
            # Misma figura (imagen + caption)
            current_group.append((match, old_num, line_num))
        else:
            if current_group:
                figure_groups.append(current_group)
            current_group = [(match, old_num, line_num)]

        last_pos = line_num

    if current_group:
        figure_groups.append(current_group)

    # Renumerar
    new_content = content
    offset = 0

    for new_num, group in enumerate(figure_groups, 1):
        for match, old_num, line_num in group:
            if old_num != str(new_num):
                changes.append(f"  Línea ~{line_num+1}: Figura {old_num} → Figura {new_num}")

                # Calcular nueva posición con offset
                start = match.start() + offset
                end = match.end() + offset

                # Construir reemplazo
                prefix = match.group(1)
                suffix = match.group(3)
                old_text = match.group(0)
                new_text = f"{prefix}{new_num}{suffix}"

                new_content = new_content[:start] + new_text + new_content[end:]
                offset += len(new_text) - len(old_text)

    if new_content != original:
        if not dry_run:
            filepath.write_text(new_content, encoding="utf-8")
        return True, changes

    return False, []


def main(dry_run: bool = False):
    print("=" * 70)
    print("CORRECCIÓN DE NUMERACIÓN DE FIGURAS")
    if dry_run:
        print("(MODO SIMULACIÓN)")
    print("=" * 70)

    modules = ["ai", "compilers", "stats", "project_1", "project_2", "research"]
    total_fixed = 0

    for module in modules:
        module_dir = BOOK_DIR / module
        if not module_dir.exists():
            continue

        for md_file in sorted(module_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue

            modified, changes = fix_figure_numbers(md_file, dry_run)
            if modified:
                rel_path = md_file.relative_to(BOOK_DIR)
                print(f"\n📄 {rel_path}")
                for change in changes:
                    print(change)
                total_fixed += 1

    print(f"\n{'=' * 70}")
    action = "Se corregirían" if dry_run else "Archivos corregidos:"
    print(f"{action} {total_fixed}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    main(dry_run)
