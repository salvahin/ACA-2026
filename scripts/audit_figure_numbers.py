#!/usr/bin/env python3
"""
Audita la numeración de figuras en las lecturas.
Cada lectura debe tener figuras numeradas secuencialmente empezando en 1.
"""
import re
from pathlib import Path
from collections import defaultdict

BOOK_DIR = Path(__file__).parent.parent / "book"


def extract_figure_numbers(content: str) -> list[tuple[int, str, int]]:
    """
    Extrae todas las referencias a figuras con su número de línea.
    Returns: [(line_num, match_text, figure_number), ...]
    """
    figures = []
    lines = content.split('\n')

    # Patrones para encontrar figuras
    patterns = [
        r'\*\*Figura\s+(\d+(?:\.\d+)?)[:\*]',  # **Figura 1:** o **Figura 1*
        r':name:\s*fig-',  # MyST figure directive (marca presencia pero no número)
    ]

    for line_num, line in enumerate(lines, 1):
        # Buscar **Figura N:**
        match = re.search(r'\*\*Figura\s+(\d+(?:\.\d+)?)', line)
        if match:
            try:
                fig_num = float(match.group(1))
                figures.append((line_num, line.strip()[:80], fig_num))
            except ValueError:
                pass

    return figures


def audit_file(filepath: Path) -> dict:
    """Audita un archivo y retorna información sobre sus figuras."""
    content = filepath.read_text(encoding="utf-8")
    figures = extract_figure_numbers(content)

    if not figures:
        return None

    # Agrupar figuras dentro de 5 líneas (imagen + caption = misma figura)
    grouped_figures = []
    last_line = -100
    last_num = -1

    for line_num, text, fig_num in figures:
        if line_num - last_line <= 5 and fig_num == last_num:
            # Misma figura (imagen + caption)
            continue
        grouped_figures.append((line_num, text, fig_num))
        last_line = line_num
        last_num = fig_num

    # Verificar si la numeración es correcta (1, 2, 3, ...)
    expected = 1
    issues = []

    for line_num, text, fig_num in grouped_figures:
        if fig_num != expected:
            issues.append({
                'line': line_num,
                'text': text,
                'found': fig_num,
                'expected': expected
            })
        expected = int(fig_num) + 1  # Siguiente esperado

    return {
        'figures': grouped_figures,
        'issues': issues,
        'total': len(grouped_figures)
    }


def main():
    print("=" * 70)
    print("AUDITORÍA DE NUMERACIÓN DE FIGURAS")
    print("=" * 70)

    modules = ["ai", "compilers", "stats", "project_1", "project_2", "research"]
    files_with_issues = []
    files_ok = []

    for module in modules:
        module_dir = BOOK_DIR / module
        if not module_dir.exists():
            continue

        for md_file in sorted(module_dir.glob("*.md")):
            if md_file.name == "index.md":
                continue

            result = audit_file(md_file)
            if result is None:
                continue

            rel_path = md_file.relative_to(BOOK_DIR)

            if result['issues']:
                files_with_issues.append((rel_path, result))
            else:
                files_ok.append((rel_path, result['total']))

    # Mostrar archivos con problemas
    print(f"\n❌ ARCHIVOS CON PROBLEMAS ({len(files_with_issues)}):\n")

    for rel_path, result in files_with_issues:
        print(f"📄 {rel_path}")
        print(f"   Figuras encontradas: {[f[2] for f in result['figures']]}")
        for issue in result['issues']:
            print(f"   ⚠️  Línea {issue['line']}: Figura {issue['found']} (esperado: {issue['expected']})")
            print(f"       {issue['text'][:60]}...")
        print()

    # Mostrar archivos correctos
    print(f"\n✅ ARCHIVOS CORRECTOS ({len(files_ok)}):")
    for rel_path, count in files_ok:
        print(f"   {rel_path} ({count} figuras)")

    print("\n" + "=" * 70)
    print(f"Total: {len(files_with_issues)} archivos con problemas, {len(files_ok)} correctos")
    print("=" * 70)

    return files_with_issues


if __name__ == "__main__":
    main()
