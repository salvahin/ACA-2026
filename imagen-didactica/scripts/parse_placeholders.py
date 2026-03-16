#!/usr/bin/env python3
"""
parse_placeholders.py — Extrae placeholders de imagen de archivos Markdown.

Reconoce el formato estándar de lectura-didáctica:
    > 📷 **[Placeholder de imagen/diagrama]**
    > *Descripción detallada...*

También reconoce variantes comunes:
    > 📷 **[Imagen: título]**
    > *Descripción...*

    ![📷 Placeholder](descripción)

Uso como módulo:
    from parse_placeholders import extract_placeholders
    placeholders = extract_placeholders("lectura.md")

Uso desde CLI:
    python3 parse_placeholders.py lectura.md
    python3 parse_placeholders.py lectura.md --json
"""

import argparse
import json
import os
import re
import sys


def extract_placeholders(markdown_path: str) -> list:
    """
    Extrae todos los placeholders de imagen de un archivo Markdown.

    Returns:
        Lista de dicts con:
        - id: int, índice secuencial (0-based)
        - line_start: int, línea donde empieza el placeholder
        - line_end: int, línea donde termina
        - raw_text: str, el texto completo del bloque del placeholder
        - title: str, el título entre corchetes (si existe)
        - description: str, la descripción extraída (lo más importante)
        - suggested_filename: str, nombre sugerido para la imagen
    """
    if not os.path.exists(markdown_path):
        print(f"ERROR: Archivo no encontrado: {markdown_path}", file=sys.stderr)
        return []

    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    placeholders = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Patrón 1: Bloque con blockquote y emoji 📷
        # > 📷 **[Placeholder de imagen/diagrama]**
        # > *Descripción...*
        if re.search(r'>\s*📷', line):
            block_start = i
            block_lines = [line]

            # Recoger todas las líneas del blockquote
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith(">"):
                block_lines.append(lines[j])
                j += 1

            block_text = "\n".join(block_lines)

            # Extraer título entre corchetes
            title_match = re.search(r'\*\*\[([^\]]+)\]\*\*', block_text)
            title = title_match.group(1) if title_match else ""

            # Extraer descripción: recolectar todas las líneas del blockquote
            # que no son el título, limpiar marcadores de blockquote y asteriscos
            desc_parts = []
            for bline in block_lines[1:]:
                cleaned = bline.strip().lstrip(">").strip()
                if not cleaned:
                    continue
                # Quitar asteriscos de itálica al inicio/final
                cleaned = re.sub(r'^\*\s*', '', cleaned)
                cleaned = re.sub(r'\s*\*$', '', cleaned)
                if cleaned and not re.search(r'\*\*\[', cleaned) and not cleaned.startswith("📷"):
                    desc_parts.append(cleaned)

            description = " ".join(desc_parts).strip()

            # Si no hay descripción separada, usar el título
            if not description:
                description = title

            placeholders.append({
                "id": len(placeholders),
                "line_start": block_start + 1,  # 1-indexed
                "line_end": j,
                "raw_text": block_text,
                "title": title,
                "description": description.strip(),
                "suggested_filename": _slugify(description[:60]) + ".png",
            })

            i = j
            continue

        # Patrón 2: Markdown image con placeholder
        # ![📷 Descripción](placeholder)
        img_match = re.match(r'!\[📷\s*(.+?)\]\((.+?)\)', line.strip())
        if img_match:
            desc = img_match.group(1).strip()
            placeholders.append({
                "id": len(placeholders),
                "line_start": i + 1,
                "line_end": i + 1,
                "raw_text": line,
                "title": desc,
                "description": desc,
                "suggested_filename": _slugify(desc[:60]) + ".png",
            })

        i += 1

    return placeholders


def _slugify(text: str) -> str:
    """Convierte texto a un slug válido para nombre de archivo."""
    # Reemplazar caracteres especiales
    text = text.lower()
    text = re.sub(r'[áàäâ]', 'a', text)
    text = re.sub(r'[éèëê]', 'e', text)
    text = re.sub(r'[íìïî]', 'i', text)
    text = re.sub(r'[óòöô]', 'o', text)
    text = re.sub(r'[úùüû]', 'u', text)
    text = re.sub(r'[ñ]', 'n', text)
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')[:60]


def main():
    parser = argparse.ArgumentParser(description="Extrae placeholders de imagen de archivos Markdown")
    parser.add_argument("file", help="Ruta al archivo Markdown")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")

    args = parser.parse_args()

    placeholders = extract_placeholders(args.file)

    if args.json:
        print(json.dumps(placeholders, ensure_ascii=False, indent=2))
    else:
        if not placeholders:
            print("No se encontraron placeholders de imagen.")
            return

        print(f"Se encontraron {len(placeholders)} placeholder(s):\n")
        for p in placeholders:
            print(f"  [{p['id']}] Línea {p['line_start']}: {p['description'][:80]}")
            print(f"       Archivo sugerido: {p['suggested_filename']}")
            print()


if __name__ == "__main__":
    main()
