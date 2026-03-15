#!/usr/bin/env python3
"""
enrich_markdown.py — Reemplaza placeholders de imagen en Markdown con imágenes reales.

Orquestador principal que combina parse_placeholders, prompt_builder y generate_image
para transformar una lectura didáctica con placeholders en una lectura con imágenes.

Uso:
    python3 enrich_markdown.py --input lectura.md --output lectura-completa.md --image-dir imagenes/
    python3 enrich_markdown.py --input lectura.md --output lectura-completa.md --dry-run
"""

import argparse
import json
import os
import sys
import time

# Importar módulos hermanos
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from parse_placeholders import extract_placeholders
from prompt_builder import build_prompt, classify_image_type, get_config
from generate_image import generate_image


def enrich_markdown(input_path: str, output_path: str, image_dir: str = "imagenes",
                    language: str = "Spanish", model_override: str = None,
                    dry_run: bool = False) -> dict:
    """
    Procesa un Markdown con placeholders y genera las imágenes.

    Args:
        input_path: Ruta al Markdown con placeholders
        output_path: Ruta donde guardar el Markdown enriquecido
        image_dir: Directorio para guardar las imágenes generadas
        language: Idioma para etiquetas en las imágenes
        model_override: Forzar un modelo específico (ignora config por tipo)
        dry_run: Si True, solo muestra qué haría sin generar imágenes

    Returns:
        dict con resumen: total, exitosos, fallidos, costo_estimado, imágenes
    """
    # Paso 1: Extraer placeholders
    placeholders = extract_placeholders(input_path)

    if not placeholders:
        print("No se encontraron placeholders de imagen en el archivo.", file=sys.stderr)
        return {"total": 0, "success": 0, "failed": 0, "images": []}

    print(f"Encontrados {len(placeholders)} placeholders de imagen.", file=sys.stderr)

    # Crear directorio de imágenes
    os.makedirs(image_dir, exist_ok=True)

    # Leer el Markdown original
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.split("\n")

    results = []
    total_cost = 0.0

    # Paso 2: Para cada placeholder, clasificar, enriquecer y generar
    for p in placeholders:
        print(f"\n--- Placeholder [{p['id']}]: {p['description'][:60]}...", file=sys.stderr)

        # Clasificar tipo
        img_type = classify_image_type(p["description"])
        config = get_config(img_type)
        print(f"    Tipo: {img_type} | Modelo: {config['model']} | Aspecto: {config['aspect']}", file=sys.stderr)

        # Construir prompt enriquecido
        enriched_prompt = build_prompt(p["description"], img_type, language)

        # Determinar modelo
        model = model_override if model_override else config["model"]

        # Ruta de la imagen
        image_filename = p["suggested_filename"]
        image_path = os.path.join(image_dir, image_filename)

        if dry_run:
            print(f"    [DRY RUN] Generaría: {image_path}", file=sys.stderr)
            print(f"    Prompt: {enriched_prompt[:100]}...", file=sys.stderr)
            results.append({
                "placeholder_id": p["id"],
                "type": img_type,
                "filename": image_filename,
                "prompt": enriched_prompt,
                "config": config,
                "status": "dry_run"
            })
            continue

        # Generar imagen
        print(f"    Generando imagen...", file=sys.stderr)
        start = time.time()

        gen_result = generate_image(
            prompt=enriched_prompt,
            model=model,
            aspect_ratio=config.get("aspect"),
            size=config.get("size", "1K"),
            use_search=config.get("search", False),
            thinking_level=config.get("thinking", "none"),
            output_path=image_path,
        )

        elapsed = time.time() - start

        if gen_result["success"]:
            print(f"    Generada en {elapsed:.1f}s: {image_path}", file=sys.stderr)

            # Estimar costo (aproximado)
            cost = 0.07 if model in ["flash", "nb2"] else 0.13
            total_cost += cost

            results.append({
                "placeholder_id": p["id"],
                "type": img_type,
                "filename": image_filename,
                "path": image_path,
                "status": "success",
                "time_seconds": round(elapsed, 1),
                "cost_estimate": cost,
            })
        else:
            print(f"    ERROR: {gen_result['error']}", file=sys.stderr)
            results.append({
                "placeholder_id": p["id"],
                "type": img_type,
                "filename": image_filename,
                "status": "failed",
                "error": gen_result["error"],
            })

    # Paso 3: Reemplazar placeholders en el Markdown
    if not dry_run:
        new_content = _replace_placeholders(content, placeholders, results, image_dir)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"\nMarkdown enriquecido guardado en: {output_path}", file=sys.stderr)

    # Resumen
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    summary = {
        "total": len(placeholders),
        "success": success_count,
        "failed": failed_count,
        "cost_estimate": round(total_cost, 2),
        "output_path": output_path,
        "image_dir": image_dir,
        "images": results,
    }

    print(f"\nResumen: {success_count}/{len(placeholders)} exitosas, ~${total_cost:.2f}", file=sys.stderr)

    return summary


def _replace_placeholders(content: str, placeholders: list, results: list, image_dir: str) -> str:
    """Reemplaza los bloques de placeholder con referencias a las imágenes generadas."""
    lines = content.split("\n")

    # Procesar en orden inverso para no alterar los índices de línea
    for p, r in sorted(zip(placeholders, results), key=lambda x: x[0]["line_start"], reverse=True):
        if r["status"] != "success":
            continue

        start = p["line_start"] - 1  # Convertir a 0-indexed
        end = p["line_end"]

        # Construir el reemplazo: imagen con caption
        image_ref = f"![{p['description']}]({os.path.join(image_dir, r['filename'])})"
        caption = f"*{p['description']}*"

        replacement = [image_ref, "", caption]

        # Reemplazar las líneas del placeholder
        lines[start:end] = replacement

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Enriquece un Markdown reemplazando placeholders con imágenes generadas por IA"
    )
    parser.add_argument("--input", "-i", required=True, help="Archivo Markdown de entrada")
    parser.add_argument("--output", "-o", required=True, help="Archivo Markdown de salida")
    parser.add_argument("--image-dir", "-d", default="imagenes",
                        help="Directorio para guardar las imágenes (default: imagenes/)")
    parser.add_argument("--language", "-l", default="Spanish",
                        help="Idioma para etiquetas en las imágenes (default: Spanish)")
    parser.add_argument("--model", "-m", default=None,
                        choices=["flash", "pro", "nb2", "nb-pro"],
                        help="Forzar modelo específico para todas las imágenes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar qué haría sin generar imágenes")

    args = parser.parse_args()

    summary = enrich_markdown(
        input_path=args.input,
        output_path=args.output,
        image_dir=args.image_dir,
        language=args.language,
        model_override=args.model,
        dry_run=args.dry_run,
    )

    # Imprimir resumen JSON a stdout
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
