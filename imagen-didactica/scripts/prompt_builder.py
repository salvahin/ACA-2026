#!/usr/bin/env python3
"""
prompt_builder.py — Construye prompts enriquecidos para imágenes didácticas.

Transforma descripciones breves de placeholders en prompts detallados
optimizados para generación educativa de alta calidad.

Uso como módulo:
    from prompt_builder import build_prompt, classify_image_type
    prompt = build_prompt("Ciclo de Carnot en coordenadas P-V", "diagram")

Uso desde CLI:
    python3 prompt_builder.py --description "Ciclo de Carnot" --type diagram
    python3 prompt_builder.py --description "Foto de motor eléctrico" --auto-classify
"""

import argparse
import json
import re
import sys

# --- Templates por tipo de imagen ---

TEMPLATES = {
    "diagram": (
        "Educational technical diagram: {description}. "
        "Style: clean vector-style lines, professional engineering quality. "
        "Use clearly labeled components with arrows showing flow direction. "
        "All text labels in {language}. Use standard technical notation where applicable. "
        "White background, high contrast, suitable for academic textbook. "
        "No decorative elements, no shadows, no 3D effects, no watermarks. "
        "Color coding: use distinct, high-contrast colors (blue, red, green, orange) "
        "for different components or processes. Include a legend if more than 3 colors are used."
    ),
    "graph": (
        "Educational mathematical graph: {description}. "
        "Style: precise plotted lines on clearly labeled axes with grid lines visible but subtle (light gray). "
        "Axis labels in {language}. Use different line styles (solid, dashed, dotted) for multiple curves. "
        "Include labels or legend for each curve. Mark critical points (intersections, maxima, minima, "
        "asymptotes) with dots and annotations. "
        "White background, black axes, high contrast suitable for projection and printing. "
        "No decorative elements, no 3D effects. Professional engineering textbook quality."
    ),
    "schematic": (
        "Educational technical schematic: {description}. "
        "Style: engineering standard symbols and notation. "
        "Use standard component symbols where applicable. All connections clearly drawn with no ambiguity. "
        "Component labels and values clearly marked. Signal flow indicated with arrows. "
        "Input on the left, output on the right where applicable. "
        "Text labels in {language} except for universal technical symbols. "
        "White background, clean lines, professional drafting quality. "
        "No decorative elements. Suitable for engineering course material."
    ),
    "illustration": (
        "Educational conceptual illustration: {description}. "
        "Style: modern flat illustration, clean and professional. "
        "Use a clear visual metaphor that helps university students understand the concept. "
        "Minimal text overlay — let the image communicate visually. "
        "If labels are necessary, use {language}. "
        "Color palette: harmonious, professional, suitable for university-level course material. "
        "High contrast for projection in classrooms with ambient light. "
        "White or very light background. No excessive detail, no cartoonish style. "
        "No decorative borders, no watermarks."
    ),
    "infographic": (
        "Educational infographic: {description}. "
        "Style: modern data visualization with clear visual hierarchy. "
        "Title at top in large, bold text in {language}. "
        "Organize information in logical sections flowing top to bottom. "
        "Use icons, simple charts, and labeled callouts. All text in {language}. "
        "Color palette: professional, use 3-4 main colors consistently. "
        "Each section should be visually distinct with generous white space. "
        "High contrast, suitable for printing or projection. No decorative noise."
    ),
    "photo": (
        "Realistic educational photograph: {description}. "
        "Style: professional photography quality, well-lit, sharp focus. "
        "Show the subject clearly with good composition and context that helps "
        "students understand scale and real-world application. "
        "Even studio-like lighting or natural daylight. "
        "No artistic filters, no extreme color grading. "
        "Suitable as a reference image in academic material."
    ),
}

# Configuración recomendada por tipo
TYPE_CONFIG = {
    "diagram":      {"model": "nb2", "thinking": "none", "aspect": "4:3",  "size": "2K", "search": False},
    "graph":        {"model": "pro",  "thinking": "none", "aspect": "4:3",  "size": "2K", "search": False},
    "schematic":    {"model": "pro",  "thinking": "none", "aspect": "16:9", "size": "2K", "search": False},
    "illustration": {"model": "nb2", "thinking": "none", "aspect": "16:9", "size": "1K", "search": False},
    "infographic":  {"model": "pro",  "thinking": "none", "aspect": "9:16", "size": "4K", "search": False},
    "photo":        {"model": "nb2", "thinking": "none", "aspect": "16:9", "size": "1K", "search": False},
}

# Palabras clave para auto-clasificación
CLASSIFICATION_KEYWORDS = {
    "diagram": [
        "diagrama", "diagram", "flujo", "flowchart", "bloques", "block diagram",
        "estado", "state", "secuencia", "sequence", "uml", "decisión", "árbol",
        "tree", "ciclo", "cycle", "retroalimentación", "feedback", "lazo",
    ],
    "graph": [
        "gráfica", "graph", "plot", "coordenadas", "ejes", "axes", "función",
        "function", "bode", "nyquist", "distribución", "distribution", "señal",
        "signal", "espectro", "spectrum", "histograma", "histogram", "curva",
        "curve", "parábola", "exponencial", "logarítmica", "senoidal",
    ],
    "schematic": [
        "esquema", "schematic", "circuito", "circuit", "arquitectura", "architecture",
        "sistema", "system", "red", "network", "topología", "topology", "componentes",
        "resistencia", "capacitor", "transistor", "amplificador", "filtro",
    ],
    "illustration": [
        "ilustración", "illustration", "concepto", "concept", "analogía", "analogy",
        "metáfora", "principio", "principle", "fenómeno", "phenomenon", "visualización",
        "representación", "idea", "abstracto",
    ],
    "infographic": [
        "infografía", "infographic", "resumen", "summary", "comparación", "comparison",
        "línea de tiempo", "timeline", "estadísticas", "statistics", "datos", "data",
        "proceso paso a paso", "step by step",
    ],
    "photo": [
        "foto", "photo", "fotografía", "imagen real", "equipo", "equipment",
        "laboratorio", "laboratory", "máquina", "machine", "dispositivo", "device",
        "planta", "plant", "instalación", "facility",
    ],
}


def classify_image_type(description: str) -> str:
    """
    Clasifica automáticamente el tipo de imagen basándose en la descripción.

    Busca palabras clave en la descripción y devuelve el tipo con más coincidencias.
    Si no hay coincidencias claras, devuelve 'illustration' como default seguro.
    """
    description_lower = description.lower()
    scores = {}

    for img_type, keywords in CLASSIFICATION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in description_lower)
        if score > 0:
            scores[img_type] = score

    if not scores:
        return "illustration"

    return max(scores, key=scores.get)


def build_prompt(description: str, image_type: str, language: str = "Spanish") -> str:
    """
    Construye un prompt enriquecido combinando la descripción con el template del tipo.

    Args:
        description: La descripción breve del placeholder
        image_type: Uno de: diagram, graph, schematic, illustration, infographic, photo
        language: Idioma para las etiquetas dentro de la imagen

    Returns:
        Prompt enriquecido listo para enviar a la API
    """
    template = TEMPLATES.get(image_type, TEMPLATES["illustration"])
    desc = description.strip().rstrip(".")
    return template.format(description=desc, language=language)


def get_config(image_type: str) -> dict:
    """Devuelve la configuración recomendada para el tipo de imagen."""
    return TYPE_CONFIG.get(image_type, TYPE_CONFIG["illustration"]).copy()


def main():
    parser = argparse.ArgumentParser(description="Construye prompts enriquecidos para imágenes didácticas")
    parser.add_argument("--description", "-d", required=True, help="Descripción de la imagen")
    parser.add_argument("--type", "-t", default=None,
                        choices=["diagram", "graph", "schematic", "illustration", "infographic", "photo"],
                        help="Tipo de imagen (se auto-clasifica si no se especifica)")
    parser.add_argument("--language", "-l", default="Spanish", help="Idioma para etiquetas")
    parser.add_argument("--auto-classify", action="store_true", help="Auto-clasificar tipo de imagen")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")

    args = parser.parse_args()

    # Determinar tipo
    if args.type:
        image_type = args.type
    elif args.auto_classify or args.type is None:
        image_type = classify_image_type(args.description)
        print(f"Tipo auto-clasificado: {image_type}", file=sys.stderr)
    else:
        image_type = "illustration"

    # Construir prompt
    prompt = build_prompt(args.description, image_type, args.language)
    config = get_config(image_type)

    if args.json:
        output = {
            "type": image_type,
            "prompt": prompt,
            "config": config,
            "original_description": args.description,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"Tipo: {image_type}")
        print(f"Config: {json.dumps(config)}")
        print(f"\nPrompt enriquecido:\n{prompt}")


if __name__ == "__main__":
    main()
