#!/usr/bin/env python3
"""
generate_image.py — Motor de generación de imágenes didácticas via API de Gemini.

Llama directamente al endpoint de Gemini para generar imágenes,
con soporte para múltiples modelos, resoluciones y modos de thinking.

Uso:
    python3 generate_image.py --prompt "..." --output imagen.png
    python3 generate_image.py --prompt "..." --type diagram --aspect 4:3 --size 2K
    python3 generate_image.py --prompt "..." --model pro --search --output foto.png
"""

import argparse
import base64
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' no está instalado. Ejecuta: pip install requests --break-system-packages", file=sys.stderr)
    sys.exit(1)


# --- Configuración de modelos ---
MODELS = {
    "flash": "gemini-2.5-flash-image",
    "nb2": "gemini-3.1-flash-image-preview",
    "pro": "gemini-3-pro-image-preview",
    "nb-pro": "gemini-3-pro-image-preview",
}

# Mapeo de tamaño a resolución aproximada (el modelo decide la exacta)
SIZES = {
    "512": "512x512",
    "1K": "1024x1024",
    "2K": "2048x2048",
    "4K": "4096x4096",
}

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def get_api_key():
    """Obtiene la API key de variables de entorno o archivo .env."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    # Buscar en archivos .env comunes
    env_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.expanduser("~/.nano-banana/.env"),
        os.path.expanduser("~/.gemini/.env"),
    ]
    for path in env_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def generate_image(prompt, model="flash", aspect_ratio=None, size="1K",
                   use_search=False, thinking_level="none",
                   reference_image=None, output_path="output.png"):
    """
    Genera una imagen usando la API de Gemini.

    Args:
        prompt: El prompt descriptivo para la imagen
        model: 'flash' o 'pro'
        aspect_ratio: '1:1', '16:9', '4:3', etc. (None para default del modelo)
        size: '512', '1K', '2K', '4K'
        use_search: Activar Google Search grounding
        thinking_level: 'none', 'minimal', 'high'
        reference_image: Ruta a imagen de referencia (opcional)
        output_path: Ruta donde guardar la imagen generada

    Returns:
        dict con 'success', 'path', 'model', 'prompt_used', 'error'
    """
    api_key = get_api_key()
    if not api_key:
        return {
            "success": False,
            "error": "GEMINI_API_KEY no encontrada. Establécela con: export GEMINI_API_KEY='tu-clave'",
            "path": None
        }

    model_id = MODELS.get(model, model)  # Permite pasar model ID directo
    endpoint = f"{API_BASE}/{model_id}:generateContent?key={api_key}"

    # Construir las partes del contenido
    parts = []

    # Agregar imagen de referencia si se proporciona
    if reference_image and os.path.exists(reference_image):
        with open(reference_image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(reference_image)[1].lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime = mime_map.get(ext, "image/png")
        parts.append({"inlineData": {"mimeType": mime, "data": img_b64}})

    parts.append({"text": prompt})

    # Construir payload
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        }
    }

    # Herramientas opcionales
    tools = []
    if use_search:
        tools.append({"googleSearch": {}})
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}

    # Hacer la llamada
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=120)

        if response.status_code != 200:
            error_detail = response.text[:500]
            return {
                "success": False,
                "error": f"API error {response.status_code}: {error_detail}",
                "path": None
            }

        result = response.json()

        # Extraer imagen de la respuesta
        if "candidates" not in result or not result["candidates"]:
            return {
                "success": False,
                "error": f"No se generó imagen. Respuesta: {json.dumps(result)[:300]}",
                "path": None
            }

        text_response = ""
        image_found = False

        for part in result["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                img_data = base64.b64decode(part["inlineData"]["data"])

                # Crear directorio si no existe
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

                with open(output_path, "wb") as f:
                    f.write(img_data)
                image_found = True

            elif "text" in part:
                text_response += part["text"]

        if not image_found:
            return {
                "success": False,
                "error": f"La respuesta no incluyó imagen. Texto: {text_response[:300]}",
                "path": None
            }

        return {
            "success": True,
            "path": os.path.abspath(output_path),
            "model": model_id,
            "prompt_used": prompt[:200],
            "text_response": text_response,
            "size_bytes": os.path.getsize(output_path)
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout: la API tardó más de 120s", "path": None}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": f"Error de conexión: {e}", "path": None}
    except Exception as e:
        return {"success": False, "error": f"Error inesperado: {e}", "path": None}


def main():
    parser = argparse.ArgumentParser(
        description="Genera imágenes didácticas usando la API de Gemini (Nano Banana 2)"
    )
    parser.add_argument("--prompt", required=True, help="Prompt descriptivo para la imagen")
    parser.add_argument("--output", "-o", default="output.png", help="Ruta del archivo de salida")
    parser.add_argument("--model", "-m", default="flash", choices=["flash", "pro", "nb2", "nb-pro"],
                        help="Modelo a usar (default: flash)")
    parser.add_argument("--aspect", "-a", default=None, help="Aspect ratio (ej: 16:9, 4:3, 1:1)")
    parser.add_argument("--size", "-s", default="1K", choices=["512", "1K", "2K", "4K"],
                        help="Resolución (default: 1K)")
    parser.add_argument("--search", action="store_true", help="Activar Google Search grounding")
    parser.add_argument("--thinking", default="none", choices=["none", "minimal", "high"],
                        help="Nivel de thinking (default: none)")
    parser.add_argument("--ref", "-r", default=None, help="Imagen de referencia")
    parser.add_argument("--type", "-t", default=None,
                        choices=["diagram", "graph", "schematic", "illustration", "infographic", "photo"],
                        help="Tipo de imagen didáctica (opcional, para prompt enrichment automático)")

    args = parser.parse_args()

    # Si se especifica tipo, enriquecer el prompt
    if args.type:
        from prompt_builder import build_prompt
        enriched = build_prompt(args.prompt, args.type)
        print(f"Prompt enriquecido ({args.type}):", file=sys.stderr)
        print(f"  {enriched[:150]}...", file=sys.stderr)
    else:
        enriched = args.prompt

    print(f"Generando imagen con modelo '{args.model}', tamaño {args.size}...", file=sys.stderr)
    start = time.time()

    result = generate_image(
        prompt=enriched,
        model=args.model,
        aspect_ratio=args.aspect,
        size=args.size,
        use_search=args.search,
        thinking_level=args.thinking,
        reference_image=args.ref,
        output_path=args.output
    )

    elapsed = time.time() - start

    if result["success"]:
        print(f"Imagen generada en {elapsed:.1f}s: {result['path']}", file=sys.stderr)
        print(f"Tamaño: {result['size_bytes']} bytes", file=sys.stderr)
        # Imprimir JSON a stdout para que otros scripts puedan parsearlo
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
