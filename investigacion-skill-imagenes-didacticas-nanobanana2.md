# Investigación: Skill de Generación de Imágenes Didácticas con Nano Banana 2

**Fecha:** 14 de marzo de 2026
**Autor:** Claude (investigación para Salvador Hinojosa)

---

## 1. Estado actual del ecosistema

### 1.1 ¿Qué es Nano Banana 2?

Nano Banana 2 es el nombre comercial del modelo **Gemini 3.1 Flash Image Preview** (`gemini-3.1-flash-image-preview`), lanzado por Google el 26 de febrero de 2026. Combina la calidad profesional de Nano Banana Pro con la velocidad de Flash. Sus capacidades clave para uso didáctico son:

- **Resoluciones de 512px a 4K** con múltiples aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4, etc.)
- **Renderizado preciso de texto** dentro de imágenes — crucial para diagramas etiquetados y fórmulas
- **Localización in-image** — puede generar o traducir texto en múltiples idiomas directamente
- **Consistencia de personajes** — hasta 5 sujetos y 14 objetos consistentes entre imágenes
- **Grounding con Google Search** — puede generar imágenes basadas en datos reales y actuales
- **Image Search** — puede buscar y referenciar imágenes existentes para composición

### 1.2 Modelos disponibles

| Modelo | ID | Costo aprox. | Mejor para |
|--------|----|-------------|------------|
| Nano Banana 2 (Flash) | `gemini-3.1-flash-image-preview` | ~$0.04-0.15/img | Volumen alto, velocidad |
| Nano Banana Pro | `gemini-3-pro-image-preview` | ~$0.13-0.30/img | Calidad máxima, texto perfecto |
| Nano Banana (original) | `gemini-2.5-flash-image` | ~$0.04/img | Compatibilidad legacy |

### 1.3 Skills existentes que debemos considerar

**nano-banana (skill actual):** Wrapper ligero sobre la extensión de Gemini CLI. Usa comandos como `gemini --yolo "/generate 'prompt'"`. Orientado a generación genérica de imágenes.

**lectura-didactica:** Genera lecturas en Markdown con **placeholders** de imagen (`📷 [Placeholder]`) pero NO genera las imágenes reales. Aquí está la oportunidad.

**nano-banana-2-skill (kingbootoshi, GitHub):** CLI independiente en Bun/TypeScript que llama directamente a la API de Gemini. Soporta transparencia, reference images, style transfer. También es plugin de Claude Code.

---

## 2. Tres enfoques posibles para el skill

### Enfoque A: Extensión del skill nano-banana existente (via Gemini CLI)

**Cómo funciona:** Usa `gemini --yolo "/comando 'prompt'"` como intermediario.

```
Usuario → Skill → Gemini CLI → Extensión nanobanana → API Gemini → Imagen
```

**Ventajas:**
- Ya está instalado y probado en el entorno
- No requiere manejar API keys directamente
- Los comandos `/diagram`, `/icon`, `/pattern` ya están optimizados
- Menor complejidad de implementación

**Desventajas:**
- Dependencia de la extensión Gemini CLI (versión 1.0.12, modelo default `gemini-2.5-flash-image`)
- No aprovecha Nano Banana 2 por default (habría que setear `NANOBANANA_MODEL`)
- No tiene control fino sobre parámetros de la API (thinking, search grounding)
- La extensión genera archivos en `./nanobanana-output/` — hay que moverlos

**Implementación:**
```yaml
# SKILL.md
name: imagen-didactica
allowed-tools: Bash(gemini:*)
```
El skill construiría prompts especializados para contenido educativo y los enviaría via `gemini --yolo`.

---

### Enfoque B: Llamada directa a la API de Gemini (Python/curl)

**Cómo funciona:** Script Python o llamadas curl directas al endpoint de Gemini.

```
Usuario → Skill → Script Python → API REST Gemini → Base64 → Archivo PNG
```

**Ventajas:**
- Control total sobre todos los parámetros: modelo, resolución, thinking, search grounding
- Puede usar Nano Banana 2 (`gemini-3.1-flash-image-preview`) nativamente
- Puede activar `imageSearch` para buscar imágenes de referencia académica
- Puede usar `thinkingConfig` para generación más precisa de diagramas complejos
- No depende de extensiones externas
- Control total del formato de salida y naming

**Desventajas:**
- Requiere manejar la API key (GEMINI_API_KEY)
- Hay que implementar el manejo de base64 → archivo
- Más código para mantener
- Hay que manejar errores de API, rate limits, etc.

**Implementación ejemplo:**

```python
# scripts/generate_didactic_image.py
import requests
import base64
import json
import os
import sys

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-image-preview"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

def generate_image(prompt, aspect_ratio="16:9", size="2K", output_path="output.png",
                   use_search=False, thinking_level="high"):
    """Genera una imagen didáctica usando Nano Banana 2."""

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": size
            }
        },
        "thinkingConfig": {
            "thinkingLevel": thinking_level,
            "includeThoughts": False
        }
    }

    # Activar búsqueda de imágenes para referencia visual
    if use_search:
        payload["tools"] = [{
            "googleSearch": {
                "searchTypes": {
                    "webSearch": {},
                    "imageSearch": {}
                }
            }
        }]

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    response = requests.post(ENDPOINT, json=payload, headers=headers)
    result = response.json()

    # Extraer imagen base64 de la respuesta
    for part in result["candidates"][0]["content"]["parts"]:
        if "inline_data" in part:
            img_data = base64.b64decode(part["inline_data"]["data"])
            with open(output_path, "wb") as f:
                f.write(img_data)
            return output_path

    return None
```

---

### Enfoque C: Híbrido — Script Python dedicado + integración con lectura-didáctica

**Cómo funciona:** Un skill que reemplaza los placeholders de `lectura-didactica` con imágenes reales generadas por Nano Banana 2.

```
lectura-didactica genera MD con placeholders
        ↓
imagen-didactica parsea placeholders
        ↓
Para cada placeholder → construye prompt didáctico enriquecido
        ↓
Llama API Gemini con parámetros optimizados por tipo
        ↓
Inserta imágenes en el Markdown final
```

**Ventajas:**
- Integración natural con el flujo existente de lectura-didáctica
- Cada tipo de imagen (diagrama, gráfica, esquema) se optimiza diferente
- Puede generar un paquete completo: lectura + imágenes
- El prompt engineering se especializa por tipo de contenido educativo

**Desventajas:**
- Más complejo de implementar
- Depende de que lectura-didáctica mantenga su formato de placeholders
- Mayor costo por lectura (múltiples llamadas a la API)

---

## 3. Recomendación: Enfoque C (Híbrido) con fallback al B

Este es el enfoque más potente para un contexto educativo. Aquí está el diseño propuesto:

### 3.1 Arquitectura del skill

```
imagen-didactica/
├── SKILL.md                          # Definición principal
├── scripts/
│   ├── generate_image.py             # Motor de generación (API directa)
│   ├── parse_placeholders.py         # Extrae placeholders del Markdown
│   ├── prompt_builder.py             # Construye prompts especializados por tipo
│   └── enrich_markdown.py            # Inserta imágenes en el MD final
├── references/
│   ├── prompt_templates.md           # Templates de prompts por tipo de imagen
│   └── educational_styles.md         # Guía de estilos didácticos
└── assets/
    └── example_outputs/              # Ejemplos de referencia
```

### 3.2 Tipos de imagen didáctica y su optimización

| Tipo | Modelo recomendado | Thinking | Search | Aspect Ratio | Resolución |
|------|--------------------|----------|--------|-------------|------------|
| Diagrama de flujo | NB2 Flash | high | no | 4:3 | 2K |
| Gráfica matemática | NB Pro | high | no | 4:3 | 2K |
| Diagrama de bloques | NB2 Flash | high | no | 16:9 | 2K |
| Foto de caso real | NB2 Flash | minimal | sí | 16:9 | 1K |
| Esquema de sistema | NB2 Flash | high | no | 16:9 | 2K |
| Ilustración conceptual | NB2 Flash | minimal | no | 1:1 | 1K |
| Infografía con texto | NB Pro | high | no | 9:16 | 4K |

### 3.3 Estrategia de prompts para contenido educativo

El componente más crítico es el **prompt engineering** especializado. Un prompt genérico como "diagrama del ciclo de Carnot" produce resultados mediocres. El skill debe enriquecer automáticamente:

```
Placeholder original:
"Diagrama del ciclo de Carnot en coordenadas P-V y T-S"

Prompt enriquecido automáticamente:
"Educational technical diagram of the Carnot cycle.
Left side: P-V diagram with clearly labeled axes (Pressure vs Volume),
showing four processes: isothermal expansion (1→2), adiabatic expansion (2→3),
isothermal compression (3→4), adiabatic compression (4→1).
Right side: T-S diagram (Temperature vs Entropy) showing the same four processes.
Use clean lines, professional engineering style, labeled arrows showing
direction of processes, shaded area representing net work output.
Colors: blue for cold reservoir processes, red for hot reservoir processes.
Include clear labels in Spanish: 'Expansión isotérmica', 'Expansión adiabática',
'Compresión isotérmica', 'Compresión adiabática'.
White background, suitable for academic textbook. No decorative elements."
```

### 3.4 Flujo de uso propuesto

**Modo 1 — Standalone (generación individual):**
```
Usuario: "Genera una imagen didáctica de un diagrama de Bode para un sistema de segundo orden"
→ Skill detecta tipo (diagrama técnico)
→ Construye prompt enriquecido con contexto académico
→ Genera con NB2 Flash, thinking=high, 2K, 4:3
→ Entrega imagen
```

**Modo 2 — Integrado con lectura-didáctica:**
```
Usuario: "Genera las imágenes para esta lectura" + archivo .md con placeholders
→ Skill parsea todos los placeholders 📷
→ Para cada uno: clasifica tipo → construye prompt → genera imagen
→ Reemplaza placeholders con ![alt](ruta_imagen.png)
→ Entrega Markdown enriquecido + carpeta de imágenes
```

### 3.5 SKILL.md propuesto (borrador)

```yaml
---
name: imagen-didactica
description: >
  Genera imágenes didácticas de alta calidad para material educativo universitario
  usando Nano Banana 2 (Gemini 3.1 Flash Image). Especializado en diagramas técnicos,
  gráficas, esquemas de sistemas, ilustraciones conceptuales e infografías para
  ingeniería. Úsalo cuando el usuario pida "generar imágenes para la lectura",
  "crear un diagrama de...", "ilustrar este concepto", o cuando quiera reemplazar
  los placeholders de imagen de una lectura didáctica con imágenes reales generadas
  por IA. También se activa para cualquier solicitud de imagen con contexto académico
  o educativo.
allowed-tools: Bash(python3:*), Read, Write, Glob
---
```

---

## 4. Consideraciones técnicas importantes

### 4.1 API Key
El skill necesita `GEMINI_API_KEY`. La API tiene un **tier gratuito** en Google AI Studio sin tarjeta de crédito. Para uso educativo el volumen debería ser manejable.

### 4.2 Limitaciones de Nano Banana 2
- Todas las imágenes incluyen **watermark SynthID** (invisible al ojo humano)
- El texto en español renderiza bien, pero el inglés es más confiable para labels complejos
- Los diagramas muy técnicos (circuitos, fórmulas) pueden necesitar NB Pro ($2x costo)
- Rate limits aplican — para lotes grandes considerar Batch API (24h turnaround)

### 4.3 Alternativas para diagramas de alta precisión
Para diagramas que necesitan precisión milimétrica (circuitos electrónicos, grafos de estado exactos), considerar un enfoque híbrido:
- Usar Nano Banana 2 para ilustraciones conceptuales y fotos de casos reales
- Usar el skill **excalidraw-diagram** para diagramas que requieren precisión geométrica
- El skill podría decidir automáticamente qué motor usar según el tipo

### 4.4 Costos estimados por lectura
Una lectura típica tiene 3-5 placeholders de imagen:
- Con NB2 Flash a 2K: ~$0.10 × 4 imágenes = **~$0.40 por lectura**
- Con NB Pro a 2K: ~$0.20 × 4 imágenes = **~$0.80 por lectura**
- Mixto (recomendado): **~$0.50 por lectura**

---

## 5. Pasos siguientes para implementar

1. **Decidir enfoque** — ¿Enfoque C (completo) o empezar con B (standalone)?
2. **Crear el skill con skill-creator** — Usar el skill-creator existente para scaffolding, evals y optimización de description
3. **Implementar `generate_image.py`** — Motor core con llamada directa a API
4. **Implementar `prompt_builder.py`** — Templates de prompts por tipo educativo
5. **Implementar `parse_placeholders.py`** — Parser de Markdown para modo integrado
6. **Crear evals** — Test cases con diferentes tipos de imágenes didácticas
7. **Iterar** — Usar el ciclo eval/grade/improve del skill-creator
8. **Optimizar description** — Para que trigger correctamente en contextos educativos

---

## Fuentes consultadas

- [Nano Banana image generation - Google AI for Developers](https://ai.google.dev/gemini-api/docs/image-generation)
- [Build with Nano Banana 2 - Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/build-with-nano-banana-2/)
- [Nano Banana 2: Google's latest AI image generation model](https://blog.google/innovation-and-ai/technology/ai/nano-banana-2/)
- [Google launches Nano Banana 2 - TechCrunch](https://techcrunch.com/2026/02/26/google-launches-nano-banana-2-model-with-faster-image-generation/)
- [kingbootoshi/nano-banana-2-skill - GitHub](https://github.com/kingbootoshi/nano-banana-2-skill)
- [gemini-cli-extensions/nanobanana - GitHub](https://github.com/gemini-cli-extensions/nanobanana)
- [Nanobanana DeepWiki](https://deepwiki.com/gemini-cli-extensions/nanobanana)
- [kkoppenhaver/cc-nano-banana - GitHub](https://github.com/kkoppenhaver/cc-nano-banana)
