---
name: imagen-didactica
description: >
  Genera imágenes didácticas de alta calidad para material educativo universitario
  usando la API de Gemini (Nano Banana 2 / Flash Image). Especializado en diagramas
  técnicos, gráficas, esquemas de sistemas, ilustraciones conceptuales e infografías
  para ingeniería. Úsalo cuando el usuario pida "generar imágenes para la lectura",
  "crear un diagrama de...", "ilustrar este concepto", "generar las imágenes del
  material", o cuando quiera reemplazar los placeholders de imagen (📷) de una
  lectura didáctica con imágenes reales generadas por IA. También se activa para
  cualquier solicitud de imagen con contexto académico, educativo o de clase,
  incluso si el usuario no dice explícitamente "didáctica". Si el usuario tiene un
  archivo Markdown con placeholders de imagen y pide generar las imágenes, este
  skill es el indicado.
---

# Skill: Generación de Imágenes Didácticas con Nano Banana 2

Genera imágenes educativas profesionales llamando directamente a la API de Gemini.
Este skill tiene dos modos de operación: generación individual y enriquecimiento
de lecturas (reemplazo de placeholders `📷` en Markdown).

## Prerrequisitos

Antes de generar imágenes, verifica que la API key esté disponible:

```bash
[ -n "$GEMINI_API_KEY" ] && echo "OK" || echo "Falta GEMINI_API_KEY"
```

Si no está configurada, pide al usuario que la establezca:
```bash
export GEMINI_API_KEY="su-clave-aquí"
```

La clave se obtiene gratis en [Google AI Studio](https://aistudio.google.com/apikey).

## Modos de operación

### Modo 1: Generación individual

Cuando el usuario pide una imagen específica ("genera un diagrama de...",
"crea una ilustración de..."), usa el script `generate_image.py` directamente.

### Modo 2: Enriquecimiento de lectura

Cuando el usuario tiene un Markdown con placeholders `📷` (generado por
lectura-didáctica u otro medio), usa `enrich_markdown.py` para reemplazar
todos los placeholders con imágenes reales.

## Flujo de trabajo

### Paso 1: Clasificar el tipo de imagen

Lee la referencia `references/prompt_templates.md` para entender los tipos
de imagen didáctica disponibles y cómo optimizar cada uno. Cada tipo tiene
un modelo recomendado, aspect ratio y nivel de thinking diferente.

Los tipos principales son:

| Tipo | Cuándo usarlo |
|------|---------------|
| `diagram` | Diagramas de flujo, bloques, estado, secuencia |
| `graph` | Gráficas matemáticas, coordenadas, funciones |
| `schematic` | Esquemas de sistemas, circuitos, arquitecturas |
| `illustration` | Conceptos abstractos, analogías visuales |
| `infographic` | Resúmenes visuales con texto y datos |
| `photo` | Fotos realistas de casos, equipos, fenómenos |

### Paso 2: Construir el prompt enriquecido

El prompt es el componente más crítico. Un prompt genérico como "diagrama PID"
produce resultados mediocres. El script `prompt_builder.py` transforma
descripciones breves en prompts detallados optimizados para generación educativa.

El enriquecimiento automático incluye:
- Especificación de estilo visual (clean lines, labeled, white background)
- Indicación de idioma para etiquetas (español por default)
- Instrucciones de composición (ejes, leyendas, flechas direccionales)
- Indicación "suitable for academic textbook, no decorative elements"
- Contraste alto para legibilidad en proyección

### Paso 3: Generar la imagen

```bash
python3 <skill-path>/scripts/generate_image.py \
  --prompt "El prompt enriquecido completo" \
  --type diagram \
  --output ruta/imagen.png \
  --aspect 4:3 \
  --size 2K
```

Parámetros opcionales:
- `--model flash` (default) o `--model pro` para calidad máxima
- `--search` para activar Google Search grounding (fotos de casos reales)
- `--thinking high` (default para diagramas) o `--thinking minimal`

### Paso 4: Para enriquecimiento de Markdown completo

```bash
python3 <skill-path>/scripts/enrich_markdown.py \
  --input lectura.md \
  --output lectura-con-imagenes.md \
  --image-dir imagenes/
```

Esto parsea todos los placeholders `📷`, construye prompts enriquecidos para
cada uno, genera las imágenes y produce un nuevo Markdown con las referencias
a las imágenes insertadas.

## Directrices para prompts educativos

Estas reglas hacen la diferencia entre una imagen útil y una decorativa:

1. **Siempre especifica qué debe etiquetar la imagen.** No digas solo "diagrama
   de Carnot" — di exactamente qué ejes, qué procesos, qué flechas mostrar.

2. **Pide fondo blanco** para material impreso o proyectado. Fondo oscuro solo
   si es para presentaciones con tema oscuro.

3. **Indica el idioma de las etiquetas.** Por default usa español para todo el
   texto dentro de la imagen. Si hay términos técnicos que se usan en inglés
   en la disciplina, mézclalos (ej: "Bode plot" pero "Frecuencia (Hz)").

4. **Especifica "no decorative elements, no watermarks, no borders"** para
   mantener la imagen limpia y profesional.

5. **Para diagramas con texto**, usa el modelo Pro — renderiza texto con mayor
   fidelidad. Para ilustraciones conceptuales sin texto, Flash es suficiente.

6. **Pide alto contraste** — las imágenes se proyectan en salones con luz
   ambiente y se imprimen a veces en blanco y negro.

## Estructura del placeholder esperada

El skill reconoce este formato de placeholder (usado por lectura-didactica):

```markdown
> 📷 **[Placeholder de imagen/diagrama]**
> *Descripción detallada de lo que debería mostrar la imagen.*
```

La descripción entre asteriscos es lo que se usa como base para el prompt.

## Costos de referencia

| Modelo | Resolución | Costo aprox. |
|--------|-----------|-------------|
| Flash (NB2) | 1K | ~$0.07 |
| Flash (NB2) | 2K | ~$0.10 |
| Pro | 1K | ~$0.13 |
| Pro | 2K | ~$0.20 |

Una lectura típica con 4 imágenes a 2K cuesta ~$0.40-0.80.
