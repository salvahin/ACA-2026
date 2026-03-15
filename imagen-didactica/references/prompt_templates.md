# Plantillas de Prompts por Tipo de Imagen Didáctica

Este documento contiene los templates que `prompt_builder.py` usa para enriquecer
las descripciones breves de los placeholders. Cada tipo tiene un template base que
se combina con la descripción específica del usuario.

---

## Tipos de imagen y configuración óptima

### 1. diagram — Diagramas técnicos

**Cuándo:** Flujos, bloques, estados, secuencia, UML, decisiones.

**Config:** model=flash, thinking=high, aspect=4:3, size=2K

**Template base:**
```
Educational technical diagram: {description}.
Style: clean vector-style lines, professional engineering quality.
Use clearly labeled components with arrows showing flow direction.
All text labels in {language}. Use standard technical notation.
White background, high contrast, suitable for academic textbook.
No decorative elements, no shadows, no 3D effects, no watermarks.
Color coding: use distinct, high-contrast colors (blue, red, green, orange)
for different components or processes. Include a legend if more than 3 colors.
```

### 2. graph — Gráficas y funciones matemáticas

**Cuándo:** Coordenadas cartesianas, Bode, Nyquist, distribuciones, señales.

**Config:** model=pro, thinking=high, aspect=4:3, size=2K

**Template base:**
```
Educational mathematical graph: {description}.
Style: precise plotted lines on clearly labeled axes.
X-axis: {x_label if specified, else "x"}. Y-axis: {y_label if specified, else "y"}.
Grid lines visible but subtle (light gray). Axis labels in {language}.
Use different line styles (solid, dashed, dotted) for multiple curves.
Include labels/legend for each curve. Mark critical points (intersections,
maxima, minima, asymptotes) with dots and annotations.
White background, black axes, high contrast for projection.
No decorative elements, no 3D effects. Suitable for engineering textbook.
```

### 3. schematic — Esquemas de sistemas

**Cuándo:** Circuitos, arquitecturas de software, sistemas mecánicos, redes.

**Config:** model=pro, thinking=high, aspect=16:9, size=2K

**Template base:**
```
Educational technical schematic: {description}.
Style: engineering standard symbols and notation.
Use standard component symbols (resistors, capacitors, gates, blocks)
where applicable. All connections clearly drawn with no ambiguity.
Component labels and values clearly marked. Signal flow indicated
with arrows. Input on the left, output on the right.
Text labels in {language} except for standard technical symbols.
White background, clean lines, professional drafting quality.
No decorative elements. Suitable for engineering course material.
```

### 4. illustration — Ilustraciones conceptuales

**Cuándo:** Analogías, conceptos abstractos, fenómenos naturales, principios.

**Config:** model=flash, thinking=minimal, aspect=16:9, size=1K

**Template base:**
```
Educational conceptual illustration: {description}.
Style: modern flat illustration, clean and professional.
Use a clear visual metaphor that helps students understand the concept.
Minimal text — let the image speak. If labels are needed, use {language}.
Color palette: harmonious, professional, suitable for university course.
High contrast for projection in classrooms with ambient light.
White or very light background. No excessive detail.
Should feel approachable and modern, not cartoonish.
No decorative borders, no watermarks.
```

### 5. infographic — Infografías con datos

**Cuándo:** Comparaciones, resúmenes, estadísticas, líneas de tiempo, procesos.

**Config:** model=pro, thinking=high, aspect=9:16 (vertical), size=4K

**Template base:**
```
Educational infographic: {description}.
Style: modern data visualization with clear hierarchy.
Title at top in large, bold text in {language}.
Organize information in logical sections flowing top to bottom.
Use icons, simple charts, and labeled callouts. All text in {language}.
Color palette: professional, use 3-4 main colors consistently.
Each section should be visually distinct. Use high contrast.
Generous white space between sections.
Suitable for printing or projection. No decorative noise.
```

### 6. photo — Fotos realistas

**Cuándo:** Equipos reales, laboratorios, fenómenos físicos, casos de estudio.

**Config:** model=flash, thinking=minimal, aspect=16:9, size=1K, search=true

**Template base:**
```
Realistic educational photograph: {description}.
Style: professional photography quality, well-lit, sharp focus.
Show the subject clearly with good composition.
Lighting: even studio-like lighting or natural daylight.
If showing equipment or processes, include context that helps
students understand scale and application.
No artistic filters, no extreme color grading.
Suitable as a reference image in academic material.
```

---

## Notas sobre idioma

- Por default, `{language}` se reemplaza con "Spanish" para text labels
- Los términos técnicos universales se mantienen en inglés (PID, FFT, HTML, etc.)
- Si el usuario especifica otro idioma, se respeta
- Para materiales bilingües, se puede indicar "Spanish with English technical terms"

## Notas sobre búsqueda (Search Grounding)

Solo activar `--search` para:
- Fotos de equipos/dispositivos reales
- Datos estadísticos actuales
- Casos de estudio recientes
- Fenómenos naturales específicos

NO activar para:
- Diagramas técnicos (la IA debe generarlos, no buscarlos)
- Gráficas matemáticas
- Esquemas conceptuales
