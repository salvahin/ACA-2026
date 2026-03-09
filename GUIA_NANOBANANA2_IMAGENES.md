# Guía de Generación de Imágenes con Nano Banana 2

> **Propósito**: Documento de contexto para generar prompts de imágenes científicas, diagramas y visualizaciones para el curso TC3002B.

---

## Estructura de Prompt Recomendada

```
[Tipo de gráfico] + [Tema científico] + [Elementos clave] + [Estilo visual] + [Requisitos de anotación]
```

### Ejemplo Completo

```
Diagrama de flujo técnico mostrando el pipeline de tokenización en transformers.
Elementos: texto de entrada → tokenizer BPE → embeddings → positional encoding.
Estilo: diagrama técnico minimalista, fondo blanco, líneas azul oscuro.
Anotaciones: etiquetas en español, font sans-serif bold, tamaño consistente.
Aspect ratio 16:9 para presentación.
```

---

## 1. Principios de Prompting

### Ser Específico con Terminología del Dominio

| En lugar de... | Usa... |
|----------------|--------|
| "red neuronal" | "red neuronal feedforward de 3 capas con activación ReLU" |
| "diagrama de datos" | "diagrama de flujo de pipeline ETL con 5 etapas" |
| "gráfica" | "scatter plot con regresión lineal y bandas de confianza 95%" |

### Nano Banana 2 es un Modelo que Piensa

- Entiende intención, física y composición
- No solo matchea tags — interpreta contexto semántico
- Puede acceder a Google Search para referencias de sujetos reales

### Framing Positivo

```
# Correcto
"calle vacía al amanecer"

# Incorrecto
"calle sin coches ni personas"
```

---

## 2. Renderizado de Texto

### Reglas Críticas

1. **Usa comillas** para todo texto deseado:
   ```
   Diagrama con etiqueta "Capa de Atención" y flecha hacia "Output"
   ```

2. **Especifica tipografía**:
   ```
   Font: bold sans-serif, color blanco sobre fondo oscuro
   ```
   ```
   Estilo tipográfico: Century Gothic 14px, alto contraste
   ```

3. **Truco Text-First** (para texto complejo):
   - Primero: conversa para definir los conceptos y texto exacto
   - Después: pide la imagen con ese texto

### Soporte Multilingüe

- Español funciona bien para etiquetas
- Para anotaciones extensas, especifica: `"etiquetas en español latino, font compatible con acentos"`

---

## 3. Paleta de Colores Académicos

### Convenciones Científicas

| Elemento | Color Recomendado |
|----------|-------------------|
| Señales activadoras / positivas | Verde |
| Señales inhibidoras / negativas | Rojo |
| Rutas neutrales / pathways | Azul o gris |
| Datos de entrada | Azul claro |
| Datos de salida | Azul oscuro |
| Errores / warnings | Naranja o rojo |
| Highlights importantes | Amarillo |

### Accesibilidad

- Cumplir WCAG 2.1 AA de contraste
- **Evitar** combinaciones rojo-verde exclusivas (daltonismo)
- Usar patrones o texturas además de color cuando sea posible

### Prompt de Ejemplo

```
Usar paleta de colores académica: azules para flujo de datos,
verde para activaciones, rojo para gradientes de error.
Alto contraste, accesible para daltonismo.
```

---

## 4. Layout y Composición

### Organización Visual

- **Flujo de lectura**: izquierda → derecha, arriba → abajo
- **Jerarquía clara**: elementos principales más grandes/prominentes
- **Espacio blanco**: suficiente separación entre elementos

### Chain-of-Thought para Composiciones Complejas

```
Primero, divide el canvas en 3 columnas iguales.
En la columna izquierda, coloca el bloque de "Input".
En la columna central, muestra el proceso de "Transformación" con 3 pasos verticales.
En la columna derecha, presenta el "Output" con métricas.
Conecta los elementos con flechas direccionales azules.
```

### Aspect Ratios por Uso

| Uso | Aspect Ratio |
|-----|--------------|
| Presentaciones / slides | 16:9 |
| Documentos / papers | 4:3 o 3:2 |
| Banners / headers | 8:1 o 21:9 |
| Posts cuadrados | 1:1 |
| Diagramas verticales | 9:16 o 1:2 |

---

## 5. Tipos de Visualización por Contenido

### Mapeo Contenido → Visualización

| Contenido del Curso | Tipo de Diagrama |
|---------------------|------------------|
| Arquitectura Transformer | Diagrama de bloques con conexiones |
| Tokenización BPE/WordPiece | Diagrama de árbol o flujo |
| Attention mechanism | Heatmap + diagrama de conexiones |
| Training loop | Diagrama de flujo cíclico |
| Comparación de modelos | Tabla visual o gráfico de barras |
| Pipeline de datos | Diagrama de flujo horizontal |
| Jerarquía de conceptos | Diagrama de árbol o mapa mental |
| Proceso temporal | Timeline horizontal |
| Relaciones entre componentes | Diagrama de red/grafo |
| Distribuciones estadísticas | Histogramas, box plots, density plots |

---

## 6. Estilos Visuales Recomendados

### Para Material Educativo

```
Estilo: diagrama técnico limpio y minimalista.
Fondo: blanco o gris muy claro (#f5f5f5).
Líneas: grosor consistente, colores sólidos.
Sin sombras ni efectos 3D innecesarios.
Etiquetas claras y legibles.
```

### Para Presentaciones

```
Estilo: infografía moderna y profesional.
Colores: paleta limitada a 3-4 colores complementarios.
Iconografía: simple y reconocible.
Alto contraste para proyección.
```

### Para Papers/Publicaciones

```
Estilo: figura científica formal.
Escala de grises con un color de acento.
Incluir leyenda si hay múltiples elementos.
Resolución: 300 DPI mínimo.
```

---

## 7. Especificaciones Técnicas

### Resolución y Formato

| Destino | Especificación |
|---------|----------------|
| Web / Jupyter Book | PNG, 150 DPI |
| Presentaciones | PNG, 200 DPI |
| Papers (Nature) | 300+ DPI, RGB, ≥180mm ancho |
| Papers (PLOS ONE) | PNG/TIFF, 300 DPI |
| Impresión general | 300 DPI mínimo |

### Prompt de Exportación

```
Output en alta resolución 4K, formato adecuado para publicación académica,
300 DPI, espacio de color RGB.
```

---

## 8. Templates por Módulo del Curso

### Módulo AI/Deep Learning

```
Diagrama técnico de [CONCEPTO].
Mostrar: [COMPONENTES PRINCIPALES].
Flujo de datos de izquierda a derecha.
Estilo: minimalista, fondo blanco, líneas azul (#1a73e8).
Etiquetas en español: "[ETIQUETA1]", "[ETIQUETA2]".
Aspect ratio 16:9.
```

### Módulo Compilers/Gramáticas

```
Diagrama de [autómata/árbol de parseo/gramática].
Elementos: estados como círculos, transiciones como flechas etiquetadas.
Estilo: diagrama de teoría de computación clásico.
Etiquetas: "[SÍMBOLOS]" en font monospace.
Fondo blanco, líneas negras, estados de aceptación con doble círculo.
```

### Módulo Estadística

```
[Tipo de gráfico estadístico] mostrando [DATOS].
Incluir: ejes etiquetados, leyenda, título.
Estilo: gráfico científico limpio.
Colores: escala de azules para datos, rojo para valores significativos.
Anotaciones: valores p, intervalos de confianza donde aplique.
```

### Módulo Research/Metodología

```
Diagrama de flujo metodológico de [PROCESO].
Etapas: [LISTA DE ETAPAS].
Estilo: flowchart profesional con formas estándar
(rectángulos para procesos, diamantes para decisiones).
Conexiones claras con flechas direccionales.
```

---

## 9. Advertencias Críticas

### Siempre Verificar

- **Precisión científica**: El modelo puede malinterpretar o inventar datos
- **Estructuras químicas**: NO confiar — han aparecido publicaciones con estructuras incorrectas
- **Fórmulas matemáticas**: Verificar cada símbolo y relación
- **Datos numéricos**: Nunca usar números generados sin verificación

### Limitaciones Conocidas

- Texto pequeño puede no renderizarse perfectamente
- Detalles muy finos pueden perderse
- Traducciones pueden tener errores gramaticales menores

### Proceso de Revisión

1. Generar imagen inicial
2. Revisar precisión técnica de TODO el contenido
3. Iterar con correcciones específicas
4. Verificación final por experto del dominio

---

## 10. Ejemplos de Prompts para TC3002B

### Arquitectura Transformer

```
Diagrama técnico de la arquitectura Transformer encoder-decoder.
Mostrar: Input Embedding, Positional Encoding, Multi-Head Attention,
Feed Forward, Layer Norm, Output.
Encoder a la izquierda, Decoder a la derecha, conectados por cross-attention.
Estilo: diagrama de bloques limpio, fondo blanco.
Colores: bloques azules para encoder, verdes para decoder, flechas grises.
Etiquetas en español: "Atención Multi-Cabeza", "Normalización de Capa".
Aspect ratio 16:9, alta resolución.
```

### Pipeline de Tokenización

```
Diagrama de flujo horizontal del proceso de tokenización BPE.
Etapas: "Texto crudo" → "Pre-tokenización" → "Entrenamiento BPE" →
"Vocabulario" → "Encoding" → "Token IDs".
Mostrar ejemplo: "tokenización" → ["token", "ización"] → [1547, 892].
Estilo: infografía educativa, colores suaves.
Flechas direccionales entre cada etapa.
```

### Autómata Finito

```
Diagrama de autómata finito determinista (DFA) para reconocer
el lenguaje de identificadores válidos.
Estados: q0 (inicial), q1 (aceptación), q_err (error).
Transiciones etiquetadas con: letras [a-z], dígitos [0-9], underscore.
Estilo: diagrama clásico de teoría de computación.
Estados como círculos, q1 con doble círculo.
Flechas con etiquetas en font monospace.
Fondo blanco, líneas negras.
```

### Box Plot Comparativo

```
Box plot comparando distribuciones de accuracy entre 3 modelos:
"BERT", "GPT-2", "T5".
Eje Y: Accuracy (0.0 - 1.0).
Mostrar: mediana, cuartiles, outliers.
Estilo: gráfico científico formal.
Colores: azul, verde, naranja para cada modelo.
Incluir leyenda y título: "Comparación de Accuracy por Modelo".
Grid sutil en fondo.
```

---

## Referencias

- [Google Cloud: Ultimate prompting guide for Nano Banana](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-nano-banana)
- [Nano Banana Pro Scientific Illustration Guide](https://help.apiyi.com/en/nano-banana-pro-scientific-illustration-guide-en.html)
- [Beyond Infographics: How to Use Nano Banana for Learning](https://drphilippahardman.substack.com/p/beyond-infographics-how-to-use-nano)
- [ImagineArt: Nano Banana 2 Prompt Guide](https://www.imagine.art/blogs/nano-banana-2-prompt-guide)
- [DeepDream: Nano Banana 2 Best Prompts](https://deepdreamgenerator.com/blog/nano-banana-2-best-prompts)

---

*Última actualización: Marzo 2026*
