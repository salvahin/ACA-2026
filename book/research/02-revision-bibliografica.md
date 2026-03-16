---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Revisión Bibliográfica: Aprender a Leer Papers como un Investigador
```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
# Este módulo no requiere instalaciones de Python adicionales.
# Asegúrate de haber ejecutado 00_verificacion_entorno.ipynb antes de iniciar el proyecto.
print("Módulo Research — sin dependencias de código adicionales.")
print('Dependencies installed!')
```

> **Módulo:** Research
> **Semana:** 2
> **Tiempo de lectura:** ~28 minutos

---

```{admonition} 🎬 Video Recomendado
:class: tip

**[Andrew Ng: How to Read a Research Paper](https://www.youtube.com/watch?v=733m6qBH-jI)** - El reconocido investigador de IA comparte su método personal para leer papers de investigación de manera eficiente, incluyendo la técnica de múltiples pasadas.
```

## Introducción

Si alguien te pidiera leer 200 papers de investigación para tu proyecto de investigación, probablemente entraría en pánico. Y con razón: a un ritmo de una hora por paper, serían meses de lectura pura. Pero aquí está la verdad incómoda: **no tienes que leer cada papel completamente para extraer su valor**.

Los investigadores experimentados desarrollan una técnica sofisticada: la lectura "dirigida por propósito". Lees diferentes partes de diferentes papers dependiendo de qué necesitas aprender. Es como ser un chef que conoce exactamente qué ingredientes necesita para cada plato, en lugar de probar toda la despensa.

En esta lectura aprenderás esa técnica, cómo gestionar la abrumadora cantidad de papers, cómo sintetizar información de múltiples fuentes, y cómo comparar los enfoques existentes (como XGrammar vs Outlines vs Guidance vs LMQL) en el contexto de tu investigación.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Aplicar la estrategia de lectura de 3 pasadas (Abstract → Conclusions → Methodology)
2. Clasificar papers por relevancia y extraer eficientemente lo que necesitas
3. Distinguir entre resumir (repetir) y sintetizar (conectar ideas)
4. Comparar sistemáticamente trabajos relacionados en tu campo

---

## La Estrategia de las Tres Pasadas: Lee Inteligentemente

La mayoría de estudiantes comienzan a leer un paper desde el título y avanzan linealmente palabra por palabra. Esto es ineficiente. En su lugar, usa una estrategia de selección y filtrado.

### Primera Pasada: ¿Es Este Paper Relevante? (5-10 minutos)

Lee **en este orden específico:**

1. **Título** - ¿Aborda un tema que me importa?
2. **Autores e institución** - ¿Quiénes son? ¿Dónde trabajan? (A veces, el pedigree importa)
3. **Abstract** - ¿Cuál es el problema, el enfoque propuesto, y los resultados?
4. **Conclusiones** - ¿Qué afirman haber aprendido?

En esta pasada, **tomas una decisión binaria:** ¿Debo leer este papel más a fondo?

**Preguntas para decidir:**
- ¿Este paper aborda un problema relacionado a mi investigación?
- ¿Propone un método o insight que debería entender?
- ¿Presenta resultados que contrastan con lo que yo espero encontrar?

Si la respuesta es "no" a todas, descarta el paper. Si es "sí" a al menos una, avanza a la segunda pasada.

> 💡 **Concepto clave:** La primera pasada es filtrado agresivo. Está bien descartar la mayoría de papers; no todos merecen tu atención profunda.

### Segunda Pasada: Entiende la Contribución Principal (15-20 minutos)

Ahora enfócate en estos elementos:

1. **Introducción** - ¿Cómo motivan el problema? ¿Qué trabajo anterior existe?

2. **Figuras y tablas** - Las imágenes frecuentemente comunican la idea central mejor que texto. Estudia los gráficos de resultados, arquitectura del sistema, etc.

3. **Sección de metodología** - ¿Cuál es la idea central? (No necesitas cada detalle técnico aún)

4. **Sección de resultados** - ¿Qué métrica es más importante? ¿Dónde es fuerte su método? ¿Dónde es débil?

En esta pasada, deberías poder responder: "¿Cuál fue la idea original y cómo funcionó?"

### Tercera Pasada: Profundiza en lo que Necesitas (Varía según tu necesidad)

Según lo que busques:

**Si necesitas reproducir el trabajo:**
- Lee la sección completa de metodología con rigor
- Nota detalles de configuración, hiperparámetros, tamaños de dataset
- Busca código suplementario o repositorios públicos

**Si necesitas entender el algoritmo:**
- Lee la descripción matemática de lado a lado
- Usa lápiz y papel para trabajar a través de un ejemplo

**Si solo necesitas citarlo como "trabajo relacionado":**
- La segunda pasada probablemente es suficiente

> 💡 **Consejo práctico:** No intentes la tercera pasada profunda en todos los papers. Reserva esa energía para los 5-10 papers más cruciales para tu investigación.

:::{figure} diagrams/three_pass_reading.png
:name: fig-three-pass-reading
:alt: Diagrama de estrategia de lectura de tres pasadas, mostrando la progresión desde lectura rápida de título y abstract, hasta profundización selectiva en metodología y detalles técnicos
:align: center
:width: 90%

**Figura 1:** Estrategia de las tres pasadas para lectura efectiva de papers académicos, permitiendo procesar múltiples artículos de manera eficiente y enfocar tiempo en los más relevantes.
:::

---

## Organización y Gestión de Tu Información

Mientras acumulas papers, necesitas un sistema para recuperarlos y sintetizar lo aprendido.

### Sistema de Etiquetado

En tu herramienta de referencias (Zotero, Mendeley), crea categorías como:

```
├── CORE (papers absolutamente esenciales para tu argumento)
├── Method/Related (papers sobre métodos similares o relacionados)
├── Background (papers que proporcionan contexto necesario)
├── Contrast (papers con enfoques diferentes a considerar)
└── Future (papers que sugieren direcciones futuras)
```

Mientras lees cada paper, añade:
- Una etiqueta de relevancia
- Notas clave (2-3 frases sobre su contribución)
- Conexiones con otros papers en tu colección

### Crear una Tabla Comparativa

Para papers en tu categoría "CORE" o sobre métodos relacionados, crea una tabla comparativa. Por ejemplo, para herramientas de generación grammar-constrained:

| Herramienta | Año | Lenguaje | Restricción | Velocidad | Aplicación |
|---|---|---|---|---|---|
| XGrammar | 2024 | Especific. de gramática | EBNF completo | GPU-optimizado | Gen. de cualquier lenguaje |
| Outlines | 2023 | Patrones regex | Regex + CFG simple | Moderad | JSON, código |
| Guidance | 2023 | Especific. GBNF | Limitado | Moderad | Formato de salida |
| LMQL | 2023 | Lenguaje específico | Control de flujo | Var. | Flujos complejos |

Esta tabla se convierte en un resumen ejecutivo de tu campo y ayuda a identificar brechas (quizás tu investigación aborda una que nadie más ha cubierto).

---

## Resumen vs. Síntesis: La Diferencia Crucial

Aquí es donde muchas revisiones bibliográficas fallan.

### Resumir (Evitar)

Un resumen simplemente repite lo que ya está en los papers. Es pasivo, tedioso de leer, y no añade valor.

**Ejemplo de mal resumen:**
"Smith et al. (2023) propusieron un método para generar kernels GPU usando transformers. Entrenaron su modelo en 10,000 kernels y lograron 87% de corrección. También encontraron que las restricciones gramaticales mejoraban la corrección a 92%."

Sí, estos hechos vienen del paper. Pero entonces, ¿por qué no solo leer el papel?

### Sintetizar (Hacer)

Síntesis significa conectar ideas de múltiples fuentes, identificar patrones, y extraer implicaciones.

**Ejemplo de buena sínproyecto de investigación:**
"Tres trabajos recientes (Smith et al. 2023, Johnson et al. 2024, Lee et al. 2023) convergen en un hallazgo: las restricciones gramaticales mejoran la corrección de código generado por LLMs, pero el mecanismo subyacente varía. Smith et al. atribuyen esto a la reducción del espacio de búsqueda; Johnson et al. sugieren que es sobre refuerzo de conceptos sintácticos durante entrenamiento. Esta divergencia sugiere una investigación más profunda es necesaria para entender si ambos mecanismos aplican o si uno domina."

Observa que:
- Citas múltiples trabajos como apoyo
- Identifica un patrón común
- Señala una tensión o pregunta abierta
- Prepara el camino para tu propia investigación

---

## Comparación de Enfoques: Un Caso de Estudio

Para ilustrar la sínproyecto de investigación, veamos cómo comparar herramientas de generación grammar-constrained.

### Dimensiones de Comparación

No compares "en general". Identifica dimensiones específicas:

**1. Modelo de restricción**
- XGrammar: Especificación EBNF completa, compilada a formato binario
- Outlines: Combina regex para patrones simples + CFG para más complejos
- Guidance: GBNF (parecido a EBNF pero orientado a LLMs)
- LMQL: Control de flujo imperativo, restricciones durante ejecución

**2. Overhead computacional**
- ¿Cuánto ralentiza la generación el enforcement de restricciones?
- ¿Es predecible (mismo overhead en todo) o variable?

**3. Expresividad**
- ¿Qué lenguajes/formatos pueden describirse?
- ¿Hay restricciones que no se pueden expresar?

**4. Facilidad de uso**
- ¿Necesitas PhD para escribir una especificación?
- ¿Hay documentación y ejemplos?

**5. Integración**
- ¿Funciona con tu LLM específico (Llama, GPT, Mistral)?
- ¿Requiere fine-tuning o funciona con modelos base?

### Síntesis de Comparación

"Mientras cada herramienta toma un enfoque diferente a restricciones gramaticales, todas operan en el mismo principio: reducir el espacio de tokens válidos en cada paso de generación. XGrammar logra el overhead más bajo porque pre-compila la gramática, pero requiere especificaciones más formales. Outlines sacrifica expresividad para ser más accesible, permitiendo usuarios escribir patrones familiares (regex). Esta es una verdadera diferencia de diseño: compilación vs. interpretación, con ventajas y tradeoffs."

---

## Construcción Progresiva de tu Revisión Bibliográfica

Tu revisión no se escribe en una sesión. Se construye gradualmente mientras lees.

### Semana 1-2: Mapeo del Paisaje
- Identifica los 3-4 "papers seminales" en tu área
- Identifica los trabajos recientes (últimos 2 años)
- Crea tu tabla comparativa inicial

### Semana 3-4: Profundización
- Lee los papers "CORE" en profundidad
- Identifica subtemas y escribe párrafos temáticos
- Nota preguntas que surgen

### Semana 5+: Síntesis y Cierre
- Conecta subtemas en narrativas coherentes
- Identifica brechas (dónde tu investigación entra)
- Escribe tu conclusión de revisión

> 💡 **Consejo práctico:** Revisa tu revisión cada 2 semanas con tu supervisor. Retroalimentación temprana evita reescrituras masivas después.

---

## Resumen

En esta lectura exploramos:

- **Lectura estratégica:** Las 3 pasadas (Abstract→Conclusions→Methodology) te permiten procesar 200+ papers eficientemente
- **Organización sistemática:** Un sistema de etiquetas y tablas comparativas convierte caos en claridad
- **Resumen vs. sínproyecto de investigación:** Los buenos revisiones conectan ideas, no solo las repiten
- **Comparación estructurada:** Identificar dimensiones específicas te permite comparaciones justas entre enfoques

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Cuáles son los tres elementos que siempre lees en la "primera pasada"? ¿Por qué estos específicamente?

2. ¿Cuál es la diferencia entre "resumen" y "sínproyecto de investigación"? Proporciona un ejemplo que NO sea del texto.

3. ¿En qué se diferencian XGrammar y Outlines en su enfoque a las restricciones gramaticales? ¿Cuáles son las implicaciones?

### Ejercicio práctico

**Tarea 1: Las 3 Pasadas (45 minutos)**

Selecciona 2 papers de tu campo (puedes buscar en arXiv.org o Google Scholar). Para cada uno:
- Pasada 1: Toma una decisión sobre si es relevante (5 min)
- Pasada 2: Escribe un párrafo sobre la contribución principal (10 min)
- Pasada 3: Profundiza en una sección según lo que necesites (15 min)

Documenta el tiempo que gastaste en cada pasada.

**Tarea 2: Tabla Comparativa (30 minutos)**

Crea una tabla comparando 3-4 herramientas o métodos relevantes a tu investigación. Incluye al menos 4 dimensiones de comparación. Intenta usar términos específicos en lugar de "bueno/malo".

**Tarea 3: Síntesis de Dos Papers (20 minutos)**

Lee dos papers sobre restricciones gramaticales o generación de código. Escribe un párrafo que sintetice los dos (no resuma cada uno por separado), identificando cómo sus contribuciones se relacionan.

### Para pensar

> *¿Cómo cambiará tu forma de leer papers ahora que entiendes que no necesitas leerlos completamente? ¿Cuáles son los riesgos de esta estrategia, y cómo podrías mitigarlos?*

---

## Próximos pasos

En la siguiente semana aprenderás a tomar lo que has aprendido de tu revisión bibliográfica y convertirlo en un **planteamiento de problema preciso y formalmente argumentado**. Verás cómo tu tabla comparativa te ayuda a identificar exactamente dónde tu investigación encaja y qué pregunta aún sin respuesta aborda.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Google Scholar. [Google Scholar](https://scholar.google.com/). Google.
- Semantic Scholar. [Semantic Scholar](https://www.semanticscholar.org/). AI2.
- Zotero. [Reference Management Software](https://www.zotero.org/). Zotero.
