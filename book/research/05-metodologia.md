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

# Metodología: La Receta Exacta de tu Investigación

> **Módulo:** Research
> **Semana:** 5
> **Tiempo de lectura:** ~25 minutos

---

## Introducción

Imagina que compartías una receta para hacer un pastel. Si solo escribes "combina ingredientes y hornea", nadie más podrá reproducir tu pastel. Debes especificar: qué ingredientes exactamente, en qué cantidades, a qué temperatura, por cuánto tiempo.

Tu sección de Metodología es exactamente esto: una **receta tan detallada que otro investigador debería poder reproducir tu investigación exactamente**, obteniendo los mismos resultados.

Esta es la sección donde la precisión no es solo importante; es fundamental. Vaguedad en Metodología destruye la reproducibilidad, la cual es el corazón de la investigación científica.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Estructurar una sección de Metodología que es clara y reproducible
2. Especificar versiones exactas, configuraciones, y hiperparámetros
3. Documentar datasets, modelos, y herramientas usadas
4. Diseñar un experimento que responde tu pregunta de investigación
5. Evitar las trampas comunes de metodología deficiente

---

## La Diferencia entre "Lo que Hice" y "Metodología"

Muchos estudiantes confunden estos.

**"Lo que hice"** (insuficiente):
"Entrené un modelo transformer fine-tuned para generar kernels GPU. Luego lo evaluamos en un dataset de 500 kernels."

**Metodología rigurosa** (suficiente):
"Usamos el modelo Llama 2 7B (versión huggingface/llama-2-7b-chat, descargado el 2024-01-15). Fine-tunning sobre 10,000 kernels de GitHub (repositorios públicos con licencia MIT/Apache, filtrados por tamaño <1000 líneas). Fine-tunning parameters: learning rate 2e-5, batch size 16, 3 epochs, AdamW optimizer con weight decay 0.01. Evaluamos en 500 kernels retenidos (20% de split aleatorio, seed 42). Métricas: sintaxis válida (compilación exitosa con nvcc), corrección funcional (test suite de 50 tests), latencia de generación (<200ms en A100 GPU)."

¿Ves la diferencia? La segunda permite reproducción.

> 💡 **Concepto clave:** Tu Metodología debería ser tan específica que un colega en otra universidad pudiera implementarla sin contactarte.

---

## Estructura de una Sección de Metodología

La mayoría de metodologías siguen esta estructura:

:::{figure} diagrams/methodology_flow.png
:name: fig-methodology-flow
:alt: Diagrama de flujo mostrando los pasos de una metodología experimental: visión general, datasets, modelos/herramientas, procedimiento experimental, métricas de evaluación y análisis estadístico
:align: center
:width: 90%

**Figura 1:** Flujo típico de una sección de Metodología, mostrando los componentes secuenciales necesarios para una investigación reproducible y rigurosa.
:::

```
1. Visión General / Enfoque General
   "Tomamos un enfoque empírico donde..."

2. Datasets
   - Descripción, origen, tamaño, características
   - Cómo fue dividido (train/val/test)
   - Cualquier preprocessing

3. Modelos / Herramientas
   - Qué modelos o sistemas usaste
   - Versiones exactas
   - Configuración

4. Procedimiento Experimental
   - Paso a paso qué hiciste
   - Orden de operaciones
   - Cualquier proceso iterativo

5. Métricas de Evaluación
   - Qué mediste
   - Cómo calculaste cada métrica
   - Qué significa cada métrica

6. Análisis Estadístico
   - Cómo determinaste si diferencias eran "significativas"
   - Intervalos de confianza, p-valores, etc.
```

### Ejemplo: Investigación de Restricciones Gramaticales en Generación de Kernels

**1. Visión General**

"Condujimos un estudio controlado donde comparamos tres enfoques para generar kernels CUDA seguros: (1) modelo base sin restricciones, (2) modelo con restricciones regex simples, (3) modelo con restricciones EBNF completas. Para cada enfoque, entrenamos un modelo separado, lo evaluamos en un dataset de prueba común, y medimos múltiples dimensiones de calidad."

**2. Datasets**

"Nuestro dataset consiste en 10,000 kernels CUDA funcionales de código abierto:
- Fuentes: Repositorios NVIDIA, OpenACC benchmarks, GitHub (búsqueda de repositorios >10 estrellas con lenguaje CUDA)
- Filtrado: Solo kernels entre 50-500 líneas (pequeños suficientes para estudiar, grandes suficientes para ser realistas)
- Balancing: Distribuido entre tipos de kernel: 30% memory-bound, 40% compute-bound, 30% I/O-bound
- División: 70% entrenamiento (7,000), 15% validación (1,500), 15% prueba (1,500)
- Preprocesamiento: Removed comments, formatted con clang-format estándar"

**3. Modelos / Herramientas**

"Partimos del modelo Llama 2 Chat 7B (versión huggingface/llama-2-7b-chat, commit SHA: 08d707db89fdf6ba0f10898cc281f963).

Para implementar restricciones:
- Restricciones regex: Implementadas con librería 'guidance' (versión 0.0.64)
- Restricciones EBNF: Implementadas con librería 'xgrammar' (versión 0.1.2, compiladas para CUDA)

Fine-tuning: LoRA (Low-Rank Adaptation) con rango 8, learning rate 2e-4 (determinado en validación), batch size 16, optimizer AdamW, weight decay 0.01, 3 epochs, early stopping si validación loss no mejora por 2 checkpoints."

**4. Procedimiento Experimental**

"Para cada una de las tres condiciones:
1. Especificamos la restricción (regex pattern o gramática EBNF)
2. Fine-tuned el modelo Llama 2 sobre el dataset de entrenamiento
3. Evaluamos en el dataset de validación, ajustando hiperparámetros según validación loss
4. Una vez finalizados hiperparámetros, entrenamos en train+val combinado (8,500 kernels) y evaluamos una vez en el dataset de prueba (1,500 kernels)
5. Para cada kernel de prueba, generamos 5 salidas diferentes (temperature 0.7) y reportamos estadísticas agregadas

Todas las corridas fueron en el mismo hardware (NVIDIA A100 80GB GPU) para control de varianza."

**5. Métricas de Evaluación**

"Reportamos múltiples métricas:

a) Validez sintáctica: ¿Compila el kernel con nvcc?
   - Definición: Binario (compila / no compila)
   - Agregación: Porcentaje de kernels que compilan exitosamente

b) Corrección funcional: ¿Produce el kernel resultados correctos?
   - Definición: Ejecuta un test suite de 50 kernels pequeños con datos de entrada conocida
   - Métrica: Porcentaje de tests que producen output correcto
   - Nota: No evaluamos performance absoluta, solo corrección

c) Latencia de generación: ¿Cuánto tarda generar un kernel?
   - Definición: Tiempo de wall-clock desde inicio de generación hasta generación de primer kernel válido
   - Agregación: Mediana, percentil 75, percentil 95 (reportamos tres para capturar distribución)

d) Similitud a código de entrenamiento: ¿Cuán similares son kernels generados al código de entrenamiento?
   - Definición: Similitud de AST (Abstract Syntax Tree) cosine, comparado con kernels de entrenamiento más cercanos
   - Métrica: Percentil 95 de similitud (captura kernels 'más copiados')"

**6. Análisis Estadístico**

"Para determinar si las diferencias entre condiciones eran estadísticamente significativas:
- Condujimos t-tests apareados (comparando modelo base vs. cada uno de los modelos con restricciones)
- Asumimos significance level α = 0.05
- Reportamos p-valores y intervalos de confianza del 95%
- Para métricas no-normales (ej: latencia), verificamos con test de Wilcoxon signed-rank como sanidad-check"

---

## Evitar Trampas Comunes de Metodología

### Trampa 1: Overfitting Accidental

Usas el dataset de prueba para ajustar hiperparámetros. Luego reportas resultados en el mismo dataset. Esto infla resultados.

✅ Correcto: Train → Validation (ajustar) → Test (reportar solo una vez)

### Trampa 2: Versionitis

"Usamos Python, TensorFlow, Huggingface." Seis meses después, TensorFlow está en v2.13, Huggingface en v4.30. ¿Cuál usaste?

✅ Correcto: "Python 3.10.8, PyTorch 2.0.1, transformers==4.28.1, xgrammar==0.1.2"

### Trampa 3: Incompletitud de Detalles

"Entrenamos con los parámetros estándar." Estudiante 2 intenta reproducir, no sabe cuáles son "estándar".

✅ Correcto: List every relevant parameter explícitamente

### Trampa 4: Cambios Post-hoc

Después de ver resultados: "Oh, ese outlier debe ser descartado" o "Cambié cómo calculo esa métrica porque me parece mejor ahora." Esto es moving the goalposts.

✅ Correcto: Especifica todos los criterios **antes** de ver resultados

### Trampa 5: Falta de Seeds/Randomness

"Entrenamos 3 veces y reportamos el promedio." ¿Usaste la misma seed? ¿Diferentes seeds?

✅ Correcto: "Usamos seed 42 para todos los procesos estocásticos. Reportamos resultados de una corrida; para estabilidad, entrenamos 3 veces con seeds 42, 123, 456 y reportamos media ± desviación estándar en la sección de Resultados."

### Trampas Específicas de ML/AI

**Trampas ML/AI:** (1) Data leakage: normalizar con todo el dataset en vez de solo train. (2) Convergencia no verificada: reportar modelo que aún no convergió. (3) Cherry-picking: probar hiperparámetros en test set. (4) Ignorar class imbalance: reportar accuracy cuando hay 95%/5% split. (5) No documentar hardware: GPU model afecta resultados. (6) Baseline débil: comparar contra método trivial.

> 💡 **Consejo práctico:** Documenta tu metodología MIENTRAS haces la investigación, no después. Conforme haces decisiones, anótalas. Cuando escribas la sección, tendrás todo el detalle necesario.

---

## Reproducibilidad: Más Allá de la Escritura

Escribir una buena Metodología es solo el primer paso. Para reproducibilidad verdadera:

**1. Código Público**
Comparte tu código en GitHub (o similar). Idealmente con un README que:
- Explica cómo usar el código
- Especifica dependencias y versiones
- Da ejemplos de cómo reproducir resultados clave

**2. Datasets Públicos (o Acceso)**
Si usaste datos públicos, proporciona exactamente dónde descargarlos. Si datos privados, explica cómo otros pueden obtener acceso.

**3. Configuraciones Guardadas**
Guarda todos los hiperparámetros, seeds, y configuraciones en archivos (JSON, YAML). Alguien debería poder reproducir solo usando esos archivos.

**4. Documentación de Cambios**
Si realizaste cambios después de ver resultados iniciales, documéntalo explícitamente. ("En el v1.0, usamos learning rate 1e-4; en v1.1, lo cambiamos a 2e-4 basado en validación loss mejorado.")

---

## Diferencias de Metodología según Tipo de Investigación

No todas las investigaciones son iguales. Ajusta tu estructura según tipo:

### Investigación Experimental Cuantitativa (La tuya probablemente)

Énfasis en: Datasets exactos, versiones exactas, métricas precisas, análisis estadístico

### Investigación Cualitativa

Énfasis en: Selección de participantes, protocolo de entrevista, estrategia de análisis de datos

### Investigación Mixta

Énfasis en: Ambos arriba (metodología cuantitativa + datos de entrevista)

### Estudio de Caso

Énfasis en: Por qué seleccionaste este caso específico, cómo es representativo o informativo

---

## Checklist de Metodología Completa

Antes de considerar tu Metodología "hecha", pasa esto:

- [ ] Cada dataset es descrito con tamaño, origen, y características
- [ ] Cada herramienta/modelo incluye nombre, versión exacta, link a documentación
- [ ] Cada hiperparámetro está listado con su valor
- [ ] Procedimiento experimental es descrito paso-a-paso, en orden
- [ ] Cada métrica es definida operacionalmente (cómo exactamente se calcula)
- [ ] Análisis estadístico es descrito (qué test, qué significance level)
- [ ] Reproducibilidad: Un colega podría reproducir en otras universidad
- [ ] Claridad: Un lector sin especialidad entiende el enfoque general
- [ ] Integridad: No has movido los goalposts post-hoc

---

## Resumen

En esta lectura exploramos:

- **Reproducibilidad como objetivo central:** Tu Metodología es una receta; debería ser exacta
- **Estructura estándar:** Visión general, datasets, modelos, procedimiento, métricas, análisis
- **Especificidad implacable:** Versiones, hiperparámetros, seeds, todo documentado
- **Trampas a evitar:** Overfitting en prueba, vaguedad, cambios post-hoc
- **Más allá de escribir:** Código público, documentación, configuraciones guardadas

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Cuál es la diferencia entre "lo que hice" y una Metodología rigurosa? Proporciona un ejemplo de ambas.

2. ¿Por qué es importante documentar la versión exacta de software, no solo el nombre?

3. ¿Qué es "overfitting en el dataset de prueba" y cómo prevenirlo?

### Ejercicio práctico

**Tarea 1: Outline de Metodología (40 minutos)**

Escribe un outline para tu sección de Metodología. Debería incluir:

```
1. Visión General / Enfoque
   - Qué tipo de estudio (empírico, comparativo, etc.)
   - Condiciones a comparar

2. Datasets
   - Tamaño, origen, características
   - División train/val/test
   - Preprocesamiento

3. Modelos / Herramientas
   - Nombres exactos y versiones
   - Configuraciones principales

4. Procedimiento
   - Pasos en orden
   - Hardware usado

5. Métricas
   - Cada métrica, cómo calculada
   - Por qué es importante

6. Análisis Estadístico
   - Qué tests, qué significance level
```

**Tarea 2: Especificidad de Detalles (30 minutos)**

Selecciona un componente de tu metodología (ej: un modelo, un dataset, un hiperparámetro). Escribe una descripción que sea tan específica que alguien pudiera reproducirla perfectamente. Incluye:
- Nombres exactos
- Versiones (con commits o links si aplicable)
- Valores precisos
- Reasoning (por qué escogiste estos valores)

**Tarea 3: Reproducibilidad Checklist (20 minutos)**

Imagina que compartiste tu proyecto de investigación con alguien en otra universidad. Pasa por el checklist y para cada ítem, escribe: "Sí, documenté esto" o "No, necesito documentar..."

### Para pensar

> *¿Cuál sería más valioso para el campo: un proyecto de investigación con resultados impresionantes pero vaga Metodología, o un proyecto de investigación con resultados modestos pero Metodología perfectamente reproducible? ¿Por qué?*

---

## Próximos pasos

Con tu Metodología clara, estás lista para presentar tus **Resultados**. Aquí documentarás qué encontraste, presentando data en forma clara, gráficos bien diseñados, e interpretación cuidadosa. Verás cómo tu Metodología cuidadosa permite resultados creíbles.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*
