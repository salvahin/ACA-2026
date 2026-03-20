# ACA: Grammar-Constrained GPU Kernel Generation

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/intro.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton xgrammar
print('Dependencies installed!')
```

```{admonition} El Reto: Un Problema de Investigación Abierta
:class: important

**¿Puedes enseñar a una IA a escribir código GPU ultra-optimizado?**

En este reto construirás un sistema que combina **Inteligencia Artificial**, **Teoría de Compiladores** y **GPU Computing** para generar automáticamente kernels de alto rendimiento con garantías formales de corrección.
```

```{admonition} ¿Por qué es un Problema Abierto?
:class: warning

Este no es un problema resuelto. Las herramientas actuales enfrentan **limitaciones fundamentales**:

| Desafío | Estado Actual | Oportunidad de Mejora |
|---------|---------------|----------------------|
| **Corrección semántica** | Las gramáticas CFG solo garantizan sintaxis, no semántica | Gramáticas con atributos, verificación formal |
| **Optimización de rendimiento** | Código correcto ≠ código eficiente | Patrones de optimización en gramáticas |
| **Generalización** | Gramáticas específicas por dominio | Meta-gramáticas adaptativas |
| **Latencia de inferencia** | Token masking añade overhead | Cacheo predictivo, compilación ahead-of-time |
| **Context-sensitive constraints** | CFGs no pueden expresar "variable declarada antes de uso" | Extensiones Type-1, análisis semántico híbrido |

**Nadie ha resuelto completamente** cómo generar código GPU que sea simultáneamente:
1. Sintácticamente correcto (garantizado por gramáticas)
2. Semánticamente válido (sin errores de memoria, race conditions)
3. Óptimamente eficiente (rendimiento competitivo con código humano)
4. Generalizable (funciona para nuevos tipos de operaciones)
```

```{admonition} Tu Contribución: Propuestas de Mejora
:class: seealso

Como parte de tu proyecto, se te invita a **proponer mejoras** que ayuden a resolver aspectos de este problema abierto. Algunas direcciones prometedoras:

**Mejoras al Pipeline de Gramáticas:**
- Diseñar gramáticas que capturen patrones de optimización GPU (coalescing, tiling, etc.)
- Implementar verificación semántica post-generación eficiente
- Explorar gramáticas jerárquicas que combinen estructura y optimización

**Mejoras al Constrained Decoding:**
- Desarrollar estrategias de caching más agresivas para token masking
- Investigar lookahead para reducir generaciones fallidas
- Combinar constrained decoding con beam search optimizado

**Mejoras a la Evaluación:**
- Definir métricas que capturen tanto corrección como eficiencia
- Crear benchmarks representativos de workloads reales
- Desarrollar tests de regresión automatizados

**Las mejores propuestas podrían convertirse en:**
- Contribuciones a proyectos open-source (XGrammar, vLLM, Triton)
- Publicaciones en workshops de ML Systems (MLSys, NeurIPS)
- Fundamento para tesis de posgrado o proyectos de investigación
```

---

## El Problema

Las GPUs son el motor de la revolución de IA. Cada modelo que usas—GPT-4, Llama, Stable Diffusion—depende de miles de operaciones ejecutándose en paralelo en hardware especializado.

**Pero hay un problema:**

- Escribir kernels GPU eficientes es **extremadamente difícil**
- Un error de indexación puede corromper silenciosamente tus resultados
- La diferencia entre un kernel lento y uno rápido puede ser **100x**
- Los LLMs pueden generar código, pero **sin garantías de corrección**

```{admonition} La Solución
:class: important

**Gramáticas formales que restringen la generación de código.**

En lugar de dejar que el LLM genere cualquier cosa, lo guiamos con reglas precisas que garantizan que el código generado sea sintácticamente correcto y siga patrones de optimización probados.
```

---

## ¿Por Qué Este Reto?

```{code-cell} ipython3
:tags: [remove-input]

import plotly.graph_objects as go

# Crear gráfico de radar de habilidades
categories = ['IA Generativa', 'Compiladores', 'GPU Computing',
              'Investigación', 'MLOps', 'Comunicación']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=[95, 90, 85, 80, 75, 70],
    theta=categories,
    fill='toself',
    name='Habilidades que desarrollarás',
    line_color='#6366f1',
    fillcolor='rgba(99, 102, 241, 0.3)'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100])
    ),
    showlegend=True,
    title="Perfil de Competencias ACA",
    height=400
)

fig.show()
```

### Trabaja en la Frontera

Este no es un proyecto académico aislado. Es el **mismo problema** que resuelven equipos en:

- **NVIDIA** - Optimización de kernels para sus GPUs
- **Meta** - Aceleración de Llama con Triton
- **OpenAI** - Eficiencia de inference para GPT
- **Google** - Compiladores XLA para TPUs

### Habilidades de Alta Demanda

Los ingenieros que dominan la intersección de IA + Compiladores + GPU son **extremadamente escasos** y **muy bien pagados**.

```{admonition} Dato de Mercado
:class: note

El salario promedio para GPU/AI Engineers en Silicon Valley supera los **$200,000 USD** anuales. La demanda crece más rápido que la oferta de talento.
```

---

## Objetivos de Aprendizaje

Al finalizar este reto, serás capaz de:

```{admonition} Competencias Técnicas
:class: seealso

1. **Diseñar gramáticas formales** para restringir generación de código
2. **Implementar kernels GPU optimizados** con Triton y CUDA
3. **Construir pipelines de constrained decoding** con LLMs
4. **Evaluar y benchmarkear** rendimiento de código generado
5. **Aplicar metodología de investigación** rigurosa
6. **Comunicar resultados técnicos** efectivamente
```

---

## Tu Travesía: 10 Semanas

Durante 10 semanas, cursarás **6 módulos en paralelo**, cada uno con 2 horas de clase semanales.

```{code-cell} ipython3
:tags: [remove-input]

import plotly.graph_objects as go

modulos = ['AI', 'Compiladores', 'Estadística', 'Research', 'Project 1', 'Project 2']
lecturas = [8, 10, 10, 10, 10, 10]
colores = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e']

fig = go.Figure()

fig.add_trace(go.Bar(
    x=modulos,
    y=lecturas,
    marker_color=colores,
    text=[f'{l} lecturas' for l in lecturas],
    textposition='outside'
))

fig.update_layout(
    title="Contenido por Módulo",
    yaxis_title="Número de lecturas",
    showlegend=False,
    height=300,
    yaxis=dict(range=[0, 12])
)

fig.show()
```

**Carga semanal:** 12 horas presenciales (6 módulos × 2 horas)

Consulta el [calendario detallado](calendario) para ver los temas semana a semana.

---

## Los 6 Módulos

````{grid} 2
:gutter: 3

```{grid-item-card} Inteligencia Artificial
:link: ai/index
:link-type: doc

Fundamentos de Deep Learning, arquitectura Transformer, LLMs, tokenización, sampling strategies, fine-tuning y evaluación.

**8 lecturas** | Nivel: Intermedio
```

```{grid-item-card} Compiladores
:link: compilers/index
:link-type: doc

Teoría de lenguajes formales, gramáticas libres de contexto, BNF/EBNF, parsing, FSMs y su aplicación en constrained decoding.

**10 lecturas** | Nivel: Intermedio
```

```{grid-item-card} Estadística
:link: stats/index
:link-type: doc

Fundamentos de probabilidad, pruebas de hipótesis, power analysis, diseño experimental, análisis de resultados.

**10 lecturas** | Nivel: Intermedio
```

```{grid-item-card} Research
:link: research/index
:link-type: doc

Metodología de investigación, escritura académica, revisión bibliográfica, estructura de proyectos, preparación de defensa.

**10 lecturas** | Nivel: Todos
```

```{grid-item-card} Project 1: LLM Code Generation
:link: project_1/index
:link-type: doc

Generación de código con LLMs, XGrammar, constrained decoding, gramáticas Triton, serving y sistemas agénticos.

**10 lecturas** | Nivel: Avanzado
```

```{grid-item-card} Project 2: GPU Computing
:link: project_2/index
:link-type: doc

Arquitectura GPU, CUDA, Triton, optimización de memoria, benchmarking, debugging y análisis de rendimiento.

**10 lecturas** | Nivel: Avanzado
```
````

---

## Stack Tecnológico

```{code-cell} ipython3
:tags: [remove-input]

import plotly.graph_objects as go

categories = ['LLMs', 'Frameworks', 'Gramáticas', 'GPU', 'MLOps']
tools = [
    ['CodeLlama', 'DeepSeek', 'GPT-4'],
    ['PyTorch', 'Triton', 'vLLM'],
    ['XGrammar', 'JSON Schema', 'EBNF'],
    ['CUDA', 'Nsight', 'cuDNN'],
    ['W&B', 'Docker', 'Git']
]

fig = go.Figure()

colors = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899']

for i, (cat, tool_list) in enumerate(zip(categories, tools)):
    fig.add_trace(go.Bar(
        name=cat,
        x=[cat],
        y=[len(tool_list)],
        text='<br>'.join(tool_list),
        textposition='inside',
        marker_color=colors[i],
        hovertemplate=f"<b>{cat}</b><br>" + "<br>".join(tool_list) + "<extra></extra>"
    ))

fig.update_layout(
    title="Tecnologías que Dominarás",
    showlegend=False,
    height=350,
    yaxis_title="Herramientas"
)

fig.show()
```

## El Entregable Final

```{admonition} Tu Proyecto
:class: important

Un **sistema funcional** que genera kernels GPU correctos y optimizados usando LLMs con restricciones gramaticales, evaluado rigurosamente contra baselines de la industria.

**Incluye:**
- Implementación del pipeline completo
- Gramática formal para kernels Triton
- Suite de benchmarks y evaluación
- Análisis estadístico de resultados
- Documento de proyecto defendible
```

---

## 📐 Convención de Código del Curso

Para mantener coherencia en todo el repositorio seguimos esta convención:

```{admonition} Convención: Español para comunicar, Inglés para programar
:class: important

| Elemento | Idioma | Ejemplo |
|---|---|---|
| **Comentarios** | 🇲🇽 Español | `# Calcular el error cuadrático medio` |
| **Variables / Funciones** | 🇺🇸 Inglés | `learning_rate`, `batch_size`, `forward_pass()` |
| **Strings de texto** | 🇲🇽 Español | `print("Error de entrenamiento:")` |
| **Docstrings** | 🇲🇽 Español | `"""Calcula la pérdida MSE entre predicción y target."""` |
| **Nombres de clases** | 🇺🇸 Inglés | `class TransformerBlock:`, `class NeuralNetwork:` |

**Rationale:** Las variables y funciones en inglés son estándar en la industria global. Los comentarios en español facilitan la comprensión para todos los estudiantes del curso.
```

```python
# ✅ CORRECTO — sigue la convención
def compute_loss(predictions, targets):
    """Calcula el error cuadrático medio entre predicciones y objetivos."""
    # Diferencia entre predicción y valor real
    error = predictions - targets
    return (error ** 2).mean()

learning_rate = 0.001
num_epochs    = 100

# ❌ INCORRECTO — comentario en inglés
def compute_loss(predictions, targets):
    # Compute MSE between predictions and targets
    error = predictions - targets
    return (error ** 2).mean()

# ❌ INCORRECTO — variable en español
tasa_aprendizaje = 0.001
```

---

## Comienza Ahora

```{admonition} Siguiente Paso
:class: tip

Empieza por el módulo de **Inteligencia Artificial** para construir las bases de LLMs y generación de texto, o explora el **índice completo** abajo.

📴 **Sin Internet:** Consulta [Alternativas Offline](alternativas_offline.md) para equivalentes de texto/código de todos los videos de YouTube del curso.
🔑 **Sin GPU:** Revisa el [Notebook de Verificación de Entorno](notebooks/00_verificacion_entorno.ipynb) para confirmar tu configuración y opciones de CPU Fallback.
```

---

```{admonition} ⚠️ Contenido Generado con Inteligencia Artificial
:class: warning

Este material educativo fue generado con asistencia de modelos de lenguaje (LLMs). Aunque ha sido revisado, **puede contener imprecisiones** que requieren corrección:

- **Texto en imágenes:** Las visualizaciones generadas por IA pueden contener alucinaciones en etiquetas, números o texto embebido
- **Datos y cifras:** Estadísticas, fechas y referencias bibliográficas deben verificarse con fuentes primarias
- **Código:** Los ejemplos han sido probados, pero siempre verifica la lógica antes de usar en producción

**¿Encontraste un error?** Ayúdanos a mejorar este recurso:

👉 [Reportar un problema o sugerir corrección](URL_PLACEHOLDER_GOOGLE_FORMS)

Tu retroalimentación es valiosa para mantener la calidad del contenido, especialmente dado el volumen de material en los 6 módulos del curso.
```

---

```{tableofcontents}
```
