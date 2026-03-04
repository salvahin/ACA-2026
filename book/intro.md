# ACA: Grammar-Constrained GPU Kernel Generation

```{admonition} El Reto
:class: tip

**¿Puedes enseñar a una IA a escribir código GPU ultra-optimizado?**

En este reto construirás un sistema que combina **Inteligencia Artificial**, **Teoría de Compiladores** y **GPU Computing** para generar automáticamente kernels de alto rendimiento con garantías formales de corrección.
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

```{code-cell} ipython3
:tags: [remove-input]

import plotly.graph_objects as go

# Timeline del curso
fig = go.Figure()

phases = [
    {"name": "Fundamentos", "start": 1, "end": 3, "color": "#818cf8",
     "topics": "IA + Compiladores + Stats"},
    {"name": "Integración", "start": 4, "end": 6, "color": "#34d399",
     "topics": "Constrained Decoding + GPU"},
    {"name": "Aplicación", "start": 7, "end": 10, "color": "#f472b6",
     "topics": "Sistema + Tesis"}
]

for i, phase in enumerate(phases):
    fig.add_trace(go.Bar(
        x=[phase["end"] - phase["start"] + 1],
        y=[phase["name"]],
        orientation='h',
        name=phase["name"],
        marker_color=phase["color"],
        text=f"Semanas {phase['start']}-{phase['end']}: {phase['topics']}",
        textposition='inside',
        insidetextanchor='middle',
        base=phase["start"]-1
    ))

fig.update_layout(
    title="Roadmap del Reto ACA",
    xaxis_title="Semanas",
    barmode='stack',
    showlegend=False,
    height=250,
    xaxis=dict(tickmode='linear', tick0=1, dtick=1, range=[0, 11])
)

fig.show()
```

### Fase 1: Fundamentos (Semanas 1-3)

Construyes las bases teóricas en tres áreas clave:

- **IA Generativa**: Transformers, LLMs, tokenización, sampling
- **Compiladores**: Gramáticas formales, parsing, FSMs
- **Estadística**: Diseño experimental, pruebas de hipótesis

### Fase 2: Integración (Semanas 4-6)

Conectas teoría con práctica:

- **Constrained Decoding**: XGrammar, JSON Schema, EBNF
- **GPU Programming**: CUDA basics, Triton, optimización de memoria
- **Diseño de Gramáticas**: Capturando restricciones de kernels

### Fase 3: Aplicación (Semanas 7-10)

Construyes y evalúas tu sistema:

- **Implementación**: Pipeline completo de generación
- **Experimentación**: Benchmarks rigurosos vs baselines
- **Tesis**: Documentación y defensa de tu trabajo

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

Metodología de investigación, escritura académica, revisión bibliográfica, estructura de tesis, preparación de defensa.

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

---

## El Entregable Final

```{admonition} Tu Tesis
:class: important

Un **sistema funcional** que genera kernels GPU correctos y optimizados usando LLMs con restricciones gramaticales, evaluado rigurosamente contra baselines de la industria.

**Incluye:**
- Implementación del pipeline completo
- Gramática formal para kernels Triton
- Suite de benchmarks y evaluación
- Análisis estadístico de resultados
- Documento de tesis defendible
```

---

## Comienza Ahora

```{admonition} Siguiente Paso
:class: tip

Empieza por el módulo de **Inteligencia Artificial** para construir las bases de LLMs y generación de texto, o explora el **índice completo** abajo.
```

---

```{tableofcontents}
```
