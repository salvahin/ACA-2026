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

# Calendario del Curso

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/calendario.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton xgrammar
print('Dependencies installed!')
```

> **Duración:** 10 semanas | **Carga:** 12 horas/semana (6 módulos × 2 horas)
> **Inicio:** 23 de marzo 2026 | **Fin:** 3 de junio 2026

```{admonition} Nota sobre Vacaciones
:class: warning

Entre la Semana 1 (23 marzo) y la Semana 2 (8 abril) hay un período de vacaciones sin clases.
```

---

## Vista General

```{code-cell} ipython3
:tags: [remove-input]

import plotly.graph_objects as go

# Datos del calendario con fechas
fechas = ['23 Mar', '8 Abr', '15 Abr', '22 Abr', '29 Abr', '6 May', '13 May', '20 May', '27 May', '3 Jun']
modulos = ['AI', 'Compiladores', 'Estadística', 'Research', 'Project 1', 'Project 2']

# Matriz de contenidos (1 = lectura, 0.5 = revisión/proyecto)
data = [
    [1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5],  # AI (8 lecturas)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],      # Compiladores
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],      # Estadística
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],      # Research
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],      # Project 1
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],      # Project 2
]

fig = go.Figure()

# Crear heatmap
fig.add_trace(go.Heatmap(
    z=data,
    x=fechas,
    y=modulos,
    colorscale=[[0, '#f3f4f6'], [0.5, '#fbbf24'], [1, '#10b981']],
    showscale=False,
    hovertemplate='%{y}<br>%{x}<extra></extra>'
))

fig.update_layout(
    title="Mapa de Contenidos por Semana",
    height=300,
    margin=dict(b=50)
)

fig.show()
```

---

## Calendario Detallado

Todos los módulos se imparten en paralelo durante las 10 semanas del curso.

```{list-table}
:header-rows: 1
:widths: 12 15 15 15 15 15 15

* - Semana
  - AI (2h)
  - Compiladores (2h)
  - Estadística (2h)
  - Research (2h)
  - Project 1 (2h)
  - Project 2 (2h)
* - **1** (23 Mar)
  - [IA Clásica vs Generativa](ai/01_ia_clasica_vs_generativa)
  - [Introducción Compiladores](compilers/01-introduccion-compiladores)
  - [Fundamentos Probabilidad](stats/01-fundamentos-probabilidad)
  - [Qué es un Proyecto](research/01-que-es-un-proyecto)
  - [Setup e Intro LLMs](project_1/01_setup_intro_llms)
  - [GPU Fundamentals y PyTorch en GPU](project_2/01_gpu_fundamentals_pytorch)
* - **2** (8 Abr)
  - [Deep Learning](ai/02_fundamentos_deep_learning)
  - [Lenguajes Regulares](compilers/02-lenguajes-regulares-dfas)
  - [Distribuciones](stats/02-distribuciones-descriptiva)
  - [Revisión Bibliográfica](research/02-revision-bibliografica)
  - [LLMs para Código](project_1/02_llms_para_codigo)
  - [CUDA Conceptos y Primer Kernel Triton](project_2/02_cuda_triton_intro)
* - **3** (15 Abr)
  - [Generación Autoregresiva](ai/03_generacion_autoregresiva)
  - [NFAs y DFAs](compilers/03-nfas-conversion-dfa)
  - [Pruebas Hipótesis](stats/03-pruebas-hipotesis)
  - [Planteamiento Problema](research/03-planteamiento-del-problema)
  - [Prompt Engineering](project_1/03_prompt_engineering)
  - [Triton Completo: API y Patrones de Kernels](project_2/03_triton_patrones)
* - **4** (22 Abr)
  - [Arquitectura Transformer](ai/04_arquitectura_transformer)
  - [Gramáticas CFG](compilers/04-gramaticas-libres-contexto)
  - [Power Analysis](stats/04-poweranalysis-diseno)
  - [Marco Teórico](research/04-marco-teorico)
  - [XGrammar Constrained](project_1/04_xgrammar_constrained)
  - [Optimización de Kernels y Benchmarking](project_2/04_optimizacion_kernels)
* - **5** (29 Abr)
  - [BERT, GPTs, Tokens](ai/05_bert_gpts_tokenizacion)
  - [BNF/EBNF Parsing](compilers/05-bnf-ebnf-parsing)
  - [Reproducibilidad](stats/05-reproducibilidad)
  - [Metodología](research/05-metodologia)
  - [Gramática JSON](project_1/05_json_grammar)
  - [KernelBench y Pipeline de Evaluación](project_2/05_kernelbench_evaluacion)
* - **6** (6 May)
  - [Sampling Constrained](ai/06_sampling_constrained_decoding)
  - [Pipeline FSM-XGrammar](compilers/06-pipeline-fsm-xgrammar)
  - [Comparaciones Múltiples](stats/06-comparaciones-multiples)
  - [Estilo Académico](research/06-estilo-academico)
  - [Gramática Triton L1-L2](project_1/06_gramatica_triton_l1l2)
  - [Del Prompt al Agente Kernel](project_2/06_prompt_a_agente_kernel)
* - **7** (13 May)
  - [Fine-tuning Evaluación](ai/07_fine_tuning_evaluacion)
  - [Diseño Gramáticas DSL](compilers/07-diseno-gramaticas-dsl)
  - [Pruebas No Paramétricas](stats/07-pruebas-noparametricas)
  - [Resultados](research/07-resultados)
  - [Gramática Triton L3-L4](project_1/07_gramatica_triton_l3l4)
  - [Sesión 7: Agente con Loop de Retroalimentación](project_2/07_agente_loop)
* - **8** (20 May)
  - [MLOps Visualización](ai/08_mlops_visualizacion)
  - [Compilación Gramáticas](compilers/08-compilacion-gramaticas)
  - [Tamaño de Efecto](stats/08-tamano-efecto)
  - [Conclusiones Abstract](research/08-conclusiones-y-abstract)
  - [Serving: Inference](project_1/08a_serving_inference) / [Optimization](project_1/08b_serving_optimization)
  - [Debugging como Feedback del Agente](project_2/08_debugging_feedback)
* - **9** (27 May)
  - *Revisión y Proyecto*
  - [Limitaciones Context-Sensitive](compilers/09-limitaciones-context-sensitive)
  - [MLOps Visualización](stats/09-mlops-visualizacion)
  - [Revisión Estructura](research/09-revision-de-estructura)
  - [Token Economics](project_1/09_token_economics_integracion)
  - [Sesión 9: SYCL, Portabilidad y Multi-target](project_2/09_sycl_portabilidad)
* - **10** (3 Jun)
  - *Proyecto Final*
  - [Parser Generators](compilers/10-parser-generators-reflexion)
  - [Reporte Estadístico](stats/10-reporte-estadistico)
  - [Preparación Defensa](research/10-preparacion-defensa)
  - [Sistemas Agénticos](project_1/10_agentic_systems)
  - [Evaluación Final y Presentaciones](project_2/10_integracion_final)
```

---

## Resumen de Carga

```{code-cell} ipython3
:tags: [remove-input]

import plotly.graph_objects as go

fechas = ['23 Mar', '8 Abr', '15 Abr', '22 Abr', '29 Abr', '6 May', '13 May', '20 May', '27 May', '3 Jun']
semanas = list(range(1, 11))
horas_por_semana = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=fechas,
    y=horas_por_semana,
    marker_color='#6366f1',
    text=[f'Sem {i}<br>{h}h' for i, h in zip(semanas, horas_por_semana)],
    textposition='outside'
))

fig.update_layout(
    title="Carga Horaria por Fecha",
    yaxis_title="Horas",
    xaxis_title="Fecha de inicio de semana",
    showlegend=False,
    height=300,
    yaxis=dict(range=[0, 16])
)

fig.show()
```

---

## Notas

- **Vacaciones:** Entre semana 1 (23 Mar) y semana 2 (8 Abr) hay período de vacaciones
- Cada módulo tiene **2 horas de clase** por semana
- Total: **12 horas semanales** de contenido presencial (6 módulos × 2 horas)
- El módulo de **AI** tiene 8 lecturas; semanas 9-10 son revisión y proyecto
- Se recomienda **2-3 horas adicionales** de estudio independiente por módulo
