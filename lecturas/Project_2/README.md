# Project 2 - GPU Computing & Kernel Optimization

Conjunto completo de 10 lecturas para el módulo "Grammar-Constrained GPU Kernel Generation - Project 2".

Este módulo está diseñado para un profesor experto en GPU/PyTorch (perfil Intel AI Engineer) y cubre desde fundamentos de GPU hasta optimización avanzada de kernels con Triton.

## Estructura de Lecturas

### Semana 1: GPU Fundamentals (01_GPU_Fundamentals.md)
**Tema**: Arquitectura GPU, SIMT, jerarquía de memoria
**Concepto Clave**: GPUs no son CPUs rápidas - son máquinas paralelas masivas
**Contenido**:
- CPU vs GPU: diferentes filosofías de diseño
- Modelo de ejecución SIMT y warps
- Jerarquía: Grid → Blocks → Threads → Warps
- Jerarquía de memoria: registros, shared, global
- Impacto económico de GPU computing

### Semana 2: CUDA y PyTorch Internals (02_CUDA_PyTorch_Internals.md)
**Tema**: CUDA basics, PyTorch GPU internals
**Concepto Clave**: Entender qué hace PyTorch bajo el capó
**Contenido**:
- Modelo de programación CUDA
- Indexación de threads y coalescing
- PyTorch tensors y memoria GPU
- CUDA caching allocator
- CUDA streams y profiling

### Semana 3: Optimización de Memoria (03_Memoria_Optimizacion.md)
**Tema**: Coalescing, bank conflicts, tiling, roofline
**Concepto Clave**: La diferencia entre kernel lento y rápido está en cómo accede a memoria
**Contenido**:
- Patrones de acceso coalesced
- Bank conflicts en shared memory
- Tiling para mejorar localidad
- Modelo roofline: memory vs compute bound

### Semana 4: Triton Completo (04_Triton_Completo.md)
**Tema**: Filosofía Triton, API completa, reducciones
**Concepto Clave**: Piensa en bloques, no en threads
**Contenido**:
- Filosofía Triton vs CUDA
- API de triton.language completa
- Reducciones locales y globales (dos etapas)
- Autotuning con @triton.autotune
- Ejemplos: softmax, layernorm, GELU

### Semana 5: Patrones Avanzados (05_Patrones_Avanzados.md)
**Tema**: Tiling, unrolling, autotuning, optimización
**Concepto Clave**: Del kernel correcto al kernel rápido
**Contenido**:
- Tiling para matrices
- Loop unrolling con tl.static_range
- Autotuning de parámetros
- Ocupancia y num_warps/num_stages
- Profiling y diagnóstico

### Semana 6: KernelBench y Corpus (06_KernelBench_Corpus.md)
**Tema**: Benchmark structure, corpus design, evaluation
**Concepto Clave**: Medir correctitud Y velocidad
**Contenido**:
- Estructura de KernelBench L1/L2/L3/L4
- Tipos de operaciones (element-wise, reduction, complex)
- Diseño de corpus de benchmarks
- Métricas: correctness, time, speedup
- Estrategias de selección informada

### Semana 7: Debugging y Taxonomía (07_Debugging_Taxonomia.md)
**Tema**: Error classification, debugging techniques
**Concepto Clave**: Debugging GPU es un arte
**Contenido**:
- Taxonomía de errores (compilación, runtime, correctitud, performance)
- Errores comunes: bounds, sync, indexing, dtype
- Metodología: reproducir → clasificar → aislar → verificar
- Test suites para kernels generados
- Documentación de errores

### Semana 8: Baseline y Experimentos (08_Baseline_Experimentos.md)
**Tema**: Benchmarking, experiment design A/B/C/D
**Concepto Clave**: Antes de optimizar, medir dónde estás
**Contenido**:
- Establecer baselines confiables
- Benchmarking correcto (warmup, sync, stats)
- Diseño de experimentos A/B/C/D
- Control de variables y randomización
- Análisis estadístico (ANOVA, t-tests, Cohen's d)
- Automatización y CI/CD

### Semana 9: Análisis y Visualización (09_Analisis_Visualizacion.md)
**Tema**: Data analysis, visualization, reporting
**Concepto Clave**: Los datos visualizados cuentan historias
**Contenido**:
- Análisis exploratorio de benchmarks
- Visualizaciones: boxplots, scaling, heatmaps, roofline
- Detección de anomalías y patrones
- Reportes ejecutivos y técnicos
- Comunicación para diferentes audiencias

### Semana 10: Integración y Documentación (10_Integracion_Documentacion.md)
**Tema**: Grammar-GPU integration, reproducibility
**Concepto Clave**: Cierre del círculo - de GPU a generación automática
**Contenido**:
- Cómo gramáticas capturan restricciones GPU
- Límites de generación automática
- Documentación profesional
- Reproducibilidad de experimentos
- Reporte final y checklist

## Características Comunes

Cada lectura incluye:

- **Ejemplos de Código**: Ejemplos prácticos en Triton y PyTorch
- **Ejercicios**: Tareas prácticas al final
- **Preguntas de Reflexión**: Para pensamiento crítico
- **Extensión**: ~30-50 minutos de lectura
- **Español**: Completamente en español
- **Markdown**: Formateado para fácil lectura

## Progresión Pedagógica

```
Semanas 1-2: Fundamentos
├─ Arquitectura GPU
└─ CUDA y PyTorch internals

Semanas 3-4: Programación Triton
├─ Optimización de memoria
└─ API y patrones Triton

Semanas 5-6: Patrones y Evaluación
├─ Patrones avanzados
└─ KernelBench y corpus

Semanas 7-8: Debugging y Experimentación
├─ Taxonomía de errores
└─ Diseño experimental

Semanas 9-10: Análisis y Cierre
├─ Visualización
└─ Integración y documentación
```

## Archivos

```
01_GPU_Fundamentals.md          (~15 KB)
02_CUDA_PyTorch_Internals.md    (~12 KB)
03_Memoria_Optimizacion.md      (~10 KB)
04_Triton_Completo.md           (~11 KB)
05_Patrones_Avanzados.md        (~13 KB)
06_KernelBench_Corpus.md        (~14 KB)
07_Debugging_Taxonomia.md       (~15 KB)
08_Baseline_Experimentos.md     (~13 KB)
09_Analisis_Visualizacion.md    (~17 KB)
10_Integracion_Documentacion.md (~11 KB)
README.md                       (Este archivo)
```

**Total**: ~130 KB de contenido técnico

## Requisitos Previos

- Python 3.10+
- PyTorch 2.0+
- Triton
- Acceso a GPU NVIDIA (CUDA 11.8+)
- Conocimientos básicos de Deep Learning

## Relación con Otros Módulos

Este módulo (Project 2) complementa:
- **Project 1**: Gramáticas y LLMs para generación de código
- **Stats**: Análisis estadístico de resultados
- **AI**: Fundamentos de IA aplicados

---

**Última actualización**: Marzo 2026
**Versión**: 2.0 (Fusionado con contenido GPU)
**Idioma**: Español
**Nivel**: Intermedio-Avanzado
