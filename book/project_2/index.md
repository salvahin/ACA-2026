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

# Project 2: GPU Kernel Agent

En este proyecto construirás un agente que genera kernels GPU optimizados usando LLMs. Partirás desde los fundamentos de arquitectura GPU y PyTorch, aprenderás a escribir kernels Triton eficientes, y desarrollarás un sistema agéntico con loop de retroalimentación para mejorar iterativamente la calidad del código generado.

## Objetivos de Aprendizaje

Al completar este proyecto, serás capaz de:

- [ ] Explicar la arquitectura GPU (SIMT, warps, jerarquía de memoria)
- [ ] Escribir kernels Triton optimizados para operaciones comunes (softmax, layernorm)
- [ ] Aplicar técnicas de optimización (coalescing, tiling, roofline model)
- [ ] Implementar un pipeline de evaluación usando KernelBench
- [ ] Construir un agente con loop de retroalimentación para generación de kernels
- [ ] Clasificar y usar errores de compilación/ejecución como feedback

## Prerequisitos

- Python 3.10+
- PyTorch 2.0+
- Acceso a GPU NVIDIA (CUDA 11.8+)
- Conocimientos de Deep Learning básicos
- **Project 1 en paralelo** (prompts, gramáticas, sistemas agénticos)

## Temas por Sesión

| Sesión | Tema | Descripción |
|:------:|------|-------------|
| 1 | [GPU Fundamentals y PyTorch](01_gpu_fundamentals_pytorch) | Arquitectura GPU, tensores en GPU, torch.profiler |
| 2 | [CUDA y Primer Kernel Triton](02_cuda_triton_intro) | Modelo CUDA, @triton.jit, kernel elementwise |
| 3 | [Triton Patrones](03_triton_patrones) | API triton.language, kernels fusionados |
| 4 | [Optimización y Benchmarking](04_optimizacion_kernels) | Coalescing, tiling, @triton.autotune |
| 5 | [KernelBench Evaluación](05_kernelbench_evaluacion) | Pipeline de evaluación, métricas |
| 6 | [Del Prompt al Agente](06_prompt_a_agente_kernel) | Aplicación de prompts/gramáticas al dominio GPU |
| 7 | [Agente con Loop](07_agente_loop) | Generate → Execute → Verify → Classify → Retry |
| 8 | [Debugging como Feedback](08_debugging_feedback) | Taxonomía de errores, feedback estructurado |
| 9 | [SYCL y Portabilidad](09_sycl_portabilidad) | Multi-target, comparación Triton vs CUDA vs SYCL |
| 10 | [Evaluación Final](10_integracion_final) | Demo final, presentaciones, checklist de entrega |

## Relación con Otros Módulos

```{admonition} Conexiones
:class: tip

- **← Project 1:** Gramáticas y LLMs para generación de código (en paralelo)
- **← AI:** Fundamentos de IA aplicados
- **← Compilers:** Gramáticas formales para restricciones de sintaxis
- **→ Stats:** Análisis estadístico de benchmarks y experimentos
```
