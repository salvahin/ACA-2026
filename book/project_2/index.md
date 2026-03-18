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

# Project 2 - GPU Kernel Agent

Este modulo guia a los alumnos desde fundamentos de GPU y PyTorch hasta la construccion de un **agente que genera kernels GPU optimizados** usando LLMs. 

## Estructura de Sesiones

Cada sesion sigue el formato: 25' clase + 25' trabajo + 10' discusion + 25' clase + 25' trabajo + 10' cierre.

### Fase 1: Fundamentos GPU (S1-S2)

#### S1: GPU Fundamentals y PyTorch en GPU
**Archivo**: `01_gpu_fundamentals_pytorch.md`
- Arquitectura GPU, SIMT, warps, jerarquia de memoria
- PyTorch tensores en GPU, `.to(device)`, caching allocator, `torch.profiler`

#### S2: CUDA Conceptos y Primer Kernel Triton
**Archivo**: `02_cuda_triton_intro.md`
- Modelo CUDA (indexacion, coalescing, sincronizacion), streams, pinned memory
- Triton: filosofia "bloques no threads", `@triton.jit`, primer kernel elementwise

### Fase 2: Kernels y Optimizacion (S3-S4)

#### S3: Triton Completo y Patrones
**Archivo**: `03_triton_patrones.md`
- API completa `triton.language` (load/store, reducciones, indexacion 2D)
- Kernels fusionados: softmax, layernorm, GELU vs PyTorch

#### S4: Optimizacion de Kernels y Benchmarking
**Archivo**: `04_optimizacion_kernels.md`
- Coalescing, bank conflicts, tiling, roofline model
- `@triton.autotune`, `torch.profiler`, benchmarking correcto

### Fase 3: Evaluacion y Agente (S5-S8)

#### S5: KernelBench y Pipeline de Evaluacion
**Archivo**: `05_kernelbench_evaluacion.md`
- KernelBench L1-L4, tipos de operaciones, metricas
- Pipeline: kernel -> compile -> test vs PyTorch -> benchmark

#### S6: Del Prompt al Agente Kernel
**Archivo**: `06_prompt_a_agente_kernel.md`
- Aplicacion de prompts/gramaticas de P1 al dominio GPU
- Pipeline: LLM genera kernel -> ejecuta -> compara vs PyTorch

#### S7: Agente con Loop de Retroalimentacion
**Archivo**: `07_agente_loop.md`
- Loop agentico: generate -> execute -> verify -> classify error -> regenerate
- Herramientas del agente, estrategias de reintento

#### S8: Debugging como Feedback del Agente
**Archivo**: `08_debugging_feedback.md`
- Taxonomia de errores GPU como componente del agente
- Clasificacion automatica, test suites, feedback estructurado

### Fase 4: Portabilidad y Cierre (S9-S10)

#### S9: SYCL/CUDA Portabilidad y Multi-target
**Archivo**: `09_sycl_portabilidad.md`
- SYCL conceptos (work-items, work-groups, ND-range, buffers/USM)
- Comparacion Triton vs CUDA vs SYCL, portabilidad multi-target

#### S10: Evaluacion Final y Presentaciones
**Archivo**: `10_integracion_final.md`
- Evaluacion final del agente en KernelBench (L1-L3)
- Presentaciones de equipos, checklist de entrega

## Progresion del Reto

```
S1-S2: Entender GPU + escribir kernels basicos a mano
S3-S4: Kernels optimizados a mano + benchmarking
S5:    Pipeline de evaluacion funcionando
S6:    Primer prompt que genera kernels (~50% L1)
S7:    Agente con loop que mejora pass rate (~80% L1, ~40% L2)
S8:    Agente con error feedback (~90% L1, ~60% L2, ~20% L3)
S9:    Extension conceptual a SYCL
S10:   Demo final + reporte
```

## PyTorch como Protagonista

- **S1**: Tensores en GPU, `device`, profiler
- **S2**: CUDA streams, pinned memory, custom ops overview
- **S3**: `torch.softmax`, `torch.nn.LayerNorm` como referencia
- **S4**: `torch.profiler`, CUDA events para benchmarking
- **S5**: PyTorch como ground truth de correctitud en pipeline
- **S6**: El agente genera kernels que reemplazan ops de PyTorch
- **S7**: `torch.allclose()` como verificador, `torch.compile` como baseline
- **S8**: Testing contra PyTorch reference
- **S9**: `torch.compile`, Intel Extension for PyTorch
- **S10**: Integracion final en ecosistema PyTorch

## Requisitos Previos

- Python 3.10+
- PyTorch 2.0+
- Triton
- Acceso a GPU NVIDIA (CUDA 11.8+)
- Conocimientos basicos de Deep Learning
- **Project 1 en curso en paralelo** (prompts, gramaticas, sistemas agenticos)

## Relacion con Otros Modulos

Este modulo (Project 2) complementa:
- **Project 1**: Gramaticas y LLMs para generacion de codigo (en paralelo)
- **AI**: Fundamentos de IA aplicados
- **Compiladores**: Gramaticas formales y parsing

