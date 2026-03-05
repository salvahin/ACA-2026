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

# Patrones Avanzados y Optimización de Kernels GPU


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/05_patrones_avanzados.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 5
> **Tiempo de lectura:** ~50 minutos

---

## Introducción

Ya sabes escribir kernels funcionales. Ahora aprenderás a hacerlos **realmente rápidos**. La optimización GPU requiere entender métricas de rendimiento, patrones de acceso y estrategias que el compilador no puede inferir automáticamente.

Esta lectura cubre técnicas avanzadas: tiling, loop unrolling, autotuning, y cómo usar métricas como ocupancia y roofline para guiar optimizaciones.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Aplicar tiling para mejorar localidad de cache (5-10x speedup típico en matmul)
- Usar loop unrolling (`tl.static_range`) para eliminar overhead de branches
- Configurar autotuning en Triton para encontrar mejores parámetros automáticamente
- Calcular ocupancia y optimizarla ajustando registros y shared memory
- Usar modelo roofline para determinar si optimizar memoria o compute
- Implementar optimizaciones específicas: fusión (memory-bound) vs reducir divergencia (compute-bound)
```

---

## Tiling: Procesamiento Eficiente de Matrices

El tiling es quizás el patrón más importante para rendimiento. La idea: **procesar la memoria en bloques coherentes**.

### ¿Por Qué Tiling?

```
Sin tiling: acceso disperso a memoria
┌─ Thread 0: lee posiciones [0, 1000, 2000, ...]
├─ Thread 1: lee posiciones [1, 1001, 2001, ...]
└─ ...
(Malo: caché misses, sin coalescing)

Con tiling: acceso secuencial agrupado
┌─ Bloque 0: lee posiciones [0-255]
├─ Bloque 1: lee posiciones [256-511]
└─ ...
(Bueno: caché hits, perfecto coalescing)
```

### Ejemplo: Multiplicación de Matrices con Tiling

```{code-cell} ipython3
:tags: [skip-execution]

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Multiplicación A (M x K) @ B (K x N) = C (M x N)
    Con tiling para máxima localidad de caché
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Punto de inicio en salida
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Índices locales
    rm = start_m + tl.arange(0, BLOCK_M)
    rn = start_n + tl.arange(0, BLOCK_N)
    rm = tl.where(rm < M, rm, 0)
    rn = tl.where(rn < N, rn, 0)

    # Acumulador para resultado
    c_accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterar sobre K en bloques
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        rk = tl.where(rk < K, rk, 0)

        # Cargar tiles de A y B
        a_indices = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_tile = tl.load(a_ptr + a_indices)

        b_indices = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_tile = tl.load(b_ptr + b_indices)

        # Multiplicación: A_tile @ B_tile
        c_accum = tl.dot(a_tile, b_tile, c_accum)

    # Guardar resultado
    m_mask = rm[:, None] < M
    n_mask = rn[None, :] < N
    c_mask = m_mask & n_mask
    c_indices = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptr + c_indices, c_accum, mask=c_mask)


def matmul_triton(a, b):
    """Wrapper para multiplicación de matrices"""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
```

### Por Qué Funciona

1. **Caché**: Una vez cargado un tile, múltiples threads lo usan
2. **Coalescing**: Los threads acceden secuencialmente a memoria
3. **Reutilización**: Cada elemento de A y B se accede múltiples veces

La ganancia típica es **5-10x** comparado con acceso no tileado.

---

## Loop Unrolling con tl.static_range

El unrolling elimina sobrecarga de saltos (jumps):

```
Sin unrolling:
┌─ Calcular índice
├─ Ejecutar cuerpo
├─ Incrementar índice
├─ Saltar si no terminó
└─ Repetir 100 veces

Con unrolling:
┌─ Cuerpo 1
├─ Cuerpo 2
├─ ...
└─ Cuerpo 100
(Sin saltos, mucho más rápido)
```

### Uso en Triton

```{code-cell} ipython3
:tags: [skip-execution]

# Versión regular (el compilador genera un loop)
for k in range(0, BLOCK_K):
    # código...
    pass

# Versión desenrollada (el compilador duplica el código)
for k in tl.static_range(0, BLOCK_K):
    # código...
    pass
```

**Regla de oro**: Usa `tl.static_range` solo cuando el rango es conocido en compilación y es pequeño (<1000).

---

## Autotuning: Encontrar los Mejores Parámetros

El rendimiento depende enormemente de `BLOCK_SIZE`, pero ¿cuál es el mejor? **Depende del hardware, del problema, de la memoria**. Triton puede probar automáticamente.

```{code-cell} ipython3
:tags: [skip-execution]

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["n"],  # Retunar si este parámetro cambia
)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

### Cómo Funciona

```
Primera llamada:
1. Triton prueba cada configuración
2. Mide tiempo de ejecución
3. Selecciona la más rápida
4. Cachea resultado

Llamadas siguientes:
→ Usa configuración cacheada
```

### Configuración Avanzada

```{code-cell} ipython3
:tags: [skip-execution]

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256},
            num_warps=8,      # Cuántos warps usar
            num_stages=3,     # Pipeline stages
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def optimized_kernel(a_ptr, b_ptr, M, N,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # kernel code...
    pass
```

---

## Ocupancia (Occupancy)

La ocupancia responde: "¿Qué porcentaje de la capacidad del GPU estoy usando?"

### Cálculo

```
Ocupancia = Threads Activos / Threads Máximo Posible

Ejemplo (NVIDIA A100): 80 SMs, 2048 threads por SM máximo
Máximo global: 80 × 2048 = 163,840 threads

Si lanzas 1000 bloques de 128 threads:
Threads totales: 128,000
Ocupancia: 128,000 / 163,840 = 78%
```

### Por Qué Importa

Baja ocupancia = muchos SMs sin hacer nada. Es como una fábrica medio vacía.

```python
# Bajo ocupancia (malo)
kernel<<<100, 64>>>(...)  # Solo 6,400 threads → 3.9%

# Alto ocupancia (bueno)
kernel<<<10000, 256>>>(...)  # Satura el GPU → 100%
```

### Cómo Mejorar Ocupancia

1. **Aumentar threads por bloque** (hasta 1024)
2. **Lanzar más bloques**
3. **Reducir registros por thread** (menos variables = más threads)
4. **Reducir memoria compartida**

---

## Ancho de Banda de Memoria

El ancho de banda es el **throughput de datos**: cuántos gigabytes por segundo.

### Cálculo Teórico

```
Ancho de banda teórico = Frecuencia × Ancho de bus × Multiplicador

Ejemplo (NVIDIA H100):
- Frecuencia: 1.98 GHz
- Ancho de bus: 384 bits = 48 bytes
- Multiplicador: 2
Máximo: ~3.35 TB/s
```

### Análisis Práctico

```python
# Si transferencia = 1 GB, ancho de banda máximo = 500 GB/s
tiempo_minimo = 1 GB / 500 GB/s = 2 ms

# Si tu kernel tarda 10 ms → memoria es el cuello de botella
# Si tarda 2 ms → memoria está saturada (óptimo)
```

---

## Modelo Roofline

El modelo roofline indica si un kernel está limitado por **compute** o **memoria**:

```
Performance (TFLOPS)
       │
       │     Región Compute-Bound
       │    /
       │   /
       │  /___________
       │               \
       │                \ Región Memory-Bound
       │                 \
       │__________________|_____
       0           Arithmetic Intensity (FLOPs/byte)
```

### Intensidad Aritmética

```
Intensidad = Operaciones Flotantes / Bytes Transferidos
```

| Operación | Intensidad | Bound |
|-----------|------------|-------|
| Vector Add | 0.125 | Memory |
| Softmax | 0.5 | Memory |
| MatMul (naive) | ~0.5 | Memory |
| MatMul (tiled) | ~N | Compute |

### Análisis de Kernel

```{code-cell} ipython3
:tags: [skip-execution]

def analyze_performance(kernel_time_ms, bytes_transferred, flops):
    """Analizar si memory-bound o compute-bound"""
    bandwidth_gb_s = bytes_transferred / (kernel_time_ms * 1e-3) / 1e9
    tflops = flops / (kernel_time_ms * 1e-3) / 1e12
    arithmetic_intensity = flops / bytes_transferred

    max_bandwidth_gb_s = 500  # Ajustar según GPU
    max_tflops = 1456         # Ajustar según GPU

    print(f"Ancho de banda: {bandwidth_gb_s:.1f} GB/s (máx: {max_bandwidth_gb_s})")
    print(f"Throughput: {tflops:.1f} TFLOPS (máx: {max_tflops})")
    print(f"Intensidad: {arithmetic_intensity:.2f} FLOP/byte")

    if bandwidth_gb_s < max_bandwidth_gb_s * 0.8:
        print("⚠ Memory-bound: aumenta intensidad aritmética")
    if tflops < max_tflops * 0.8:
        print("⚠ Compute-bound: necesitas más paralelismo")
```

---

## Optimizaciones por Tipo de Bottleneck

### Para Kernels Memory-Bound

```{code-cell} ipython3
:tags: [skip-execution]

# 1. Fusionar operaciones
# MALO: 3 kernels, 3 round-trips a memoria
y = x + 1
y = y * 2
y = y - 3

# BUENO: 1 kernel fusionado
@triton.jit
def fused(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offsets = ...
    x = tl.load(x_ptr + offsets)
    y = (x + 1) * 2 - 3  # Todo en registros
    tl.store(y_ptr + offsets, y)

# 2. Reutilizar datos cargados
x = tl.load(...)
y = x * 2  # Reutiliza
z = x * 3  # Reutiliza
w = x * 4  # Reutiliza
```

### Para Kernels Compute-Bound

```{code-cell} ipython3
:tags: [skip-execution]

# 1. Usar tensor cores (tl.dot)

# 2. Evitar divergencia
# MALO
if condition:
    x = compute_a()
else:
    x = compute_b()

# BUENO
result = tl.where(condition, compute_a(), compute_b())

# 3. Loop unrolling
for i in tl.static_range(4):  # Compile-time unroll
    acc += compute(i)
```

### Optimizaciones Generales

```{code-cell} ipython3
:tags: [skip-execution]

# 1. Coalescing de memoria
# MALO: Stride grande
x = tl.load(x_ptr + threadIdx * 1000)

# BUENO: Stride pequeño
x = tl.load(x_ptr + threadIdx)

# 2. Reducir registros
# MALO: Muchas variables temporales
a = tl.load(...)
b = a + 1
c = b * 2
d = tl.sqrt(c)

# BUENO: Reutilizar
a = tl.load(...)
a = a + 1
a = a * 2
a = tl.sqrt(a)
```

---

## Profiling: Midiendo Rendimiento

### Con PyTorch

```{code-cell} ipython3
:tags: [skip-execution]

def benchmark_kernel(kernel_fn, *args, num_iterations=100):
    """Benchmark de un kernel Triton"""
    # Warm-up
    kernel_fn(*args)
    torch.cuda.synchronize()

    # Medir
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        kernel_fn(*args)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iterations
```

### Con Nsight Compute

```bash
# Perfilar programa CUDA
ncu --set full ./mi_programa

# Métricas clave:
# - SM Utilization: % de SMs ocupados
# - Memory Bandwidth: ancho de banda usado
# - L1 Cache Hit Rate: eficiencia de caché
```

---

## Proceso de Optimización Iterativa

```
1. Implementar (¿Funciona?)
   ↓
2. Medir (¿Qué métrica es baja?)
   ↓
3. Identificar cuello de botella
   ├─ Memory-bound? → Fusionar ops, reutilizar datos
   ├─ Compute-bound? → Más threads, tensor cores
   ├─ Baja ocupancia? → Menos registros, más bloques
   └─ Sin coalescing? → Reorganizar accesos
   ↓
4. Optimizar
   ↓
5. Volver a medir
   ↓
6. ¿Mejor? → Repetir o continuar
```

---

## Resumen de Técnicas

```{admonition} Resumen
:class: important
**Técnicas de optimización y cuándo aplicarlas:**

| Técnica | Cuándo usar | Beneficio típico | Dificultad |
|---------|-------------|-----------------|------------|
| **Tiling** | Matrices, convolutions | 5-10x | Media |
| **Unrolling** | Loops pequeños (<100 iter) | 10-20% | Baja |
| **Autotuning** | Múltiples configs posibles | Óptimo automático | Baja |
| **Fusión** | Ops consecutivas memory-bound | 2-4x | Media |
| **num_warps** | Ajustar ocupancia | 20-50% | Baja |

**Checklist de optimización avanzada:**
- [ ] Tiling aplicado donde hay reutilización de datos
- [ ] Autotuning configurado para BLOCK_SIZE y num_warps
- [ ] Occupancy calculado (>50% para memory-bound, >75% para compute-bound)
- [ ] Roofline analizado (¿cerca del límite teórico?)
- [ ] Operaciones fusionadas donde sea posible
```

```{admonition} ⚡ Tip de Performance: Orden de Optimización
:class: important
**Prioriza optimizaciones en este orden** (mayor impacto primero):
1. **Algoritmo correcto** (10-100x): Tiling para matmul, online softmax, etc.
2. **Fusión de kernels** (2-4x): Eliminar round-trips a memoria
3. **Autotuning** (1.2-2x): Encontrar mejor BLOCK_SIZE
4. **Loop unrolling** (1.1-1.2x): Reducir overhead de control
5. **Fine-tuning num_warps** (1.05-1.3x): Ajustar ocupancia

No gastes tiempo en (5) si no has hecho (1)-(3).
```

```{admonition} 🎯 En tu proyecto
:class: note
Aplicarás estas técnicas cuando optimices KernelBench:
- **L1 (básico)**: Solo corrección, no optimizaciones complejas
- **L2 (intermedio)**: Fusión simple de 2-3 operaciones
- **L3 (avanzado)**: Tiling + autotuning + análisis roofline
- **L4 (experto)**: Todos los trucos + optimizaciones custom

Para maximizar puntos en tiempo limitado: resuelve todos los L1 correctamente antes de intentar optimizar L3.
```

---

## Ejercicios

### Ejercicio 1: Análisis Roofline

Para un GPU con 10 TFLOPS peak y 500 GB/s bandwidth:
1. ¿Cuál es el ridge point?
2. ¿Vector addition (AI=0.125) es memory o compute bound?
3. ¿MatMul 1024x1024 (AI≈1024)?

### Ejercicio 2: Identificar Bottleneck

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def mystery_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    for _ in range(1000):
        x = tl.sqrt(tl.exp(x * 2.5) + 1.0)
    tl.store(y_ptr + offsets, x, mask=mask)
```

¿Memory-bound o compute-bound? ¿Por qué?

### Para Pensar

> *Si un kernel está al 90% del roofline pero sigue siendo lento, ¿qué opciones tienes?*

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- NVIDIA. [Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/). NVIDIA Docs.
- Triton. [Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Triton Docs.
- Hwu, W., Kirk, D., & El Hajj, I. (2022). Programming Massively Parallel Processors (4th ed.). Morgan Kaufmann.
