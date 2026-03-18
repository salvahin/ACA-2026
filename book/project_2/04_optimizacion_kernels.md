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

# Optimización de Kernels y Benchmarking

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q torch triton
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/04_optimizacion_kernels.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 4
> **Tiempo de lectura:** ~45 minutos

---

## Introducción

La diferencia entre un kernel GPU lento y uno rápido frecuentemente está en **cómo accede a la memoria** y **cómo está configurado**. Un kernel puede tener la lógica correcta pero ser 10x más lento por patrones de acceso ineficientes o parámetros subóptimos.

Esta lectura te enseña técnicas avanzadas de optimización: coalesced access, bank conflicts, tiling, autotuning, y cómo medir rendimiento correctamente para guiar tus decisiones.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Diseñar patrones de acceso coalesced (threads consecutivos → direcciones consecutivas)
- Evitar bank conflicts en shared memory usando padding
- Aplicar tiling para mejorar localidad de cache (5-10x speedup típico en matmul)
- Usar el modelo roofline para determinar si tu kernel es memory-bound o compute-bound
- Configurar autotuning en Triton para encontrar mejores parámetros automáticamente
- Medir rendimiento correctamente con warmup, sincronización y CUDA events
```

---

## Bloque 1: Optimización de Kernels

### Coalesced Memory Access

```{admonition} 🧠 Modelo Mental: Coalescing
:class: hint
Piensa en la memoria GPU como un autobús que transporta 128 bytes a la vez:
- **Coalesced**: 32 pasajeros suben en orden (1 viaje)
- **No coalesced**: 32 pasajeros en diferentes paradas (32 viajes)

Mismo número de pasajeros, 32x diferencia de tiempo.
```

#### El Problema

La GPU accede a memoria global en **transacciones de 128 bytes**. Si 32 threads de un warp acceden a direcciones dispersas, necesitas múltiples transacciones:

```
Acceso Coalesced (1 transacción):
Thread 0 → addr[0]
Thread 1 → addr[1]
...
Thread 31 → addr[31]
→ Todos en el mismo bloque de 128 bytes = 1 transacción

Acceso No-Coalesced (32 transacciones):
Thread 0 → addr[0]
Thread 1 → addr[1000]
...
Thread 31 → addr[31000]
→ Cada acceso en bloque diferente = 32 transacciones = 32x más lento!
```

#### Patrones Correctos en Triton

```{code-cell} ipython3
:tags: [skip-execution]

# BUENO: Acceso coalesced por filas
@triton.jit
def kernel_good(data_ptr, N, M, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)  # 0, 1, 2, ..., BLOCK-1

    # Threads consecutivos acceden a posiciones consecutivas
    offsets = row * M + cols  # [row*M, row*M+1, row*M+2, ...]
    x = tl.load(data_ptr + offsets)

# MALO: Acceso strided por columnas
@triton.jit
def kernel_bad(data_ptr, N, M, BLOCK: tl.constexpr):
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK)

    # Threads consecutivos acceden con stride M
    offsets = rows * M + col  # [col, M+col, 2M+col, ...]
    x = tl.load(data_ptr + offsets)  # No-coalesced!
```

#### Visualización

```
Matriz en memoria (row-major):
[a00][a01][a02][a03][a10][a11][a12][a13][a20][a21]...

Acceso por fila (coalesced):
Thread 0 lee a00, Thread 1 lee a01, Thread 2 lee a02...
→ Posiciones consecutivas en memoria ✓

Acceso por columna (no-coalesced):
Thread 0 lee a00, Thread 1 lee a10, Thread 2 lee a20...
→ Saltos de M posiciones entre accesos ✗
```

:::{figure} diagrams/memory_coalescing.png
:name: fig-memory-coalescing
:alt: Patrón de acceso coalesced (1 transacción) vs no-coalesced (múltiples transacciones) a memoria global
:align: center
:width: 90%

**Figura 1:** Coalesced Memory Access - Threads consecutivos accediendo posiciones consecutivas resulta en una sola transacción; acceso disperso requiere múltiples transacciones.
:::

---

### Shared Memory y Bank Conflicts

#### Estructura de Banks

Shared memory está dividida en **32 banks**. Cada bank puede servir una solicitud por ciclo:

```
Bank 0: addr 0, 32, 64, 96, ...
Bank 1: addr 1, 33, 65, 97, ...
...
Bank 31: addr 31, 63, 95, 127, ...
```

#### Bank Conflict

Cuando múltiples threads acceden al mismo bank (diferente dirección), hay **serialización**:

```
Sin conflict (cada thread → diferente bank):
Thread 0 → Bank 0
Thread 1 → Bank 1
...
→ 1 ciclo

2-way conflict (2 threads → mismo bank):
Thread 0 → Bank 0
Thread 1 → Bank 0  ← Conflicto!
→ 2 ciclos

32-way conflict (todos → mismo bank):
Thread 0-31 → Bank 0
→ 32 ciclos!
```

#### Solución: Padding

```{code-cell} ipython3
:tags: [skip-execution]

# MALO: Stride de 32 (todos al mismo bank)
shared_mem[threadIdx.x * 32]

# BUENO: Stride de 1 (acceso lineal)
shared_mem[threadIdx.x]

# SOLUCIÓN: Padding para evitar conflicts
shared_mem[row * 33 + col]  # 33 en vez de 32
```

---

### Tiling: Mejorando Localidad

#### Concepto

**Tiling** divide el problema en bloques pequeños que caben en cache/shared memory:

```
Sin tiling:
Cada thread carga desde global memory
→ Muchos accesos a memoria lenta

Con tiling:
1. Cargar tile a shared memory
2. Sincronizar
3. Procesar desde shared memory
→ Menos accesos a global, más a shared (rápida)
```

#### Ejemplo: Matrix Multiply con Tiling

```{code-cell} ipython3
:tags: [skip-execution]

import triton
import triton.language as tl

@triton.jit
def matmul_tiled(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Multiplicación A (M x K) @ B (K x N) = C (M x N)
    Con tiling para máxima localidad de caché
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterar sobre tiles de K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Cargar tiles de A y B
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])

        # Multiplicación en registros (muy rápido)
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

:::{figure} diagrams/tiling_strategy.png
:name: fig-tiling-strategy
:alt: Estrategia de tiling: dividir problema en tiles que caben en cache, procesarlos localmente, y combinar resultados
:align: center
:width: 90%

**Figura 2:** Estrategia de Tiling - Dividir matrices grandes en tiles pequeños que caben en Shared Memory (rápida), minimizando accesos a Global Memory.
:::

#### Beneficios

```
Sin tiling (matmul MxNxK):
- Accesos a global: M*N*K * 2
- Arithmetic intensity: 2 FLOPs / 8 bytes = 0.25

Con tiling (block size = B):
- Accesos a global: M*N*K*2 / B
- Arithmetic intensity: B/4

Para B=32: 8x mejor arithmetic intensity
```

---

### Modelo Roofline

#### Concepto

El **modelo roofline** indica si un kernel está limitado por **compute** o **memoria**:

```
Performance (FLOPS)
       │
       │         ╱ Roofline
       │        ╱
       │       ╱............ Peak Compute
       │      ╱
       │     ╱
       │    ╱
       │   ╱
       │  ╱
       │ ╱
       │╱
       └────────────────────────
         Arithmetic Intensity (FLOPS/byte)

Ridge Point: donde memoria = compute
- Izquierda: Memory-bound
- Derecha: Compute-bound
```

#### Cálculo

```{code-cell} ipython3
:tags: [skip-execution]

# Datos GPU (ejemplo A100)
peak_compute = 19.5e12  # 19.5 TFLOPS (FP32)
peak_bandwidth = 2.0e12  # 2 TB/s

ridge_point = peak_compute / peak_bandwidth  # ~10 FLOPS/byte

def analyze_kernel(flops, bytes_transferred):
    ai = flops / bytes_transferred

    if ai < ridge_point:
        print(f"Memory-bound. Max: {peak_bandwidth * ai / 1e12:.2f} TFLOPS")
    else:
        print(f"Compute-bound. Max: {peak_compute / 1e12:.2f} TFLOPS")
```

:::{figure} diagrams/roofline_model.png
:name: fig-roofline-model
:alt: Modelo Roofline: Performance vs Arithmetic Intensity, con límites de memoria y compute definidos
:align: center
:width: 90%

**Figura 3:** Modelo Roofline - Visualiza el rendimiento máximo alcanzable según la intensidad aritmética. Ridge Point separa kernels memory-bound de compute-bound.
:::

#### Operaciones Comunes

| Operación | Arithmetic Intensity | Bound Type |
|-----------|---------------------|------------|
| Vector Add | 0.08 FLOP/byte | Memory |
| Elementwise | 0.125 FLOP/byte | Memory |
| MatMul (naive) | ~0.5 FLOP/byte | Memory |
| MatMul (tiled) | ~8 FLOP/byte | Compute |
| Softmax | 0.4 FLOP/byte | Memory |

---

## Actividad de Trabajo — Bloque 1 (25 minutos)

**Ejercicio 1: Analizar patrones de acceso a memoria (10 min)**

Antes de ejecutar, **predice** cuántas transacciones de 128 bytes genera cada patrón
y si es coalescido. Luego ejecuta y verifica:

```{code-cell} ipython3
def predict_memory_transactions(access_pattern, warp_size=32, transaction_size=128):
    elements_per_transaction = transaction_size // 4  # 4 bytes por float
    addresses = [access_pattern(tid) for tid in range(warp_size)]
    blocks = set(addr // elements_per_transaction for addr in addresses)
    n = len(blocks)
    return {"num_transactions": n,
            "efficiency": warp_size / n if n else 0,
            "is_coalesced": n == 1,
            "addresses": addresses[:8]}

patterns = [
    ("tid (coalesced)",         lambda tid: tid),
    ("tid * 2",                  lambda tid: tid * 2),
    ("tid * 32",                 lambda tid: tid * 32),
    ("tid * 128  (por columnas)", lambda tid: tid * 128),
]
for label, pat in patterns:
    r = predict_memory_transactions(pat)
    print(f"{label:28s} → {r['num_transactions']:3d} tx  "
          f"eficiencia={r['efficiency']:.0%}  {'✓' if r['is_coalesced'] else '✗'}")
```

Añade un quinto patrón `lambda tid: (tid % 16) * 4` — predice antes de correrlo.

**Ejercicio 2: Modelo Roofline (10 min)**

Para una GPU con peak compute = 10 TFLOPS y bandwidth = 500 GB/s:

a) Calcula el ridge point (FLOP/byte).

b) Para cada kernel calcula su arithmetic intensity y determina si es memory-bound
o compute-bound:

| Kernel | FLOPs/elem | Bytes leídos+escritos/elem |
|--------|-----------|----------------------------|
| Vector add `c = a + b` | 1 | 12 |
| Softmax fusionado | 5 | 8 |
| MatMul BLOCK=64 | `2×64³` | `2×64²×4` bytes |

c) ¿Cuál es la performance máxima teórica de cada uno (en TFLOPS)?

**Ejercicio 3: Diagnóstico de acceso 2D (5 min)**

Identifica si el acceso a `mat_ptr` es coalescido y justifica:

```python
@triton.jit
def kernel_col(mat_ptr, out_ptr, N, BLOCK: tl.constexpr):
    col = tl.program_id(0)          # un programa por columna
    rows = tl.arange(0, BLOCK)
    offsets = rows * N + col        # thread 0→col, thread 1→N+col ...
    x = tl.load(mat_ptr + offsets, mask=rows < N, other=0.0)
```

¿Qué cambio mínimo en `offsets` haría el acceso coalescido?

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

1. **Regla del stride**: En el Ejercicio 1, ¿a partir de qué stride el acceso deja de
   ser coalescido? ¿Por qué `tid * 2` es más eficiente que `tid * 32` aunque ninguno
   sea perfectamente coalescido?
2. **Ridge point y acciones**: Si tu kernel tiene AI = 2 FLOP/byte y el ridge = 10,
   solo optimizar el cómputo no ayuda. ¿Qué técnica concreta de las vistas hoy
   aumenta la AI sin cambiar la lógica del kernel?
3. **Coalescing 2D**: En el Ejercicio 3, si `N = 1024`, ¿cuántas transacciones
   genera un warp de 32 threads? ¿Y si `N = 32`? ¿Por qué el segundo es aceptable?
```

---

## Bloque 2: Autotuning y Benchmarking

### Autotuning con Triton

El rendimiento depende enormemente de `BLOCK_SIZE`, pero ¿cuál es el mejor? **Depende del hardware, del problema, de la memoria**. Triton puede probar automáticamente.

#### Configuración Básica

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

#### Cómo Funciona

```
Primera llamada:
1. Triton prueba cada configuración
2. Mide tiempo de ejecución
3. Selecciona la más rápida
4. Cachea resultado

Llamadas siguientes:
→ Usa configuración cacheada (no retuning overhead)
```

#### Configuración Avanzada

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
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=2,
            num_stages=4,
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

### Occupancy: Uso Eficiente de la GPU

La ocupancia responde: "¿Qué porcentaje de la capacidad del GPU estoy usando?"

#### Cálculo

```
Ocupancia = Threads Activos / Threads Máximo Posible

Ejemplo (NVIDIA A100): 80 SMs, 2048 threads por SM máximo
Máximo global: 80 × 2048 = 163,840 threads

Si lanzas 1000 bloques de 128 threads:
Threads totales: 128,000
Ocupancia: 128,000 / 163,840 = 78%
```

#### Por Qué Importa

Baja ocupancia = muchos SMs sin hacer nada. Es como una fábrica medio vacía.

```python
# Bajo ocupancia (malo)
kernel<<<100, 64>>>(...)  # Solo 6,400 threads → 3.9%

# Alto ocupancia (bueno)
kernel<<<10000, 256>>>(...)  # Satura el GPU → 100%
```

#### Cómo Mejorar Ocupancia

1. **Aumentar threads por bloque** (hasta 1024)
2. **Lanzar más bloques**
3. **Reducir registros por thread** (menos variables = más threads)
4. **Reducir memoria compartida**

---

### Profiling con torch.profiler

```{code-cell} ipython3
:tags: [skip-execution]

import torch

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_flops=True
) as prof:
    my_kernel(...)

for event in prof.key_averages():
    if event.flops > 0:
        achieved = event.flops / (event.cuda_time * 1e-6)
        print(f"{event.key}: {achieved/1e12:.2f} TFLOPS")
```

---

### Benchmarking Correcto

#### La Importancia del Warmup

```{admonition} ⚠️ Warmup Crítico
:class: warning

**Sin warmup**, la primera ejecución incluye:
- JIT compilation de Triton (~50-200ms)
- CUDA initialization (~10-50ms)
- GPU frequency scaling (~10-20ms)

**Resultado**: Mediciones 10-100x más lentas que la realidad.

**Solución**: Siempre hacer warmup de 10-20 iteraciones antes de medir.
```

#### Estructura de Benchmark

```{code-cell} ipython3
:tags: [skip-execution]

def benchmark_kernel(kernel_fn, *inputs, warmup_iters=10, bench_iters=100):
    """Mide performance de un kernel correctamente."""

    # WARMUP CRÍTICO: Primera ejecución incluye JIT compilation + CUDA init +
    # GPU frequency scaling. Sin warmup: mediciones 25x más lentas.
    # Mínimo 10-20 iteraciones antes de medir.
    for _ in range(warmup_iters):
        _ = kernel_fn(*inputs)
    torch.cuda.synchronize()

    # Benchmark con CUDA events (más preciso que time.time())
    times = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = kernel_fn(*inputs)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import numpy as np
    times = np.array(times)
    return {
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "min_ms": times.min(),
        "max_ms": times.max(),
    }
```

#### Errores Comunes

```{code-cell} ipython3
:tags: [skip-execution]

# ERROR 1: No sincronizar
start = time.time()
output = kernel(x)
end = time.time()  # Mide lanzamiento, no ejecución ✗

# CORRECTO:
torch.cuda.synchronize()
start = time.time()
output = kernel(x)
torch.cuda.synchronize()
end = time.time()

# ERROR 2: No hacer warmup
# Primeras iteraciones incluyen JIT compilation ✗

# ERROR 3: Incluir overhead de Python
for i in range(100):
    result = kernel(input_list[i])  # List indexing cada vez ✗
```

#### Uso de CUDA Events

```{code-cell} ipython3
:tags: [skip-execution]

# Por qué CUDA events son mejores que time.time()
# 1. Medición en GPU (no overhead de CPU-GPU sync)
# 2. Resolución en microsegundos
# 3. No afectados por context switches del OS

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# Tu kernel aquí
kernel(...)
end_event.record()

torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

---

## Actividad de Trabajo — Bloque 2 (25 minutos)

**Ejercicio 1: Autotuning + Benchmark completo (15 min)**

Toma el kernel `softmax_triton` o `vector_add` de la Sesión 3:

1. Añade `@triton.autotune` con las siguientes configuraciones:

```python
configs=[
    triton.Config({"BLOCK_SIZE": 128},  num_warps=4),
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
]
```

2. Usa `benchmark_kernel` (definido arriba) con `warmup_iters=10, bench_iters=100`.
3. Reporta: configuración ganadora, `mean ± std ms`, speedup vs PyTorch nativo.

Discute: ¿coincide la config ganadora con la mayor occupancy? ¿Por qué podría no coincidir?

**Ejercicio 2: Diagnóstico del `mystery_kernel` (10 min)**

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

1. Estima su arithmetic intensity. ¿Memory-bound o compute-bound?
2. ¿El autotuning de `BLOCK_SIZE` mejoraría significativamente el kernel? Justifica.
3. Propón una optimización concreta (`num_warps`, `num_stages`, fusión con otra op).

---

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: Tus 32 threads acceden `data[tid * 64]`. ¿Cuántas transacciones de 128 bytes genera el warp? ¿Cómo lo corriges?
<details><summary>Respuesta</summary>
Stride de 64 × 4 bytes = 256 bytes entre threads consecutivos → cada thread cae en un
bloque de 128 bytes distinto → 32 transacciones. Corrección: usar
<code>data[pid * BLOCK + tid]</code> para que offsets sean consecutivos.
</details>

**Pregunta 2**: ¿Por qué `@triton.autotune` no retuneea en cada llamada?
<details><summary>Respuesta</summary>
Cachea la configuración ganadora indexada por los valores declarados en <code>key=[]</code>.
Mientras esos valores no cambien, reutiliza la configuración sin overhead de prueba.
Un nuevo valor de <code>n</code> (o lo que sea el key) dispara un ciclo nuevo de benchmarking
solo para ese tamaño.
</details>
```

### Para Pensar

> *Si tu kernel ya está al 90% del techo del roofline pero la aplicación sigue siendo
> lenta, ¿qué opciones tienes más allá de optimizar el kernel en sí?*
>
> *Pista: piensa en lo que ocurre antes y después del kernel...*

---

## Resumen

```{admonition} Resumen
:class: important

**Conceptos clave de optimización:**

1. **Coalesced Access**: Threads consecutivos → direcciones consecutivas (1 transacción vs 32)
2. **Bank Conflicts**: Evitar múltiples threads al mismo bank (usa padding: stride=33 en vez de 32)
3. **Tiling**: Cargar bloques a shared memory para reutilizar datos (5-10x speedup típico)
4. **Roofline**: Memory-bound (<10 FLOP/byte) vs compute-bound (>10 FLOP/byte)
5. **Autotuning**: Triton prueba configs automáticamente y cachea la mejor
6. **Occupancy**: Más threads activos = mejor uso de GPU (target >50% para memory-bound)

**Proceso de benchmarking correcto:**
1. Warmup de 10-20 iteraciones (elimina JIT overhead)
2. Usar `torch.cuda.synchronize()` antes y después de timing
3. Usar CUDA events para medición precisa
4. Múltiples samples (100+) para estadística robusta
5. Reportar media ± std para cuantificar variabilidad
```

```{admonition} Checklist de Optimización
:class: tip

**Antes de optimizar:**
- [ ] Perfilar con `torch.profiler` para identificar bottleneck
- [ ] Calcular arithmetic intensity (memory-bound vs compute-bound)
- [ ] Establecer baseline con PyTorch nativo

**Durante optimización:**
- [ ] Memory coalescing verificado (threads consecutivos → direcciones consecutivas)
- [ ] Shared memory sin bank conflicts (padding donde necesario)
- [ ] Tiling aplicado donde hay reutilización de datos
- [ ] Autotuning configurado para BLOCK_SIZE y num_warps

**Después de optimizar:**
- [ ] Benchmark con warmup + sync + CUDA events
- [ ] Speedup medido vs baseline (con std)
- [ ] Occupancy calculado (>50% para memory-bound)
- [ ] Roofline verificado (¿cerca del límite teórico?)
```

---

## Tarea para Casa

### Ejercicio: Benchmark Comparativo Completo

Elige un kernel de las sesiones anteriores (softmax, layernorm o GELU) e implementa:

1. **Versión baseline** — PyTorch nativo.
2. **Versión Triton** con `@triton.autotune` (mínimo 4 configuraciones).
3. **Benchmark correcto** usando `benchmark_kernel` con `warmup_iters=20, bench_iters=200`.
4. **Análisis roofline**: calcula la arithmetic intensity de tu kernel, ubícalo en el
   roofline de tu GPU y compara el speedup obtenido con el speedup teórico máximo esperado.

Entregable: notebook con código limpio + tabla resumen + conclusión de 3-5 líneas explicando
por qué el speedup real es mayor o menor que el teórico.

---

### Referencia: Herramienta de análisis de accesos (para usar en tarea)

### Ejercicio de referencia: Predecir Transacciones de Memoria

```{code-cell} ipython3
def predict_memory_transactions(access_pattern, warp_size=32, transaction_size=128):
    """Predice el número de transacciones de memoria para un patrón de acceso."""

    # Cada transacción cubre 128 bytes = 32 floats (4 bytes cada uno)
    bytes_per_element = 4
    elements_per_transaction = transaction_size // bytes_per_element

    # Simular accesos de un warp
    addresses = [access_pattern(thread_id) for thread_id in range(warp_size)]

    # Agrupar en bloques de 128 bytes
    blocks = set()
    for addr in addresses:
        block_id = addr // elements_per_transaction
        blocks.add(block_id)

    num_transactions = len(blocks)

    return {
        "addresses": addresses[:8],  # Mostrar solo primeros 8
        "unique_blocks": sorted(list(blocks))[:8],
        "num_transactions": num_transactions,
        "efficiency": (warp_size / num_transactions) if num_transactions > 0 else 0,
        "is_coalesced": num_transactions == 1
    }

# Caso 1: Acceso coalesced (óptimo)
print("=== Caso 1: Acceso Coalesced ===")
def coalesced_pattern(tid):
    return tid  # Thread 0→addr 0, Thread 1→addr 1, etc.

result1 = predict_memory_transactions(coalesced_pattern)
print(f"Direcciones: {result1['addresses']}...")
print(f"Transacciones: {result1['num_transactions']}")
print(f"Eficiencia: {result1['efficiency']:.1%}")
print(f"Coalesced: {'✓' if result1['is_coalesced'] else '✗'}")

# Caso 2: Acceso strided (malo)
print("\n=== Caso 2: Acceso Strided (stride=32) ===")
def strided_pattern(tid):
    return tid * 32  # Saltos grandes

result2 = predict_memory_transactions(strided_pattern)
print(f"Direcciones: {result2['addresses']}...")
print(f"Transacciones: {result2['num_transactions']}")
print(f"Eficiencia: {result2['efficiency']:.1%}")
print(f"Coalesced: {'✓' if result2['is_coalesced'] else '✗'}")

# Caso 3: Acceso por columnas (2D array row-major)
print("\n=== Caso 3: Acceso por Columnas (M=128) ===")
def column_access_pattern(tid):
    M = 128  # Ancho de matriz
    return tid * M  # Saltar M elementos

result3 = predict_memory_transactions(column_access_pattern)
print(f"Direcciones: {result3['addresses']}...")
print(f"Transacciones: {result3['num_transactions']}")
print(f"Eficiencia: {result3['efficiency']:.1%}")
print(f"Coalesced: {'✓' if result3['is_coalesced'] else '✗'}")

print("\n💡 Regla: Threads consecutivos deben acceder direcciones consecutivas para 1 transacción")
```

### Ejercicio 2: Análisis Roofline

GPU con 10 TFLOPS peak, 500 GB/s bandwidth.

1. ¿Cuál es el ridge point?
2. ¿Vector addition (AI=0.125) es memory o compute bound?
3. ¿MatMul 1024x1024 con tiling (AI≈8)?

### Ejercicio 3: Identificar Optimizaciones

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

**Preguntas:**
1. ¿Memory-bound o compute-bound? ¿Por qué?
2. ¿Qué optimizaciones aplicarías?
3. ¿Autotuning ayudaría? ¿Qué parámetros tunear?

### Para Pensar

> *Si un kernel está al 90% del roofline pero sigue siendo lento para tu aplicación, ¿qué opciones tienes?*

Pista: No solo optimización de kernel...

---

## Errores Comunes

```{admonition} Errores frecuentes en optimización
:class: warning

1. **Acceso no coalesced**: Threads consecutivos acceden a direcciones alejadas. Siempre accede por filas (row-major).
2. **Bank conflicts**: Stride de 32 en shared memory causa conflictos. Usa padding (33 en vez de 32).
3. **Olvidar máscaras**: Accesos sin máscaras pueden leer fuera de límites.
4. **No usar tiling**: Recargar datos desde global en cada operación es lento.
5. **Medir sin warmup**: Incluye JIT compilation, 10-100x más lento que realidad.
6. **No sincronizar GPU**: Mides lanzamiento, no ejecución real del kernel.
```

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Williams, S., Waterman, A., & Patterson, D. (2009). [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785). Communications of the ACM.
- NVIDIA. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). NVIDIA Developer.
- NVIDIA. [Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/). NVIDIA Docs.
- Triton. [Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Triton Docs.
- Harris, M. (2017). [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/). NVIDIA Blog.
