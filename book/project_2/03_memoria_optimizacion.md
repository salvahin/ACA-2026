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

# Optimización de Memoria GPU: Coalescing, Bank Conflicts y Roofline

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/03_memoria_optimizacion.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 3
> **Tiempo de lectura:** ~40 minutos

---

## Introducción

La diferencia entre un kernel GPU lento y uno rápido frecuentemente está en **cómo accede a la memoria**. Un kernel puede tener la lógica correcta pero ser 10x más lento por patrones de acceso ineficientes.

Esta lectura te enseña los conceptos fundamentales de optimización de memoria GPU: coalesced access, bank conflicts, tiling, y cómo usar el modelo roofline para analizar performance.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Diseñar patrones de acceso coalesced (threads consecutivos → direcciones consecutivas)
- Evitar bank conflicts en shared memory usando padding
- Aplicar tiling para mejorar localidad de cache (reducir accesos a global memory)
- Usar el modelo roofline para determinar si tu kernel es memory-bound o compute-bound
- Optimizar kernels según su bottleneck (fusionar ops si memory-bound, reducir divergencia si compute-bound)
```

---

## Coalesced Memory Access

```{admonition} 🧠 Modelo Mental: Coalescing
:class: hint
Piensa en la memoria GPU como un autobús que transporta 128 bytes a la vez:
- **Coalesced**: 32 pasajeros suben en orden (1 viaje)
- **No coalesced**: 32 pasajeros en diferentes paradas (32 viajes)

Mismo número de pasajeros, 32x diferencia de tiempo.
```

### El Problema

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

### Patrones Correctos en Triton

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

### Visualización

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

```{code-cell} ipython3
# Visualización de Memory Coalescing
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=('✅ Coalesced Access', '❌ Strided Access'))

# Coalesced: threads acceden a direcciones consecutivas
for i in range(8):
    # Thread
    fig.add_shape(type="rect", x0=i*0.1+0.05, y0=0.7, x1=i*0.1+0.13, y1=0.8,
                  fillcolor='#4ECDC4', row=1, col=1)
    fig.add_annotation(x=i*0.1+0.09, y=0.75, text=f'T{i}', showarrow=False,
                       font=dict(size=8), row=1, col=1)
    # Memory
    fig.add_shape(type="rect", x0=i*0.1+0.05, y0=0.3, x1=i*0.1+0.13, y1=0.4,
                  fillcolor='#95E1A3', row=1, col=1)
    fig.add_annotation(x=i*0.1+0.09, y=0.35, text=f'M{i}', showarrow=False,
                       font=dict(size=8), row=1, col=1)
    # Arrow
    fig.add_annotation(x=i*0.1+0.09, y=0.5, ax=i*0.1+0.09, ay=0.65,
                       arrowhead=2, arrowcolor='green', row=1, col=1)

# Strided: threads acceden con stride
stride = 4
for i in range(8):
    fig.add_shape(type="rect", x0=i*0.1+0.05, y0=0.7, x1=i*0.1+0.13, y1=0.8,
                  fillcolor='#4ECDC4', row=1, col=2)
    fig.add_annotation(x=i*0.1+0.09, y=0.75, text=f'T{i}', showarrow=False,
                       font=dict(size=8), row=1, col=2)

    mem_pos = (i * stride) % 8
    fig.add_shape(type="rect", x0=mem_pos*0.1+0.05, y0=0.3, x1=mem_pos*0.1+0.13, y1=0.4,
                  fillcolor='#FF6B6B', row=1, col=2)

    fig.add_annotation(x=i*0.1+0.09, y=0.55, ax=mem_pos*0.1+0.09, ay=0.42,
                       arrowhead=2, arrowcolor='red', row=1, col=2)

fig.add_annotation(x=0.45, y=0.15, text="1 transacción de memoria",
                   showarrow=False, font=dict(color='green', size=12), row=1, col=1)
fig.add_annotation(x=0.45, y=0.15, text="Múltiples transacciones",
                   showarrow=False, font=dict(color='red', size=12), row=1, col=2)

fig.update_layout(height=350, showlegend=False)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()
```

:::{figure} diagrams/memory_coalescing.png
:name: fig-memory-coalescing
:alt: Patrón de acceso coalesced (1 transacción) vs no-coalesced (múltiples transacciones) a memoria global
:align: center
:width: 90%

**Figura 1:** Coalesced Memory Access - Threads consecutivos accediendo posiciones consecutivas resulta en una sola transacción; acceso disperso requiere múltiples transacciones.
:::

---

## Shared Memory y Bank Conflicts

### Estructura de Banks

Shared memory está dividida en **32 banks**. Cada bank puede servir una solicitud por ciclo:

```
Bank 0: addr 0, 32, 64, 96, ...
Bank 1: addr 1, 33, 65, 97, ...
...
Bank 31: addr 31, 63, 95, 127, ...
```

### Bank Conflict

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

### Patrones Problemáticos

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

## Tiling: Mejorando Localidad

### Concepto

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

### Ejemplo: Matrix Multiply con Tiling

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def matmul_tiled(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterar sobre tiles de K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])

        acc += tl.dot(a, b)  # En registros, muy rápido

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

:::{figure} diagrams/tiling_strategy.png
:name: fig-tiling-strategy
:alt: Estrategia de tiling: dividir problema en tiles que caben en cache, procesarlos localmente, y combinar resultados
:align: center
:width: 90%

**Figura 3:** Estrategia de Tiling - Dividir matrices grandes en tiles pequeños que caben en Shared Memory (rápida), minimizando accesos a Global Memory.
:::

### Beneficios

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

## Modelo Roofline

### Concepto

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

### Cálculo

```{code-cell} ipython3
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

**Figura 2:** Modelo Roofline - Visualiza el rendimiento máximo alcanzable según la intensidad aritmética. Ridge Point separa kernels memory-bound de compute-bound.
:::

### Operaciones Comunes

| Operación | AI | Bound |
|-----------|-----|-------|
| Vector Add | 0.08 | Memory |
| Elementwise | 0.125 | Memory |
| MatMul (naive) | ~0.5 | Memory |
| MatMul (tiled) | ~8 | Compute |
| Softmax | 0.4 | Memory |

---

## Técnicas de Optimización por Tipo

### Para Kernels Memory-Bound

```{code-cell} ipython3
:tags: [skip-execution]

# 1. Fusionar operaciones
# MALO: 3 kernels, 3 round-trips
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

# 2. Vectorizar cargas (Triton lo hace automáticamente)

# 3. Prefetching
for i in range(N):
    next_data = tl.load(...)  # Prefetch
    result = compute(current_data)
    current_data = next_data
```

### Para Kernels Compute-Bound

```python
# 1. Usar tensor cores (tl.dot en Triton)

# 2. Reducir divergencia
# MALO
if tl.arange(0, 32) % 2 == 0:
    x = compute_a()
else:
    x = compute_b()

# BUENO
x = compute_a()
y = compute_b()
result = tl.where(mask, x, y)

# 3. Loop unrolling
for i in tl.static_range(4):  # Compile-time unroll
    acc += compute(i)
```

---

## Caso de Estudio: Optimizando Softmax

### Versión Naive

```{code-cell} ipython3
:tags: [skip-execution]

def softmax_naive(x):
    # 4 kernel launches, 4 round-trips
    max_x = x.max(dim=-1, keepdim=True)
    x = x - max_x
    exp_x = x.exp()
    sum_exp = exp_x.sum(dim=-1, keepdim=True)
    return exp_x / sum_exp
```

### Análisis

```
Para tensor [B, N]:
- Total bytes: ~7*BN
- FLOPs: ~5*BN
- AI: 5/28 ≈ 0.18 → Muy memory-bound
```

### Versión Optimizada (Fused)

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def softmax_fused(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    # Una sola lectura
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=-float('inf'))

    # Todo en registros
    max_x = tl.max(x, axis=0)
    x = x - max_x
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    # Una sola escritura
    tl.store(out_ptr + row * N + offsets, softmax, mask=mask)
```

### Mejora

```
Total bytes: 2*BN (una lectura, una escritura)
AI: 5/8 ≈ 0.625 → 3.5x mejor

En práctica: 2-4x speedup vs naive
```

---

## Profiling para Identificar Bottlenecks

### Con PyTorch Profiler

```{code-cell} ipython3
:tags: [skip-execution]

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

### Checklist de Diagnóstico

```
□ ¿Accesos a memoria son coalesced?
□ ¿Hay bank conflicts en shared memory?
□ ¿Occupancy es razonable?
□ ¿Estás cerca del roofline?
  → Calcular: achieved_FLOPS / theoretical_max
```

---

## Resumen

```{admonition} Resumen
:class: important
**Conceptos clave:**
- **Coalesced Access**: Threads consecutivos → direcciones consecutivas (1 transacción vs 32)
- **Bank Conflicts**: Evitar múltiples threads al mismo bank (usa padding: stride=33 en vez de 32)
- **Tiling**: Cargar bloques a shared memory para reutilizar datos (5-10x speedup típico)
- **Roofline**: Memory-bound (<10 FLOP/byte) vs compute-bound (>10 FLOP/byte)
- **Optimización por bottleneck**:
  - Memory-bound → Fusionar kernels, vectorizar, prefetch
  - Compute-bound → Reducir divergencia, usar tensor cores, unroll loops

**Checklist de optimización de memoria:**
- [ ] Memory coalescing verificado (threads consecutivos acceden direcciones consecutivas)
- [ ] Shared memory sin bank conflicts (padding aplicado donde necesario)
- [ ] Occupancy razonable (>50% para memory-bound, >75% para compute-bound)
- [ ] Roofline calculado (¿cerca del límite teórico?)
```

```{admonition} 📊 Cómo verificar optimizaciones
:class: tip
**Herramientas de profiling:**
```bash
# Verificar coalescing
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./program

# Verificar bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./program

# Ver occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./program
```

Busca:
- Global Load Efficiency > 80%
- Bank Conflicts < 1% de accesos
- Occupancy > 50%
```

```{admonition} 🎯 En tu proyecto
:class: note
Aplicarás tiling cuando implementes:
- Multiplicación de matrices (cargar tiles de A y B a shared)
- Convoluciones (cargar tiles de input para reutilizar)
- Reducciones grandes (dos etapas: local + global)

La ganancia típica es 5-10x comparado con acceso directo a global memory.
```

---

## Ejercicios

### Ejercicio 1: Analiza Accesos

```{code-cell} ipython3
:tags: [skip-execution]

offsets = tl.arange(0, 32) * 4  # [0, 4, 8, 12, ...]
x = tl.load(data_ptr + offsets)
```
¿Es coalesced? ¿Cómo mejorarlo?

### Ejercicio 2: Cálculo de Roofline

GPU con 10 TFLOPS peak, 500 GB/s bandwidth.
¿Memory-bound o compute-bound?
1. Vector addition: 1 FLOP/elemento
2. Matrix multiply 1024x1024

### Para Pensar

> *Si un kernel está al 90% del roofline pero sigue siendo lento, ¿qué opciones tienes?*

---

## Errores Comunes

```{admonition} Errores frecuentes en optimización de memoria
:class: warning

1. **Acceso no coalesced**: Threads consecutivos acceden a direcciones alejadas. Siempre accede por filas (row-major).
2. **Bank conflicts**: Stride de 32 en shared memory causa conflictos. Usa padding (33 en vez de 32).
3. **Olvidar máscaras**: Accesos sin máscaras pueden leer fuera de límites.
4. **No usar tiling**: Recargar datos desde global en cada operación es lento.
```

## Ejercicio Práctico: Predecir Transacciones de Memoria

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

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Williams, S., Waterman, A., & Patterson, D. (2009). [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785). Communications of the ACM.
- NVIDIA. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). NVIDIA Developer.
- Harris, M. (2017). [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/). NVIDIA Blog.
