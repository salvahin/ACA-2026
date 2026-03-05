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

# CUDA Conceptos y PyTorch GPU Internals


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/02_cuda_pytorch_internals.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 2
> **Tiempo de lectura:** ~45 minutos

---

## Introducción

CUDA es el lenguaje de NVIDIA para programar GPUs. PyTorch es el framework que usarás para desarrollar y evaluar kernels. Esta lectura combina ambos: entenderás los conceptos fundamentales de CUDA y cómo PyTorch los abstrae para gestionar tensores, memoria, y ejecución en GPU.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Escribir y lanzar kernels CUDA básicos con grid/block configuration
- Calcular índices globales de threads usando `blockIdx.x * blockDim.x + threadIdx.x`
- Aplicar coalescing para accesos de memoria eficientes (threads consecutivos → direcciones consecutivas)
- Minimizar transferencias CPU↔GPU usando tensores en GPU y pinned memory
- Usar CUDA streams de PyTorch para paralelismo de operaciones
- Identificar bottlenecks con `torch.profiler` (CUDA time, memory, FLOPS)
```

---

## Parte 1: Conceptos Fundamentales de CUDA

### Estructura de un Programa CUDA

Todo programa CUDA tiene esta estructura:

```cuda
#include <stdio.h>

// 1. Kernel (código que corre en GPU)
__global__ void kernel_ejemplo(float *datos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    datos[idx] = datos[idx] * 2.0f;
}

// 2. Host code (código que corre en CPU)
int main() {
    int n = 1000;
    float *d_datos;  // d_ prefijo = datos en GPU

    // Allocate memoria
    cudaMalloc(&d_datos, n * sizeof(float));

    // Lanzar kernel
    int threads_por_bloque = 256;
    int num_bloques = (n + threads_por_bloque - 1) / threads_por_bloque;
    kernel_ejemplo<<<num_bloques, threads_por_bloque>>>(d_datos);

    // Liberar memoria
    cudaFree(d_datos);

    return 0;
}
```

Elementos clave:
- `__global__`: Marca una función como kernel (ejecutable en GPU)
- `<<<num_bloques, threads_por_bloque>>>`: Sintaxis de lanzamiento
- `blockIdx`, `threadIdx`: Variables que identifican cada thread

### Lanzamiento de Kernels

```cuda
kernel_nombre<<<num_bloques, threads_por_bloque>>>(argumentos);

// Ejemplo: 100 bloques, cada uno con 256 threads
suma_kernel<<<100, 256>>>(a, b, c, n);

// Total: 25,600 threads en paralelo
```

**Reglas:**
1. Threads por bloque: Típicamente 128, 256 o 512. Máximo 1024.
2. Número de bloques: Puede ser muy grande (millones).
3. No hay garantía de orden entre bloques.

### Indexación: La Fórmula Clave

```cuda
__global__ void kernel_indexacion(float *datos) {
    int thread_id_local = threadIdx.x;        // 0 a blockDim.x-1
    int bloque_id = blockIdx.x;               // 0 a gridDim.x-1
    int threads_en_bloque = blockDim.x;

    // Índice global:
    int idx_global = blockIdx.x * blockDim.x + threadIdx.x;

    datos[idx_global] = 42.0f;
}
```

**Visualización:**
```
Grid con 2 bloques de 4 threads:

┌─ Block 0 ──────────┬─ Block 1 ──────────┐
│ blockIdx.x = 0     │ blockIdx.x = 1     │
├────────────────────┼────────────────────┤
│ Thread 0: idx = 0  │ Thread 0: idx = 4  │
│ Thread 1: idx = 1  │ Thread 1: idx = 5  │
│ Thread 2: idx = 2  │ Thread 2: idx = 6  │
│ Thread 3: idx = 3  │ Thread 3: idx = 7  │
└────────────────────┴────────────────────┘
```

### Grillas 2D

Para imágenes o matrices:

```cuda
__global__ void kernel_2d(float *imagen, int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < ancho && y < alto) {
        int idx = y * ancho + x;  // Acceso row-major
        imagen[idx] = imagen[idx] * 2.0f;
    }
}

// Lanzar kernel 2D
dim3 bloques(10, 10);        // 10x10 = 100 bloques
dim3 threads(16, 16);        // 16x16 = 256 threads por bloque
kernel_2d<<<bloques, threads>>>(img, 1920, 1080);
```

### Sincronización con __syncthreads()

```cuda
__global__ void kernel_sincronizado(float *datos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float mi_valor = datos[idx] * 2.0f;

    // BARRERA: esperar a que TODOS los threads del bloque lleguen aquí
    __syncthreads();

    // Ahora todos terminaron la fase anterior
    datos[idx] = mi_valor;
}
```

**Importante:** `__syncthreads()` solo sincroniza threads del **mismo bloque**. No hay barrera global.

### Coalescing de Memoria

```{admonition} ⚡ Tip de Performance: Memory Coalescing
:class: important
**Regla de oro**: Los primeros 32 threads de un warp deben acceder direcciones **consecutivas** para coalescing óptimo.

**Por qué importa**: GPU accede memoria en transacciones de 128 bytes. Con coalescing perfecto:
- 32 threads → 1 transacción ✓
Sin coalescing:
- 32 threads → 32 transacciones ✗ (32x más lento!)
```

Cuando múltiples threads leen de memoria:

```cuda
// Acceso coalescido (BUENO):
int idx = threadIdx.x;  // Thread 0 lee [0], Thread 1 lee [1], etc.
float valor = datos[idx];
// Se coalescen en pocas transacciones de memoria

// Acceso NO coalescido (MALO):
int idx = threadIdx.x * 2;  // Thread 0 lee [0], Thread 1 lee [2], etc.
float valor = datos[idx];
// Requiere múltiples transacciones separadas
```

```{admonition} 📊 Cómo verificar
:class: tip
Usa `nsight compute` para ver coalescing efficiency:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./program
```
Busca "Global Memory Load Efficiency" > 80% para coalescing bueno.
```

### Bank Conflicts en Memoria Compartida

```{admonition} ⚠️ Antipatrón: Bank Conflicts en Shared Memory
:class: warning
**Problema**: Shared memory se divide en 32 bancos. Si múltiples threads acceden al mismo banco (diferentes direcciones), se serializan.

**Causa común**: `smem[threadIdx.x * 32]` → todos acceden Bank 0 → 32 ciclos en vez de 1

**Impacto**: Serialización completa - rendimiento se divide por N threads conflictuados.

**Solución**: Usa padding. En vez de array[32], usa array[33] para desalinear bancos.
```

```
Memoria Compartida se divide en 32 "bancos":
Bank 0:  [word 0, word 32, word 64, ...]
Bank 1:  [word 1, word 33, word 65, ...]
...
Bank 31: [word 31, word 63, word 95, ...]

Cada banco puede servir UNA lectura por ciclo
```

```cuda
__shared__ float smem[256];

// SIN conflict (excelente):
float v = smem[threadIdx.x];  // Cada thread accede un banco diferente

// CON conflict (terrible):
float v = smem[threadIdx.x * 2];  // Threads pares acceden al mismo banco
```

```{admonition} 🎯 En tu proyecto
:class: note
Cuando uses shared memory para tiling (ej: matmul), usa padding: `__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]` para evitar bank conflicts en accesos por columnas.
```

---

## Parte 2: PyTorch GPU Internals

### Tensores en GPU

```{code-cell} ipython3
:tags: [skip-execution]

import torch

# Crear tensor en CPU
x_cpu = torch.randn(1000, 1000)

# Mover a GPU
x_gpu = x_cpu.to('cuda')        # Copia explícita
x_gpu = x_cpu.cuda()            # Equivalente

# Crear directamente en GPU
x_gpu = torch.randn(1000, 1000, device='cuda')

# Verificar dispositivo
print(x_gpu.device)  # cuda:0
```

### Costo del Movimiento CPU↔GPU

```{admonition} 🧠 Modelo Mental: CPU↔GPU Transfer
:class: hint
Piensa en CPU y GPU como dos ciudades conectadas por un puente:
- **Dentro de GPU**: Autopista de 12 carriles (2000 GB/s HBM)
- **CPU↔GPU**: Puente angosto (32 GB/s PCIe)

Mover 1GB de CPU a GPU = ~30ms (¡es MUCHO tiempo en GPU!)
Durante esos 30ms, tu GPU está parada esperando datos.
```

```
PCIe Gen4: ~32 GB/s bidireccional
HBM (GPU interna): ~2000 GB/s

Mover 1GB de CPU a GPU:
- Tiempo: ~30ms
- Durante este tiempo, la GPU está esperando

Regla: Minimiza transferencias CPU↔GPU
```

```{admonition} ⚡ Tip de Performance: Minimizar Transferencias
:class: important
Para evitar cuello de botella PCIe:
1. **Crea tensores directamente en GPU**: `torch.randn(..., device='cuda')`
2. **Batch múltiples operaciones**: No muevas datos después de cada operación
3. **Usa pinned memory** para transferencias asíncronas cuando sea necesario
4. **Pipeline**: Overlap transferencias con compute usando streams
```

```{admonition} Concepto clave
:class: note
El bandwidth PCIe es 60x menor que el HBM interno. Mantén los datos en GPU el mayor tiempo posible. Una regla simple: si tu kernel tarda <30ms pero mueves 1GB de datos, la transferencia es tu bottleneck real.
```

### Memory Layout y Contiguidad

```{code-cell} ipython3
:tags: [skip-execution]

# Tensor contiguo (óptimo para GPU)
x = torch.randn(100, 100)
print(x.is_contiguous())  # True

# Tensor no contiguo (después de transpose)
y = x.t()
print(y.is_contiguous())  # False

# Hacer contiguo (copia los datos)
y_contig = y.contiguous()
```

Tensores no contiguos causan accesos de memoria ineficientes.

### Caching Allocator de PyTorch

PyTorch usa un **caching allocator** para evitar llamadas frecuentes a `cudaMalloc`:

```{code-cell} ipython3
:tags: [skip-execution]

# Primera allocación: llama a cudaMalloc
x = torch.randn(1000, 1000, device='cuda')

# Liberar tensor
del x

# Segunda allocación del mismo tamaño: reutiliza memoria cacheada
y = torch.randn(1000, 1000, device='cuda')  # No llama cudaMalloc
```

### Monitoreo de Memoria

```{code-cell} ipython3
:tags: [skip-execution]

# Memoria actualmente allocada
allocated = torch.cuda.memory_allocated()

# Memoria reservada por el caching allocator
reserved = torch.cuda.memory_reserved()

# Pico de memoria
max_allocated = torch.cuda.max_memory_allocated()

print(f"Allocated: {allocated / 1e9:.2f} GB")
print(f"Reserved:  {reserved / 1e9:.2f} GB")
print(f"Peak:      {max_allocated / 1e9:.2f} GB")

# Liberar memoria cacheada
torch.cuda.empty_cache()
```

### CUDA Streams

Un **CUDA stream** es una secuencia de operaciones que se ejecutan en orden. Streams diferentes pueden ejecutarse en paralelo.

```{code-cell} ipython3
:tags: [skip-execution]

# Stream por defecto
x = torch.randn(1000, 1000, device='cuda')
y = x * 2  # Ejecuta en stream por defecto

# Crear stream custom
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    z = x + y
    w = z * 3

# Sincronización
torch.cuda.synchronize()  # Todos los streams
stream.synchronize()       # Stream específico
```

### Paralelismo con Streams

```{code-cell} ipython3
:tags: [skip-execution]

# Sin streams: secuencial
def sequential():
    for i in range(10):
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x.t()
    torch.cuda.synchronize()

# Con streams: paralelo
def parallel():
    streams = [torch.cuda.Stream() for _ in range(10)]
    results = []

    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            x = torch.randn(1000, 1000, device='cuda')
            y = x @ x.t()
            results.append(y)

    torch.cuda.synchronize()
    return results
```

### torch.profiler

```{code-cell} ipython3
:tags: [skip-execution]

from torch.profiler import profile, ProfilerActivity

model = torch.nn.Linear(1000, 1000).cuda()
x = torch.randn(100, 1000, device='cuda')

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(10):
        y = model(x)
        y.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Output típico:**
```
Name                    CPU Time  CUDA Time  # Calls   Memory
aten::addmm              100us     800us       10      4MB
aten::mm                  80us     600us       10      4MB
aten::sum                 20us     100us       10      0B
```

### Patrones Problemáticos

```{code-cell} ipython3
:tags: [skip-execution]

# MALO: Muchas operaciones pequeñas
for i in range(1000):
    x = x + 1  # 1000 kernel launches

# BUENO: Una operación grande
x = x + 1000  # 1 kernel launch

# MALO: Sincronización innecesaria
for i in range(100):
    x = model(x)
    print(x[0, 0].item())  # .item() sincroniza!

# BUENO: Batch las sincronizaciones
results = []
for i in range(100):
    x = model(x)
    results.append(x[0, 0])
torch.cuda.synchronize()
print([r.item() for r in results])
```

### Pinned Memory para Transferencias Rápidas

```{code-cell} ipython3
:tags: [skip-execution]

# Sin pinned memory
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.cuda()  # Copia síncrona lenta

# Con pinned memory
x_pinned = torch.randn(1000, 1000).pin_memory()
x_gpu = x_pinned.cuda(non_blocking=True)  # Copia asíncrona rápida
```

---

## Resumen de Conceptos Clave

```{admonition} Resumen
:class: important
**Mapa conceptual CUDA ↔ PyTorch:**

| Concepto | CUDA | PyTorch |
|----------|------|---------|
| Identificar thread | `blockIdx`, `threadIdx` | N/A (abstracción) |
| Lanzar kernel | `<<<bloques, threads>>>` | Automático |
| Sincronizar | `__syncthreads()` | `torch.cuda.synchronize()` |
| Memoria GPU | `cudaMalloc/cudaFree` | Caching allocator |
| Streams | `cudaStream_t` | `torch.cuda.Stream()` |
| Profiling | `nvprof`, `nsight` | `torch.profiler` |

**Checklist de optimización:**
- [ ] Indexación correcta: `idx = blockIdx.x * blockDim.x + threadIdx.x`
- [ ] Coalescing: Threads consecutivos acceden direcciones consecutivas
- [ ] Minimizar CPU↔GPU: Crear tensores en GPU desde el inicio
- [ ] Sincronización: Llamar `torch.cuda.synchronize()` antes de medir tiempo
- [ ] Pinned memory para transferencias asíncronas cuando sea necesario
```

```{admonition} ✅ Verifica tu comprensión
:class: note
**Pregunta 1**: Para grid de 100 bloques con 256 threads/bloque, ¿cuál es el índice global del thread con blockIdx=5, threadIdx=32?
<details><summary>Respuesta</summary>
`idx = 5 * 256 + 32 = 1280 + 32 = 1312`
</details>

**Pregunta 2**: ¿Por qué `x.item()` en un loop es lento?
<details><summary>Respuesta</summary>
Porque `.item()` sincroniza GPU→CPU para cada iteración, bloqueando ejecución asíncrona y pagando overhead de PCIe cada vez.
</details>

**Pregunta 3**: Tienes 1000 operaciones pequeñas. ¿Mejor lanzar 1000 kernels o fusionarlas?
<details><summary>Respuesta</summary>
Fusionar. Cada kernel launch tiene overhead (~5-10μs). 1000 launches = 5-10ms de overhead puro. Un kernel fusionado elimina ese overhead.
</details>
```

---

## Ejercicios y Reflexión

### Ejercicio 1: Cálculo de Índices

Para un grid de 5x3 bloques con 16x16 threads por bloque:
- ¿Cuántos threads totales?
- ¿Cuál es el índice global del thread en bloque (2,1) con threadIdx (3,5)?

### Ejercicio 2: Coalescing

Analiza: ¿Cuál versión tiene mejor coalescing?

```cuda
// Versión A
int idx = threadIdx.x + i;
datos[idx] = procesar(datos[idx]);

// Versión B
int idx = threadIdx.x + blockIdx.x * blockDim.x + i;
datos[idx] = procesar(datos[idx]);
```

### Ejercicio 3: Profiling PyTorch

Usa `torch.profiler` para medir:
1. Multiplicación de matrices 1000x1000
2. Forward pass de ResNet-18

¿Cuál tiene mejor utilización de GPU?

### Para Pensar

> *Si el profiler muestra 80% del tiempo en transferencias CPU↔GPU, ¿qué cambios arquitectónicos considerarías?*

---

## Próximos Pasos

En la siguiente semana, exploraremos **Optimización de Memoria GPU**: coalesced access patterns, bank conflicts, tiling, y el modelo roofline.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- PyTorch. [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html). PyTorch Docs.
- NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). NVIDIA Developer.
- Paszke, A. et al. (2019). [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703). NeurIPS 2019.
