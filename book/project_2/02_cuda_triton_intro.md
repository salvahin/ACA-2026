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

# CUDA Conceptos y Primer Kernel Triton

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q torch triton
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/02_cuda_triton_intro.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 2
> **Tiempo de lectura:** ~40 minutos

---

## Introducción

Ahora que entiendes la arquitectura GPU (Grid→Blocks→Threads→Warps), es momento de escribir código que ejecute en ella. Esta sesión fusiona dos mundos: **CUDA** (el lenguaje de bajo nivel de NVIDIA) y **Triton** (una abstracción moderna que simplifica la escritura de kernels).

Aprenderás la estructura fundamental de CUDA para comprender **qué hace la GPU bajo el capó**, y luego verás cómo Triton te permite escribir kernels optimizados sin gestionar manualmente todos los detalles de sincronización y memoria.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Escribir y lanzar kernels CUDA básicos con `__global__` y `<<<bloques, threads>>>`
- Calcular índices globales usando la fórmula `blockIdx.x * blockDim.x + threadIdx.x`
- Aplicar coalescing de memoria (threads consecutivos → direcciones consecutivas)
- Usar `__syncthreads()` para sincronizar threads dentro de un bloque
- Gestionar CUDA streams en PyTorch para paralelismo de operaciones
- Escribir tu primer kernel Triton con `@triton.jit`, `tl.load()`, `tl.store()`
- Comparar el modelo "threads" de CUDA vs "bloques de datos" de Triton
```

---

## Bloque 1: Modelo CUDA (25' clase + 25' trabajo)

### Estructura de un Programa CUDA

Todo programa CUDA tiene esta estructura básica:

```cuda
#include <stdio.h>

// 1. Kernel (código que corre en GPU)
__global__ void kernel_ejemplo(float *datos, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        datos[idx] = datos[idx] * 2.0f;
    }
}

// 2. Host code (código que corre en CPU)
int main() {
    int n = 1000;
    float *d_datos;  // d_ prefijo = datos en GPU

    // Allocate memoria en GPU
    cudaMalloc(&d_datos, n * sizeof(float));

    // Lanzar kernel
    int threads_por_bloque = 256;
    int num_bloques = (n + threads_por_bloque - 1) / threads_por_bloque;
    kernel_ejemplo<<<num_bloques, threads_por_bloque>>>(d_datos, n);

    // Sincronizar (esperar a que termine)
    cudaDeviceSynchronize();

    // Liberar memoria
    cudaFree(d_datos);

    return 0;
}
```

**Elementos clave:**
- `__global__`: Marca una función como kernel (ejecutable en GPU desde CPU)
- `<<<num_bloques, threads_por_bloque>>>`: Sintaxis de lanzamiento del kernel
- `blockIdx`, `threadIdx`: Variables built-in que identifican cada thread
- `cudaMalloc/cudaFree`: Gestión explícita de memoria GPU

### Lanzamiento de Kernels

```cuda
kernel_nombre<<<num_bloques, threads_por_bloque>>>(argumentos);

// Ejemplo: 100 bloques, cada uno con 256 threads
suma_kernel<<<100, 256>>>(a, b, c, n);

// Total: 25,600 threads ejecutando en paralelo
```

**Reglas importantes:**
1. **Threads por bloque**: Típicamente 128, 256 o 512. Máximo 1024.
2. **Número de bloques**: Puede ser muy grande (millones).
3. **No hay garantía de orden** entre bloques (pero sí dentro de un bloque con `__syncthreads()`).

### Indexación: La Fórmula Mágica

Esta es **la línea más importante** que escribirás en CUDA:

```cuda
__global__ void kernel_indexacion(float *datos, int n) {
    // Índice local dentro del bloque (0 a blockDim.x-1)
    int thread_id_local = threadIdx.x;

    // ID del bloque (0 a gridDim.x-1)
    int bloque_id = blockIdx.x;

    // Tamaño del bloque
    int threads_en_bloque = blockDim.x;

    // FÓRMULA CLAVE: Índice global único
    int idx_global = blockIdx.x * blockDim.x + threadIdx.x;

    // Siempre verificar límites
    if (idx_global < n) {
        datos[idx_global] = 42.0f;
    }
}
```

**Visualización del mapeo:**
```
Grid con 3 bloques de 4 threads cada uno:

┌─ Block 0 ──────────┬─ Block 1 ──────────┬─ Block 2 ──────────┐
│ blockIdx.x = 0     │ blockIdx.x = 1     │ blockIdx.x = 2     │
├────────────────────┼────────────────────┼────────────────────┤
│ Thread 0: idx = 0  │ Thread 0: idx = 4  │ Thread 0: idx = 8  │
│ Thread 1: idx = 1  │ Thread 1: idx = 5  │ Thread 1: idx = 9  │
│ Thread 2: idx = 2  │ Thread 2: idx = 6  │ Thread 2: idx = 10 │
│ Thread 3: idx = 3  │ Thread 3: idx = 7  │ Thread 3: idx = 11 │
└────────────────────┴────────────────────┴────────────────────┘

Formula: idx = blockIdx.x * 4 + threadIdx.x
```

### Grillas 2D para Matrices

Para operaciones sobre imágenes o matrices, usa grillas 2D:

```cuda
__global__ void kernel_2d(float *imagen, int ancho, int alto) {
    // Índices 2D
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Verificar límites
    if (x < ancho && y < alto) {
        // Convertir 2D a 1D (row-major order)
        int idx = y * ancho + x;
        imagen[idx] = imagen[idx] * 2.0f;
    }
}

// Lanzar kernel 2D
dim3 bloques(10, 10);        // 10x10 = 100 bloques
dim3 threads(16, 16);        // 16x16 = 256 threads por bloque
kernel_2d<<<bloques, threads>>>(img, 1920, 1080);
```

```{admonition} 🧠 Modelo Mental: 2D → 1D
:class: hint
La memoria GPU es **siempre lineal** (1D), pero indexamos en 2D por conveniencia:
- **Row-major**: `idx = fila * ancho + columna`
- Thread (x=3, y=2) en imagen de ancho=10: `idx = 2 * 10 + 3 = 23`

Es como leer un libro: vas de izquierda a derecha (columnas), luego saltas a la siguiente línea (fila).
```

### Coalescing de Memoria

```{admonition} ⚡ Regla de Oro: Memory Coalescing
:class: important
**Patrón óptimo**: Los primeros 32 threads de un warp acceden direcciones **consecutivas**.

**Por qué importa**: GPU accede memoria en transacciones de 128 bytes.
- Con coalescing perfecto: 32 threads → 1 transacción ✓
- Sin coalescing: 32 threads → 32 transacciones ✗ (32x más lento!)

**Regla práctica**: Thread `i` debe acceder posición `i` (o `i * k` donde `k` es pequeño).
```

Cuando múltiples threads leen de memoria simultáneamente:

```cuda
// ACCESO COALESCIDO (BUENO):
__global__ void kernel_bueno(float *datos, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread 0 lee [0], Thread 1 lee [1], Thread 2 lee [2], etc.
    if (idx < n) {
        float valor = datos[idx];  // Accesos consecutivos
        datos[idx] = valor * 2.0f;
    }
}

// ACCESO NO COALESCIDO (MALO):
__global__ void kernel_malo(float *datos, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread 0 lee [0], Thread 1 lee [2], Thread 2 lee [4], etc.
    int stride_idx = idx * 2;
    if (stride_idx < n) {
        float valor = datos[stride_idx];  // Accesos con stride
        datos[stride_idx] = valor * 2.0f;
    }
}
```

````{admonition} 📊 Cómo verificar coalescing
:class: tip
Usa `nsight compute` para medir eficiencia de memoria:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./program
```
Busca "Global Memory Load Efficiency" > 80% para coalescing bueno.
```
````

### Sincronización con `__syncthreads()`

```cuda
__global__ void kernel_sincronizado(float *datos, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Fase 1: Todos leen
        float mi_valor = datos[idx] * 2.0f;

        // BARRERA: esperar a que TODOS los threads del bloque lleguen aquí
        __syncthreads();

        // Fase 2: Ahora todos pueden usar resultados de Fase 1
        datos[idx] = mi_valor;
    }
}
```

```{admonition} ⚠️ Importante: Scope de `__syncthreads()`
:class: warning
`__syncthreads()` solo sincroniza threads del **mismo bloque**, no de toda la GPU.

**Implicación**: No puedes sincronizar entre bloques diferentes. Si necesitas sincronización global, debes lanzar múltiples kernels secuencialmente.
```

**Caso de uso típico**: Cuando usas shared memory para comunicación entre threads:

```cuda
__shared__ float smem[256];  // Memoria compartida del bloque

__global__ void kernel_shared(float *datos, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar a shared memory
    if (idx < n) {
        smem[threadIdx.x] = datos[idx];
    }

    __syncthreads();  // Esperar a que todos terminen de cargar

    // Ahora todos pueden leer smem[] de forma segura
    if (idx < n) {
        float vecino = smem[(threadIdx.x + 1) % blockDim.x];
        datos[idx] = (smem[threadIdx.x] + vecino) / 2.0f;
    }
}
```

> **Nota**: CUDA Streams y Pinned Memory se cubrieron como lectura opcional en la sesión anterior.
> Para repasar esos conceptos consulta [Sesión 01 — Bloque 2](01_gpu_fundamentals_pytorch.md).

---

## Trabajo para Alumnos: Bloque 1 (25 minutos)

### Ejercicio 1: Calcular Índices Globales

Para un kernel lanzado con `<<<5 bloques, 32 threads>>>`:

**a)** ¿Cuál es el índice global del thread con `blockIdx.x = 3`, `threadIdx.x = 17`?

**b)** ¿Cuántos threads totales se ejecutan?

**c)** Si tu vector tiene 150 elementos, ¿cuál es el índice más alto que necesitas verificar?

### Ejercicio 2: Analizar Patrones de Coalescing

¿Cuál versión tiene mejor coalescing? Justifica.

```cuda
// Versión A
int idx = threadIdx.x;
float val = data[idx];

// Versión B
int idx = threadIdx.x * 32;
float val = data[idx];

// Versión C
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];
```

### Ejercicio 3: Debugging de Sincronización

¿Por qué este código puede dar resultados incorrectos?

```cuda
__global__ void buggy_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        data[idx] = data[idx] * 2;
        // ¿Falta algo aquí?
        data[idx] = data[idx] + data[(idx + 1) % n];
    }
}
```

<details><summary>Hint</summary>
¿Qué pasa si un thread lee `data[(idx + 1) % n]` mientras otro thread todavía está modificando esa posición?
</details>

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

Preguntas para discutir en clase antes de continuar:

1. **Coalescing en la práctica**: Del Ejercicio 2, ¿qué criterio rápido usarías para determinar
   a ojo si un patrón de acceso está coalescido o no? ¿Por qué el stride del índice es el
   indicador clave?
2. **`__syncthreads()` vs kernels separados**: ¿Cuándo conviene usar
   `__syncthreads()` dentro de un kernel versus lanzar dos kernels secuenciales para
   garantizar que todos los threads terminaron su fase anterior?
3. **Race condition del Ejercicio 3**: Dentro de un mismo warp, ¿pueden dos threads
   sufrir la race condition al mismo tiempo? ¿Por qué sí o por qué no?
```

---

## Bloque 2: Primer Kernel Triton (25' clase + 25' trabajo)

### Filosofía de Triton: "Bloques, No Threads"

```{admonition} 🧠 Modelo Mental: CUDA vs Triton
:class: hint
**CUDA**: Piensas como un thread individual
- "Soy el thread 42, proceso el elemento `datos[42]`"
- Control manual: indexación, sincronización, shared memory

**Triton**: Piensas como un **bloque de datos**
- "Soy el programa 5, proceso el bloque `[1280:1536]`"
- Compilador maneja: coalescing, sincronización, vectorización

Es como NumPy vs loops manuales: operas sobre bloques, no elementos individuales.
```

**Comparación visual:**

```
CUDA:   1 thread  → 1 elemento
        Thread 0 procesa datos[0]
        Thread 1 procesa datos[1]
        Thread 2 procesa datos[2]

Triton: 1 programa → 1 bloque de datos
        Programa 0 procesa datos[0:256]
        Programa 1 procesa datos[256:512]
        Programa 2 procesa datos[512:768]
```

### El Decorador `@triton.jit`

```{code-cell} ipython3
:tags: [skip-execution]

import triton
import triton.language as tl

@triton.jit
def kernel_simple(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    # ¿Qué bloque de datos soy?
    pid = tl.program_id(axis=0)  # ID de este programa

    # Crear índices del bloque
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Máscara para límites (evitar out-of-bounds)
    mask = offsets < n

    # Cargar bloque de datos
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Computar (todo el bloque en paralelo)
    y = x * 2

    # Guardar bloque de resultados
    tl.store(y_ptr + offsets, y, mask=mask)
```

**Elementos clave:**
- `@triton.jit`: Marca función para compilación JIT (Just-In-Time)
- `BLOCK_SIZE: tl.constexpr`: Constante en tiempo de compilación (permite optimizaciones)
- `tl.program_id(0)`: ¿Cuál programa/bloque soy? (equivalente a `blockIdx` en CUDA)
- `tl.arange(0, BLOCK_SIZE)`: Crear vector de índices locales [0, 1, 2, ..., BLOCK_SIZE-1]
- `mask=...`: Protección contra accesos fuera de límites

### API Básica de `triton.language`

#### Creación de Vectores

```{code-cell} ipython3
:tags: [skip-execution]

# Índices secuenciales
v = tl.arange(0, 256)           # [0, 1, 2, ..., 255]

# Vectores constantes
zeros = tl.zeros((256,), dtype=tl.float32)
ones = tl.ones((256,), dtype=tl.float32)
```

#### Carga y Almacenamiento con Máscaras

```{code-cell} ipython3
:tags: [skip-execution]

# Cargar datos (SIEMPRE usar máscara)
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
# `other=0.0` → valor por defecto para elementos fuera de límites

# Guardar datos
tl.store(y_ptr + offsets, y, mask=mask)
```

```{admonition} ⚠️ Antipatrón: Olvidar Máscaras
:class: warning
**Error común**: `x = tl.load(x_ptr + offsets)` sin máscara

**Problema**: Si `BLOCK_SIZE=256` pero tu vector tiene 100 elementos, el programa 0 intentará acceder posiciones 0-255, incluyendo 100-255 que no existen → crash o resultados incorrectos.

**Solución**: Siempre usar `mask=`:
```python
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```
```

#### Operaciones Matemáticas

```{code-cell} ipython3
:tags: [skip-execution]

# Aritméticas básicas
y = x + 1
y = x * 2
y = x / 3

# Funciones matemáticas
y = tl.sqrt(x)
y = tl.exp(x)
y = tl.log(x)
y = tl.abs(x)

# Comparaciones y condicionales
mask = x > 0
y = tl.where(x > 0, x, 0.0)  # ReLU
```

### Primer Kernel Completo: Vector Add

```{code-cell} ipython3
:tags: [skip-execution]

import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,  # Puntero a vector x
    y_ptr,  # Puntero a vector y
    z_ptr,  # Puntero a resultado z
    n,      # Tamaño del vector
    BLOCK_SIZE: tl.constexpr  # Tamaño del bloque (constante)
):
    # ¿Qué programa soy?
    pid = tl.program_id(0)

    # Calcular rango de índices que proceso
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Máscara para no salir de límites
    mask = offsets < n

    # Cargar datos
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Computar
    z = x + y

    # Guardar resultado
    tl.store(z_ptr + offsets, z, mask=mask)

# Función Python que lanza el kernel
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Verificar que estén en GPU
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    n = x.numel()
    z = torch.empty_like(x)

    # Configurar grid
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # Lanzar kernel
    vector_add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)

    return z

# Test
x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')
z = vector_add(x, y)

# Verificar contra PyTorch
expected = x + y
assert torch.allclose(z, expected, rtol=1e-5)
print("✓ Kernel funciona correctamente")
```

### Kernel ReLU: Operación Elementwise

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Cargar
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ReLU: max(0, x)
    y = tl.where(x > 0, x, 0.0)

    # Guardar
    tl.store(y_ptr + offsets, y, mask=mask)

def relu_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    n = x.numel()
    y = torch.empty_like(x)

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    relu_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
    return y
```

---

## Trabajo para Alumnos: Bloque 2 (25 minutos)

### Ejercicio 1: Escribir Kernel Triton

Implementa un kernel Triton para la operación `y = 3 * x + 2` (AXPY):

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def axpy_kernel(x_ptr, y_ptr, n, a, b, BLOCK_SIZE: tl.constexpr):
    # TODO: Implementar usando pid, offsets, mask, tl.load y tl.store
    # Nota: a y b son escalares de runtime (no constexpr), pueden cambiar entre llamadas
    pass

# Test
x = torch.randn(1000, device='cuda')
y = torch.empty_like(x)
a, b = 3.0, 2.0

grid = lambda meta: (triton.cdiv(1000, meta['BLOCK_SIZE']),)
axpy_kernel[grid](x, y, 1000, a=a, b=b, BLOCK_SIZE=256)

expected = a * x + b
assert torch.allclose(y, expected)
```

### Ejercicio 2: Comparar Triton vs PyTorch

Implementa un kernel ReLU y compara resultados:

```{code-cell} ipython3
:tags: [skip-execution]

# Tu kernel ReLU (completar)
@triton.jit
def my_relu_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    # TODO: Implementar ReLU con tl.where()
    pass

# Test con valores negativos y positivos
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
y = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
my_relu_kernel[grid](x, y, x.numel(), BLOCK_SIZE=256)

expected = torch.relu(x)
print(f"Tu kernel: {y}")
print(f"PyTorch:   {expected}")
print(f"¿Correcto? {torch.allclose(y, expected)}")
```

### Ejercicio 3: Debugging - Encontrar el Bug

Este kernel tiene un bug. ¿Puedes encontrarlo?

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def buggy_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # BUG: ¿Qué falta aquí?
    x = tl.load(x_ptr + offsets)
    y = tl.where(x < 0, x, 0.0)  # BUG: ¿Lógica correcta?
    tl.store(y_ptr + offsets, y)  # BUG: ¿Falta algo?

# Test que expone el bug
x = torch.randn(100, device='cuda')  # ¡Nota: 100 no es múltiplo de 256!
y = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
try:
    buggy_kernel[grid](x, y, x.numel(), BLOCK_SIZE=256)
    expected = torch.relu(x)
    print(f"¿Correcto? {torch.allclose(y, expected)}")
except Exception as e:
    print(f"Error: {e}")
```

<details><summary>Bugs presentes</summary>

1. **Falta máscara en `tl.load`**: Cuando `n=100` y `BLOCK_SIZE=256`, el programa 0 intenta cargar posiciones 0-255, pero solo existen 0-99.

2. **Lógica invertida**: `tl.where(x < 0, x, 0.0)` devuelve `x` si es negativo, lo contrario de ReLU.

3. **Falta máscara en `tl.store`**: Sin máscara, escribe 256 valores cuando solo debería escribir 100.

**Versión corregida:**
```python
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.where(x > 0, x, 0.0)
tl.store(y_ptr + offsets, y, mask=mask)
```
</details>

---

## Cierre de Sesión *(10 min)*

## Comparación: Triton vs CUDA

| Aspecto | CUDA | Triton |
|---------|------|--------|
| **Abstracción** | Bajo nivel (control total) | Alto nivel (automático) |
| **Unidad de trabajo** | Thread individual | Programa (bloque de datos) |
| **Shared memory** | Manual (`__shared__`) | Automático |
| **Sincronización** | Manual (`__syncthreads()`) | Automática |
| **Coalescing** | Manual (tú diseñas accesos) | Automático |
| **Debugging** | Difícil | Más fácil |
| **Performance** | 100% (máxima) | ~95%+ (excelente) |
| **Curva de aprendizaje** | Empinada | Moderada |

### Ejemplo Lado a Lado

**CUDA:**
```cuda
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**Triton equivalente:**
```python
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
```

**Observaciones:**
- CUDA: Indexación explícita por thread
- Triton: Indexación por bloque de datos
- CUDA: Verificación de límites manual con `if`
- Triton: Verificación con `mask=` en load/store
- Ambos: Misma performance para kernels simples

```{admonition} 🎯 Cuándo Usar Cada Uno
:class: note
**Usa Triton cuando:**
- Escribes kernels de deep learning (elementwise, reducciones, matmul)
- Quieres prototipado rápido
- Performance ~95%+ es suficiente
- El patrón de acceso es regular

**Usa CUDA cuando:**
- Necesitas control absoluto del hardware
- Patrones de acceso muy irregulares
- Optimizaciones micro-arquitectónicas específicas
- Performance del último 5% es crítica
```

---

## Resumen de Conceptos Clave

```{admonition} Resumen
:class: important
**Conceptos CUDA:**
- `__global__` marca kernels ejecutables en GPU
- `<<<bloques, threads>>>` especifica configuración de lanzamiento
- `blockIdx.x * blockDim.x + threadIdx.x` calcula índice global
- `__syncthreads()` sincroniza threads del mismo bloque (no entre bloques)
- Coalescing: threads consecutivos → direcciones consecutivas
- CUDA streams permiten operaciones en paralelo
- Pinned memory acelera transferencias CPU↔GPU

**Conceptos Triton:**
- `@triton.jit` compila kernels JIT
- `tl.program_id(0)` identifica qué bloque de datos procesar
- `tl.arange()` crea vectores de índices
- `tl.load/store(..., mask=...)` con protección de límites
- Compilador maneja coalescing y sincronización automáticamente
- Piensa en bloques de datos, no threads individuales

**Checklist de kernel básico:**
- [ ] Indexación correcta: `blockIdx * blockDim + threadIdx` (CUDA) o `pid * BLOCK + arange` (Triton)
- [ ] Verificación de límites: `if (idx < n)` (CUDA) o `mask = offsets < n` (Triton)
- [ ] Coalescing: threads/offsets consecutivos acceden posiciones consecutivas
- [ ] Sincronización: `__syncthreads()` después de escribir shared memory
- [ ] Máscaras: Siempre usar `mask=` en `tl.load/store` (Triton)
```

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: Kernel lanzado con `<<<100, 256>>>`. Thread en `blockIdx=5, threadIdx=32`. ¿Índice global?
<details><summary>Respuesta</summary>
`idx = 5 * 256 + 32 = 1280 + 32 = 1312`
</details>

**Pregunta 2**: ¿Por qué `tl.load(ptr + offsets)` sin máscara es peligroso?
<details><summary>Respuesta</summary>
Si `BLOCK_SIZE > elementos_restantes`, intentas leer memoria fuera del vector → crash o datos basura. La máscara previene esto cargando `other=0.0` para offsets inválidos.
</details>
```

### Para Pensar

> *Si Triton genera código casi tan rápido como CUDA pero es mucho más fácil de escribir,
> ¿por qué aún se usa CUDA directamente en producción?*

---

## Tarea para Casa

### Ejercicio 1: Calcular Grid Size

Para procesar 10,000 elementos con bloques de 256 threads:
- ¿Cuántos bloques necesitas?
- Si tu GPU tiene 80 SMs y cada SM puede ejecutar 4 bloques, ¿cuántas waves?

### Ejercicio 2: Optimizar Coalescing

Reescribe para mejor coalescing:

```cuda
__global__ void transpose_naive(float *in, float *out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        // Escritura no coalescida
        out[x * n + y] = in[y * n + x];
    }
}
```

<details><summary>Hint</summary>
Usa shared memory para hacer tiling y separar lecturas coalescidas de escrituras coalescidas.
</details>

### Ejercicio 3: Debugging de Performance

Este kernel ReLU es lento. ¿Por qué?

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def slow_relu(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Procesar elemento por elemento (¡mal!)
    for i in range(n):
        pid = tl.program_id(0)
        if pid == 0:  # Solo programa 0 trabaja
            x = tl.load(x_ptr + i)
            y = tl.maximum(x, 0.0)
            tl.store(y_ptr + i, y)
```

<details><summary>Respuesta</summary>
1. **Serialización completa**: Loop procesa elementos uno a uno en vez de en paralelo.
2. **Solo un programa trabaja**: `if pid == 0` hace que todos los demás programas estén ociosos.
3. **Sin vectorización**: No aprovecha operaciones vectoriales del GPU.

**Solución**: Eliminar el loop y hacer que cada programa procese su bloque de BLOCK_SIZE elementos en paralelo.
</details>

---

## Próximos Pasos

En la siguiente sesión exploraremos **Optimización de Memoria GPU**: patrones de coalescing avanzados, uso de shared memory, tiling, y el modelo roofline para predecir bottlenecks.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). NVIDIA Developer.
- PyTorch. [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html). PyTorch Docs.
- Triton. [Triton Language and Compiler](https://github.com/triton-lang/triton). GitHub.
- Tillet, P., Kung, H.T., & Cox, D. (2019). [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). MAPL.
