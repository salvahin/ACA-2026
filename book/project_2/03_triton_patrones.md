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

# Triton Completo: API y Patrones de Kernels

```{code-cell} ipython3
import torch
import sys

# Validación temprana de entorno para cuadernos estrictamente acoplados a NVidia (Triton/vLLM)
if not torch.cuda.is_available():
    print("❌ ADVERTENCIA: Este notebook requiere una GPU NVidia y arquitectura CUDA para funcionar.")
    print("Por favor, sube este notebook a Google Colab y selecciona Entorno de ejecución -> Cambiar tipo de entorno -> T4 GPU o superior.")
    sys.exit("Entorno incompatible: Sistema sin CUDA detectado.")
else:
    print("✅ Entorno GPU detectado compatible con los requerimientos.")
```


```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q torch triton
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/03_triton_patrones.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 3
> **Tiempo de lectura:** ~45 minutos

---

## Introducción

Ya escribiste tu primer kernel Triton en la sesión anterior. Ahora aprenderás el **API completo** de `triton.language` y los patrones más importantes: reducciones y kernels fusionados.

Esta sesión cubre todo lo necesario para escribir kernels GPU de producción: desde operaciones básicas hasta implementaciones completas de softmax, layernorm y GELU que superan las versiones naive de PyTorch.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Dominar el API completo de `triton.language`: creación de vectores, load/store, aritmética, máscaras
- Aplicar indexación 2D con broadcasting (`[:, None]` y `[None, :]`)
- Implementar reducciones locales (`tl.sum` dentro de bloque) y globales (patrón de dos etapas)
- Usar operaciones atómicas (`tl.atomic_add`, `tl.atomic_max`) correctamente
- Escribir kernels fusionados completos: softmax, layernorm, GELU en una sola pasada
- Comparar y medir speedup vs implementaciones naive de PyTorch
```

---

## Contexto

En la sesión anterior aprendiste la filosofía "bloques, no threads" de Triton. Ahora profundizaremos en técnicas concretas que te permitirán escribir kernels GPU de alto rendimiento para operaciones comunes en deep learning.

---

## Bloque 1: API Completa de triton.language

### Creación de Vectores

Triton proporciona funciones para crear vectores de datos directamente:

```{code-cell} ipython3
:tags: [skip-execution]

import triton
import triton.language as tl

# Índices secuenciales
v = tl.arange(0, 256)           # [0, 1, 2, ..., 255]
v = tl.arange(0, 256, 2)        # [0, 2, 4, ..., 254] (con step=2)

# Vectores constantes
zeros = tl.zeros((256,), dtype=tl.float32)
ones = tl.ones((256,), dtype=tl.float32)
pi = tl.full((256,), 3.14159, dtype=tl.float32)
```

```{admonition} Consejo
:class: tip
`tl.arange()` es tu herramienta principal para crear índices. Siempre lo usarás para calcular qué posiciones de memoria procesar.
```

---

### Load y Store con Máscaras

El patrón básico de cualquier kernel: cargar datos, procesar, guardar.

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_example(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Cargar con máscara
    x = tl.load(x_ptr + offsets, mask=mask)

    # Cargar con valor por defecto para elementos fuera de límites
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Guardar con máscara
    tl.store(y_ptr + offsets, x * 2, mask=mask)
```

````{admonition} ⚠️ Antipatrón: Olvidar Máscaras
:class: warning
**Error común**: `x = tl.load(x_ptr + offsets)` sin máscara

**Problema**: Si `BLOCK_SIZE=256` pero `n=100`, accedes posiciones 100-255 que no existen → resultados incorrectos o crash

**Solución**: Siempre usar máscara:
```python
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```
````

---

### Operaciones Aritméticas

Triton soporta todas las operaciones matemáticas comunes:

```{code-cell} ipython3
:tags: [skip-execution]

# Operaciones básicas (elemento a elemento)
suma = a + b
resta = a - b
producto = a * b
division = a / b
potencia = a ** 2

# Funciones matemáticas
raiz = tl.sqrt(x)
exp_x = tl.exp(x)
log_x = tl.log(x)
sin_x = tl.sin(x)
cos_x = tl.cos(x)
abs_x = tl.abs(x)
tanh_x = tl.tanh(x)

# Comparaciones y selección
maximo = tl.maximum(a, b)
minimo = tl.minimum(a, b)
```

---

### Máscaras y Condicionales

Las máscaras son booleanos vectorizados que controlan qué elementos procesar:

```{code-cell} ipython3
:tags: [skip-execution]

# Crear máscaras
mask_bounds = offsets < n
mask_condition = x > threshold

# Combinar máscaras (AND, OR, NOT)
combined_and = mask_bounds & mask_condition
combined_or = mask_bounds | mask_condition
negated = ~mask_bounds

# Condicional vectorizado (como np.where)
result = tl.where(condition, value_if_true, value_if_false)

# Ejemplo: ReLU
result = tl.where(x > 0, x, 0.0)

# Ejemplo: Clipping
clipped = tl.where(x > max_val, max_val, x)
clipped = tl.where(clipped < min_val, min_val, clipped)
```

---

### Indexación 2D y Broadcasting

Para trabajar con matrices necesitas expandir vectores 1D a 2D:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_2d(matrix_ptr, height, width, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Índices locales 1D
    local_h = tl.arange(0, BLOCK_H)  # [0, 1, ..., BLOCK_H-1]
    local_w = tl.arange(0, BLOCK_W)  # [0, 1, ..., BLOCK_W-1]

    # Expandir a 2D con broadcasting
    # [:, None] expande verticalmente, [None, :] expande horizontalmente
    h_indices = pid_h * BLOCK_H + local_h[:, None]  # Shape: (BLOCK_H, 1)
    w_indices = pid_w * BLOCK_W + local_w[None, :]  # Shape: (1, BLOCK_W)

    # Broadcasting automático a (BLOCK_H, BLOCK_W)
    linear = h_indices * width + w_indices

    # Máscara 2D
    mask = (h_indices < height) & (w_indices < width)

    data = tl.load(matrix_ptr + linear, mask=mask, other=0.0)
```

```{admonition} Modelo Mental: Broadcasting
:class: hint
Piensa en el broadcasting como en NumPy:
- `local_h[:, None]` → columna vertical (BLOCK_H, 1)
- `local_w[None, :]` → fila horizontal (1, BLOCK_W)
- Al combinarlos → matriz completa (BLOCK_H, BLOCK_W)

Es como un producto cartesiano de índices.
```

---

### Reducciones Locales

Las reducciones colapsan un vector a un escalar dentro del bloque:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_reduce_local(data_ptr, result_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)

    # Reducciones - producen UN escalar por bloque
    suma = tl.sum(data)      # Suma de todos los elementos del vector
    maximo = tl.max(data)    # Máximo del vector
    minimo = tl.min(data)    # Mínimo del vector

    # Guardar (cada bloque produce 1 resultado)
    tl.store(result_ptr + pid, suma)
```

```{admonition} Importante
:class: warning
`tl.sum(data)` reduce el vector `data` a un único número. Si tu vector tiene 256 elementos, `tl.sum(data)` devuelve 1 escalar.
```

---

### Reducciones con Condiciones

Combinar máscaras con reducciones para sumar/max solo ciertos elementos:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def suma_condicional(data_ptr, result_ptr, threshold, n, BLOCK: tl.constexpr):
    """Suma solo elementos mayores que threshold"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)

    # Filtrar: reemplazar elementos que no cumplen condición con 0
    filtered = tl.where(data > threshold, data, 0.0)
    suma = tl.sum(filtered)

    tl.store(result_ptr + pid, suma)
```


---

### Reducciones Globales: Patrón de Dos Etapas

Para reducir un tensor completo (no solo un bloque), necesitas dos kernels:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_etapa1(data_ptr, temp_ptr, n, BLOCK: tl.constexpr):
    """Etapa 1: Cada bloque suma sus datos locales"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    local_sum = tl.sum(data)

    # Guardar suma parcial
    tl.store(temp_ptr + pid, local_sum)

@triton.jit
def kernel_etapa2(temp_ptr, result_ptr, num_bloques, BLOCK: tl.constexpr):
    """Etapa 2: Sumar todos los resultados parciales"""
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_bloques

    temp = tl.load(temp_ptr + offsets, mask=mask, other=0.0)
    total_sum = tl.sum(temp)

    tl.store(result_ptr, total_sum)

def suma_total(data):
    """Wrapper que ejecuta las dos etapas"""
    n = data.numel()
    BLOCK = 256
    num_bloques = triton.cdiv(n, BLOCK)

    # Etapa 1: Reducciones locales
    temp = torch.zeros(num_bloques, device='cuda')
    kernel_etapa1[(num_bloques,)](data, temp, n, BLOCK=BLOCK)

    # Etapa 2: Reducción final
    result = torch.zeros(1, device='cuda')
    kernel_etapa2[(1,)](temp, result, num_bloques, BLOCK=BLOCK)

    return result[0]
```

```{admonition} Por Qué Dos Etapas
:class: hint
**Problema**: Si tienes 1 millón de elementos y BLOCK_SIZE=256, necesitas ~4000 bloques. No puedes sincronizar 4000 bloques en un solo kernel.

**Solución**:
1. Cada bloque reduce sus 256 elementos a 1 número (4000 números)
2. Un segundo kernel reduce esos 4000 números a 1 resultado final

Es como hacer sumas parciales en equipos, luego sumar los totales de cada equipo.
```

---

### Operaciones Atómicas *(Opcional)*

Cuando múltiples threads necesitan escribir al mismo lugar:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def histogram_kernel(data_ptr, hist_ptr, n, num_bins, BLOCK: tl.constexpr):
    """Construir histograma usando operaciones atómicas"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)

    # Calcular bin para cada elemento
    bins = (data * num_bins).to(tl.int32)
    bins = tl.where(bins >= num_bins, num_bins - 1, bins)
    bins = tl.where(bins < 0, 0, bins)

    # Incrementar contadores (atómico porque múltiples threads escriben al mismo bin)
    for i in range(BLOCK):
        if offsets[i] < n:
            tl.atomic_add(hist_ptr + bins[i], 1)
```

```{admonition} ⚠️ Cuidado con Atómicas
:class: warning
Las operaciones atómicas son **lentas** porque serializan escrituras. Úsalas solo cuando sea absolutamente necesario. En muchos casos, puedes evitarlas con reducciones locales primero.
```

---

````{admonition} Actividad de Trabajo — Bloque 1 (25 minutos)
:class: note

**Ejercicio 1: Traducción CUDA → Triton (10 min)**

Dado el siguiente kernel CUDA de clipping, tradúcelo a Triton completo (firma,
`pid`, `offsets`, `mask`, `tl.load`, `tl.store`). Incluye un test con
`torch.allclose` para verificar corrección:

```cuda
__global__ void clip(float *x, float *y, float min_val, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        y[idx] = val;
    }
}
```

**Ejercicio 2: Reducciones Condicionales (10 min)**

Usando `suma_condicional` como base, implementa tres kernels separados para un
vector de floats:

a) Suma de todos los elementos positivos  
b) Cuántos elementos son mayores que 5.0 (usa `tl.sum` sobre una máscara de `int32`)  
c) Máximo de los elementos en el rango `[-10, 10]` — excluye los demás con `tl.where`

**Ejercicio 3: Indexación 2D (5 min)**

Para una matriz de 32×16 con `BLOCK_H=8, BLOCK_W=8`:

a) ¿Cuántos programas Triton se lanzan en total?  
b) El programa `(pid_h=1, pid_w=2)` ¿qué rango de filas y columnas procesa?  
c) Escribe la expresión de `linear = h_indices * width + w_indices` para ese programa.
````

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

Preguntas para discutir antes de continuar:

1. **Reducciones locales vs globales**: El patrón de dos etapas usa dos kernels
   separados. ¿Por qué no podemos sincronizar todos los bloques dentro de un único
   kernel Triton? ¿Cuándo usarías `tl.atomic_add` en vez del patrón de dos etapas?
2. **Broadcasting 2D**: Si olvidaras el `[:, None]` / `[None, :]` en la indexación
   2D, ¿obtendrías un error en tiempo de ejecución o resultados silenciosamente
   incorrectos? ¿Por qué?
3. **Máscaras obligatorias**: ¿Por qué Triton no lanza excepción automáticamente
   al hacer `tl.load` sin máscara en un vector de tamaño no múltiplo de `BLOCK_SIZE`?
   ¿Qué riesgo concreto representa ese silencio?
```

---

## Bloque 2: Kernels Fusionados

La fusión de operaciones es una de las optimizaciones más poderosas en GPU computing.

````{admonition} ⚡ Tip de Performance
:class: important
**Kernel fusionado vs múltiples kernels:**

Malo (3 kernels):
```python
y = x + 1
y = y * 2
y = y - 3
```
Requiere 3 round-trips a memoria → 6 accesos totales (3 loads + 3 stores)

Bueno (1 kernel fusionado):
```python
@triton.jit
def fused(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    y = (x + 1) * 2 - 3  # Todo en registros
    tl.store(y_ptr + offsets, y, mask=mask)
```
Solo 2 accesos (1 load + 1 store) → 3x menos tráfico de memoria
````

---

### Softmax Fusionado

Softmax requiere tres operaciones: encontrar el máximo, calcular exponenciales, y normalizar. Podemos hacerlo todo en una pasada:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def softmax_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    """
    Softmax fusionado y numéricamente estable.
    x: (M, N) - aplicar softmax a cada fila
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    # Cargar fila completa
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=-float('inf'))

    # Softmax numéricamente estable: softmax(x) = softmax(x - max(x))
    max_x = tl.max(x, axis=0)
    x = x - max_x

    # Exponencial y suma
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)

    # Normalizar
    softmax = exp_x / sum_exp

    tl.store(out_ptr + row * N + offsets, softmax, mask=mask)

def softmax_triton(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, out, M, N, BLOCK=BLOCK)
    return out
```

```{code-cell} ipython3
:tags: [skip-execution]

# Comparar con PyTorch
x = torch.randn(1000, 512, device='cuda')

# Versión PyTorch
torch_out = torch.softmax(x, dim=1)

# Versión Triton
triton_out = softmax_triton(x)

# Verificar corrección
assert torch.allclose(torch_out, triton_out, rtol=1e-5)

# Benchmark
import time

# Warm-up
for _ in range(10):
    torch.softmax(x, dim=1)
    softmax_triton(x)
torch.cuda.synchronize()

# PyTorch
start = time.time()
for _ in range(100):
    torch.softmax(x, dim=1)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / 100

# Triton
start = time.time()
for _ in range(100):
    softmax_triton(x)
torch.cuda.synchronize()
triton_time = (time.time() - start) / 100

print(f"PyTorch: {pytorch_time*1000:.3f} ms")
print(f"Triton:  {triton_time*1000:.3f} ms")
print(f"Speedup: {pytorch_time/triton_time:.2f}x")
```

**Por qué es más rápido:**
- PyTorch naive hace 3 pasadas: max, exp+sum, divide
- Triton fusionado hace 1 pasada: todo en registros
- Menos tráfico de memoria → 1.5-2x más rápido típicamente

---

### Layer Normalization

LayerNorm requiere calcular mean y variance, luego normalizar:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def layernorm_kernel(x_ptr, out_ptr, gamma_ptr, beta_ptr, M, N, eps, BLOCK: tl.constexpr):
    """
    LayerNorm fusionado: mean, variance, normalize en una pasada.
    x: (M, N)
    gamma, beta: (N,) - parámetros aprendibles
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    # Cargar datos
    x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)

    # Calcular mean
    mean = tl.sum(x, axis=0) / N

    # Calcular variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Normalizar
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # Aplicar gamma y beta
    out = gamma * x_norm + beta

    tl.store(out_ptr + row * N + offsets, out, mask=mask)

def layernorm_triton(x, gamma, beta, eps=1e-5):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    layernorm_kernel[(M,)](x, out, gamma, beta, M, N, eps, BLOCK=BLOCK)
    return out
```

```{code-cell} ipython3
:tags: [skip-execution]

# Comparar con PyTorch
batch, features = 1000, 512
x = torch.randn(batch, features, device='cuda')
gamma = torch.ones(features, device='cuda')
beta = torch.zeros(features, device='cuda')

# PyTorch
layer_norm = torch.nn.LayerNorm(features).cuda()
layer_norm.weight.data = gamma
layer_norm.bias.data = beta
torch_out = layer_norm(x)

# Triton
triton_out = layernorm_triton(x, gamma, beta)

print(f"Max diff: {(torch_out - triton_out).abs().max().item():.6f}")
print(f"Correct: {torch.allclose(torch_out, triton_out, rtol=1e-5)}")
```

**Ventaja de fusión:**
- PyTorch: 2 pasadas (calcular stats, luego normalizar)
- Triton: 1 pasada (todo junto)
- Típicamente 1.3-1.8x más rápido

---

### GELU Activation

GELU (Gaussian Error Linear Unit) es una activation común en transformers:

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    GELU activation fusionado.
    GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)

    # Constantes
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    coef = 0.044715

    # Calcular GELU
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + coef * x_cubed)
    gelu = 0.5 * x * (1.0 + tl.tanh(tanh_arg))

    tl.store(out_ptr + offsets, gelu, mask=mask)

def gelu_triton(x):
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    gelu_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out
```

```{code-cell} ipython3
:tags: [skip-execution]

# Comparar con PyTorch
x = torch.randn(1000000, device='cuda')

# PyTorch
torch_out = torch.nn.functional.gelu(x)

# Triton
triton_out = gelu_triton(x)

print(f"Max diff: {(torch_out - triton_out).abs().max().item():.6f}")
print(f"Correct: {torch.allclose(torch_out, triton_out, rtol=1e-4)}")
```

---

### Tabla de Speedups Típicos

| Operación | PyTorch Naive | Triton Fusionado | Speedup |
|-----------|---------------|------------------|---------|
| Softmax | 3 kernels (max, exp+sum, div) | 1 kernel | 1.5-2.0x |
| LayerNorm | 2 kernels (stats, normalize) | 1 kernel | 1.3-1.8x |
| GELU | Multiple ops | 1 kernel | 1.2-1.5x |
| ReLU + Dropout | 2 kernels | 1 kernel | 1.8-2.5x |

**Por qué funciona la fusión:**
- Menos round-trips a memoria global (principal cuello de botella)
- Datos en registros entre operaciones
- Mejor utilización de bandwidth
- Menor latencia total

---

## Actividad de Trabajo — Bloque 2 (25 minutos)

Cada equipo debe implementar **uno** de los siguientes kernels fusionados y hacer benchmark vs PyTorch:

### Opción 1: ReLU + Dropout Fusionado

```python
def relu_dropout_fused(x, p=0.5):
    """
    Fusionar: y = dropout(relu(x), p)
    Hints:
    - Generar random con tl.rand()
    - mask_dropout = tl.rand(...) > p
    - relu: tl.where(x > 0, x, 0)
    - Aplicar ambas máscaras
    """
    pass
```

### Opción 2: Scaled Softmax

```python
def scaled_softmax(x, scale=1.0):
    """
    Fusionar: softmax(x * scale)
    Usado en attention: softmax(Q @ K^T / sqrt(d))
    Hints:
    - Similar a softmax normal
    - Multiplicar por scale antes de max
    """
    pass
```

### Opción 3: Swish Activation

```python
def swish(x):
    """
    Swish: x * sigmoid(x) = x / (1 + exp(-x))
    Hints:
    - sigmoid(x) = 1 / (1 + exp(-x))
    - Fusionar multiplicación
    """
    pass
```

**Entregable:**
- Código del kernel
- Comparación de corrección vs PyTorch
- Benchmark mostrando speedup
- Explicación de por qué es más rápido

---

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: ¿Qué diferencia hay entre `tl.sum(x)` dentro de un kernel y el patrón de reducción de dos etapas?
<details><summary>Respuesta</summary>
`tl.sum(x)` reduce el vector dentro de <strong>un solo bloque</strong> (BLOCK_SIZE elementos) a un
escalar. Para vectores más grandes, el patrón de dos etapas primero produce N sumas
parciales (una por bloque) y luego un segundo kernel las reduce al total global.
</details>

**Pregunta 2**: ¿Por qué el softmax fusionado es numéricamente estable **y** más rápido que tres pasadas separadas?
<details><summary>Respuesta</summary>
Estable: restar el máximo antes de <code>tl.exp()</code> evita overflow
(<code>exp(1000)=∞</code>, pero <code>exp(0)=1</code>). Rápido: realiza
max+exp+sum+div en una sola pasada de memoria global vs 3 pasadas separadas —
el bandwidth de memoria deja de ser el cuello de botella.
</details>
```

### Para Pensar

> *Si en vez de softmax implementaras una reducción global sobre 1 billón de elementos,
> ¿cuántas etapas de two-pass reduction necesitarías? ¿Cuándo el patrón de dos etapas
> no es suficiente?*

---

## Resumen

```{admonition} Resumen
:class: important
**API de triton.language:**
- **Creación**: `tl.arange()`, `tl.zeros()`, `tl.ones()`, `tl.full()`
- **Memoria**: `tl.load(..., mask=...)`, `tl.store(..., mask=...)`
- **Aritmética**: `+`, `-`, `*`, `/`, `tl.sqrt`, `tl.exp`, `tl.log`, `tl.sin`, `tl.tanh`
- **Máscaras**: `tl.where(condition, true_val, false_val)`
- **Broadcasting 2D**: `[:, None]` y `[None, :]` para expandir dimensiones
- **Reducciones locales**: `tl.sum()`, `tl.max()`, `tl.min()` - dentro de bloque
- **Reducciones globales**: Patrón de dos etapas (local → temp → global)
- **Atómicas**: `tl.atomic_add()`, `tl.atomic_max()` - usar con cuidado

**Kernels fusionados:**
- **Concepto**: Múltiples operaciones en una sola pasada de memoria
- **Beneficio**: 1.5-2.5x speedup típico por reducción de tráfico de memoria
- **Ejemplos**: Softmax, LayerNorm, GELU - todos viables en producción
- **Cuándo fusionar**: Cuando operaciones son consecutivas y memory-bound

**Checklist de kernel:**
- [ ] Usar `mask=` en todos los `tl.load` y `tl.store`
- [ ] `BLOCK_SIZE: tl.constexpr` para valores de compilación
- [ ] Broadcasting correcto en 2D: `[:, None]` y `[None, :]`
- [ ] Para reducciones globales: implementar patrón de dos etapas
- [ ] Verificar corrección vs PyTorch antes de optimizar
```

---

## Tarea para Casa

### Ejercicio: Fusión Custom

Implementa un kernel Triton que calcule en una sola pasada:

```python
# y = tl.tanh(x * 0.5) * 2.0 + 1.0
```

Incluye:
1. Implementación con `@triton.jit` (firma completa, `pid`, `offsets`, `mask`, `tl.load/store`)
2. Verificación vs. PyTorch: `torch.tanh(x * 0.5) * 2.0 + 1.0`
3. Benchmark: ¿cuánto más rápido es vs 4 operaciones elementwise separadas en PyTorch?

---

## Referencias

- Triton. [Triton Language and Compiler](https://github.com/triton-lang/triton). GitHub.
- Tillet, P., Kung, H.T., & Cox, D. (2019). [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). MAPL.
- Triton. [Tutorials](https://triton-lang.org/main/getting-started/tutorials/). Triton Docs.
- NVIDIA. [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). NVIDIA Docs.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*
