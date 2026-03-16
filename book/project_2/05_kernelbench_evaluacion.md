---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# KernelBench y Pipeline de Evaluación

**Sesión 5 - Proyecto 2: Agente de Optimización de Kernels CUDA**

```{code-cell} ipython3
:tags: [skip-execution]

# Configuración para Google Colab
import sys
if 'google.colab' in sys.modules:
    !git clone https://github.com/salvahin/ACA-2026.git
    %cd ACA-2026/book/project_2
    !pip install torch triton datasets
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/project_2/05_kernelbench_evaluacion.ipynb)

---

## Objetivos de la Sesión

En esta sesión construiremos el **pipeline de evaluación** que nos permitirá medir objetivamente la calidad de los kernels CUDA generados por nuestro agente. Este pipeline será la base para todas las iteraciones del agente en las sesiones siguientes.

**Al final de esta sesión podrás:**

1. Comprender la estructura y niveles de dificultad de KernelBench
2. Identificar tipos de operaciones (elementwise, reducción, indexing, compuestas)
3. Construir un pipeline completo: compilación → ejecución → correctitud → performance
4. Evaluar kernels manualmente escritos usando el pipeline
5. Calcular scores agregados por nivel de dificultad

**Formato de la sesión:**
- **Bloque 1** (50 min): KernelBench L1-L4 (25' clase + 25' trabajo)
- **Discusión** (10 min): Clasificación de problemas
- **Bloque 2** (50 min): Pipeline de evaluación (25' clase + 25' trabajo)
- **Cierre** (10 min): Preparación para siguiente sesión

---

## Bloque 1: KernelBench L1-L4

### Introducción a KernelBench

**KernelBench** es un benchmark que contiene 250 operaciones de redes neuronales organizadas por niveles de dificultad. Fue diseñado para evaluar la capacidad de modelos de lenguaje y agentes para escribir kernels GPU correctos y eficientes que reemplacen implementaciones PyTorch.

```{admonition} Origen de KernelBench
:class: note
KernelBench fue creado por el **Scaling Intelligence Lab de Stanford** (Anne Ouyang, Simon Guo, Azalia Mirhoseini, et al.). El corpus está disponible en Hugging Face: `ScalingIntelligence/KernelBench`. Cita: *"KernelBench: Can LLMs Write Efficient GPU Kernels?"*, arXiv 2502.10517.
```

### Organización por Niveles de Dificultad

KernelBench organiza los problemas en 4 niveles:

| Nivel | Dificultad | Problemas | Descripción | Tiempo Estimado |
|-------|-----------|-----------|-------------|-----------------|
| **L1** | Básico | 100 | Single-kernel operators: matmul, conv, layernorm, elementwise. Los bloques fundamentales de redes neuronales | 15-30 min |
| **L2** | Intermedio | 100 | Patrones de fusión: operación principal (matmul/conv) + 2–5 epilogos (activaciones, norms, reductions) | 30-60 min |
| **L3** | Avanzado | 50 | Arquitecturas ML completas: AlexNet, MobileNet, MiniGPT, VGG | 1-2 horas |
| **L4** | Aspiracional | 20 | Arquitecturas de HuggingFace reales (requieren modificar código fuente de la librería) | 2-4 horas |

```{code-cell} ipython3
:tags: [skip-execution]

from datasets import load_dataset

# Cargar KernelBench desde Hugging Face
dataset = load_dataset("ScalingIntelligence/KernelBench", split="train")

print(f"Total de problemas: {len(dataset)}")
print(f"\nDistribución por nivel:")
for level in ["L1", "L2", "L3", "L4"]:
    count = sum(1 for item in dataset if item["level"] == level)
    print(f"  {level}: {count} problemas")
```

### Tipos de Operaciones

Los problemas de KernelBench se pueden clasificar en 4 categorías según el tipo de operación:

#### 1. Operaciones Elementwise

Aplican la misma operación a cada elemento independientemente. Son las más simples y se paralelizan fácilmente.

**Ejemplos:**
- `vector_add`: `c[i] = a[i] + b[i]`
- `relu`: `y[i] = max(0, x[i])`
- `gelu`: activación GELU elemento por elemento
- `silu`: activación SiLU (Swish)

```{code-cell} ipython3
:tags: [skip-execution]

# Ejemplo L1: Vector Add
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Suma dos vectores elemento por elemento."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper para el kernel de suma de vectores."""
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

# Test
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')

output_triton = vector_add(x, y)
output_torch = x + y

print(f"Correctitud: {torch.allclose(output_triton, output_torch, rtol=1e-5)}")
```

#### 2. Operaciones de Reducción

Combinan múltiples elementos en un resultado más pequeño. Requieren estrategias de sincronización.

**Ejemplos:**
- `sum`: suma todos los elementos
- `mean`: promedio de elementos
- `softmax`: reducción + normalización
- `layernorm`: normalización por capa

```{code-cell} ipython3
:tags: [skip-execution]

# Ejemplo L2: Sum Reduction
@triton.jit
def sum_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Suma todos los elementos de un vector."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)

    tl.atomic_add(output_ptr, block_sum)

def sum_reduction(x: torch.Tensor) -> torch.Tensor:
    """Wrapper para el kernel de suma."""
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sum_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output

# Test
x = torch.randn(10000, device='cuda')

output_triton = sum_reduction(x)
output_torch = x.sum()

print(f"Triton: {output_triton.item():.4f}")
print(f"PyTorch: {output_torch.item():.4f}")
print(f"Correctitud: {torch.allclose(output_triton, output_torch, rtol=1e-5)}")
```

#### 3. Operaciones con Indexing Complejo

Requieren patrones de acceso no triviales a memoria. Incluyen transposiciones, slicing, gather/scatter.

**Ejemplos:**
- `transpose`: intercambio de dimensiones
- `gather`: selección por índices
- `scatter`: escritura por índices
- `index_select`: selección de filas/columnas

```{code-cell} ipython3
:tags: [skip-execution]

# Ejemplo L3: Matrix Transpose
@triton.jit
def transpose_kernel(
    x_ptr, output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """Transpone una matriz M×N a N×M."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Cargar bloque de entrada
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Leer con patrón [M, N]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :])

    # Escribir con patrón transpuesto [N, M]
    output_ptrs = output_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(output_ptrs, x, mask=mask_n[:, None] & mask_m[None, :])

def transpose(x: torch.Tensor) -> torch.Tensor:
    """Wrapper para el kernel de transposición."""
    M, N = x.shape
    output = torch.empty((N, M), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 32
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    transpose_kernel[grid](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE)

    return output

# Test
x = torch.randn(1024, 512, device='cuda')

output_triton = transpose(x)
output_torch = x.T

print(f"Shape input: {x.shape}")
print(f"Shape output: {output_triton.shape}")
print(f"Correctitud: {torch.allclose(output_triton, output_torch, rtol=1e-5)}")
```

#### 4. Operaciones Compuestas

Combinan múltiples pasos en un solo kernel (fusión de operaciones). Requieren optimizaciones avanzadas.

**Ejemplos:**
- `layernorm`: mean + var + normalize + scale
- `softmax`: exp + sum + divide
- `fused_attention`: QK^T + scale + softmax + V
- `rmsnorm`: rms + normalize + scale

```{code-cell} ipython3
:tags: [skip-execution]

# Ejemplo L3: LayerNorm (simplificado)
@triton.jit
def layernorm_kernel(
    x_ptr, output_ptr, weight_ptr, bias_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias."""
    pid = tl.program_id(axis=0)

    # Cargar datos
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x_ptrs = x_ptr + pid * N + offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Paso 1: Calcular mean
    mean = tl.sum(x, axis=0) / N

    # Paso 2: Calcular variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Paso 3: Normalizar
    rstd = 1.0 / tl.sqrt(var + eps)
    x_normalized = x_centered * rstd

    # Paso 4: Aplicar affine transform
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    output = x_normalized * weight + bias

    # Guardar resultado
    output_ptrs = output_ptr + pid * N + offsets
    tl.store(output_ptrs, output, mask=mask)

def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    """Wrapper para el kernel de LayerNorm."""
    M, N = x.shape
    output = torch.empty_like(x)

    grid = (M,)
    layernorm_kernel[grid](x, output, weight, bias, N, eps, BLOCK_SIZE=1024)

    return output

# Test
M, N = 256, 512
x = torch.randn(M, N, device='cuda')
weight = torch.randn(N, device='cuda')
bias = torch.randn(N, device='cuda')

output_triton = layernorm(x, weight, bias)
output_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias)

print(f"Correctitud: {torch.allclose(output_triton, output_torch, rtol=1e-3)}")
```

### Formato de Problema en KernelBench

Cada problema en KernelBench tiene el siguiente formato JSON:

```json
{
  "problem_id": "vector_add_L1",
  "level": "L1",
  "name": "Vector Addition",
  "description": "Suma dos vectores elemento por elemento",
  "specification": {
    "inputs": [
      {"name": "x", "shape": "[N]", "dtype": "float32"},
      {"name": "y", "shape": "[N]", "dtype": "float32"}
    ],
    "outputs": [
      {"name": "z", "shape": "[N]", "dtype": "float32"}
    ],
    "formula": "z[i] = x[i] + y[i]",
    "constraints": ["N >= 1", "N <= 10^6"]
  },
  "test_cases": [
    {"N": 1000, "seed": 42},
    {"N": 10000, "seed": 123},
    {"N": 100000, "seed": 456}
  ],
  "reference_impl": "torch.add",
  "tags": ["elementwise", "basic"]
}
```

```{code-cell} ipython3
:tags: [skip-execution]

# Cargar y explorar un problema específico
problem = dataset[0]

print(f"Problem ID: {problem['problem_id']}")
print(f"Level: {problem['level']}")
print(f"Name: {problem['name']}")
print(f"\nDescription:")
print(problem['description'])
print(f"\nSpecification:")
print(f"  Inputs: {problem['specification']['inputs']}")
print(f"  Outputs: {problem['specification']['outputs']}")
print(f"  Formula: {problem['specification']['formula']}")
print(f"\nTest Cases:")
for tc in problem['test_cases']:
    print(f"  {tc}")
```

---

## Actividad de Trabajo — Bloque 1 (25 minutos)

**Ejercicio 1: Clasificar problemas L1 de KernelBench (15 min)**

Carga el dataset y explora mínimo 6 problemas de nivel L1. Para cada uno completa la
siguiente tabla (usa `problem['tags']` y `problem['specification']`):

| Problem ID | Tipo de op. | Variables de entrada | Conceptos Triton necesarios | Dificultad estimada (1-3) |
|------------|-------------|----------------------|-----------------------------|---------------------------|
| vector_add_L1 | elementwise | x[N], y[N] | `program_id`, `arange`, `load/store`, `mask` | 1 |
| ... | | | | |

Incluye al menos: 2 elementwise, 2 reducciones, 1 indexing, 1 compuesta.

**Ejercicio 2: Predicción de dificultad relativa (10 min)**

Para los mismos 6 problemas, ordena de más fácil a más difícil y justifica usando
como criterio:

- Número de pasadas de memoria (1, 2, o más)
- Necesidad de sincronización (ninguna / dentro de bloque / global)
- Tipo de indexación (lineal 1D / 2D con broadcasting / scatter/gather)

¿Coincide tu orden con los niveles L1/L2/L3 del dataset? ¿Hay algún problema L1 que
creiste que sería L2?

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

1. **Criterio de clasificación**: ¿Por qué `sum` (reducción) es más difícil que `relu`
   (elementwise) si ambas son L1? ¿Qué explota conceptualmente la diferencia?
2. **Indexing vs compuesta**: El `transpose` usa indexing 2D pero no fusiona operaciones;
   el `layernorm` fusiona 4 pasos. ¿Cuál creen que es más difícil de generar
   automáticamente con un LLM? Justifiquen.
3. **L3 y L4**: ¿Qué cambio cualitativo esperan entre escribir un kernel L1 individual
   y escribir el kernel de MiniGPT completo (L3)? ¿Es cuestión solo de líneas de
   código o hay algo más?
```

---

## Bloque 2: Pipeline de Evaluación

### Visión General del Pipeline

Para evaluar la calidad de un kernel CUDA, necesitamos un pipeline que verifique:

1. **Compilación**: ¿El código es sintácticamente válido?
2. **Ejecución**: ¿El kernel corre sin crashes?
3. **Correctitud**: ¿Los resultados son numéricamente correctos?
4. **Performance**: ¿Qué tan rápido es vs PyTorch?

```{admonition} PyTorch como Ground Truth
:class: important
Usamos PyTorch como implementación de referencia para verificar correctitud. La comparación se hace con `torch.allclose(rtol=1e-5, atol=1e-8)`.
```

### Pipeline Paso a Paso

#### Paso 1: Compilación

Detectar errores de sintaxis antes de ejecutar.

```{code-cell} ipython3
:tags: [skip-execution]

import traceback
from typing import Tuple, Optional

def step1_compile(kernel_code: str) -> Tuple[bool, Optional[str]]:
    """
    Intenta compilar el código del kernel.

    Returns:
        (success, error_message)
    """
    try:
        # Intentar compilar el código
        compile(kernel_code, '<kernel>', 'exec')
        return True, None
    except SyntaxError as e:
        error_msg = f"SyntaxError en línea {e.lineno}: {e.msg}"
        return False, error_msg
    except Exception as e:
        error_msg = f"Error de compilación: {type(e).__name__}: {str(e)}"
        return False, error_msg

# Test
good_code = """
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pass
"""

bad_code = """
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr)
    pass  # Falta ":"
"""

print("Código válido:")
success, error = step1_compile(good_code)
print(f"  Success: {success}, Error: {error}")

print("\nCódigo inválido:")
success, error = step1_compile(bad_code)
print(f"  Success: {success}, Error: {error}")
```

#### Paso 2: Ejecución sin Crash

Verificar que el kernel ejecuta sin errores de runtime.

```{code-cell} ipython3
:tags: [skip-execution]

def step2_execute(kernel_fn, *args, **kwargs) -> Tuple[bool, Optional[str]]:
    """
    Intenta ejecutar el kernel con los argumentos dados.

    Returns:
        (success, error_message)
    """
    try:
        # Ejecutar kernel
        result = kernel_fn(*args, **kwargs)

        # Sincronizar GPU para detectar errores CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return True, None
    except RuntimeError as e:
        error_msg = f"RuntimeError: {str(e)}"
        return False, error_msg
    except Exception as e:
        error_msg = f"Error de ejecución: {type(e).__name__}: {str(e)}"
        return False, error_msg

# Test
def good_kernel():
    x = torch.randn(1000, device='cuda')
    return vector_add(x, x)

def bad_kernel():
    # Intentar acceder fuera de límites
    x = torch.randn(1000, device='cuda')
    # Simulación de kernel con error
    raise RuntimeError("CUDA error: invalid argument")

print("Kernel válido:")
success, error = step2_execute(good_kernel)
print(f"  Success: {success}, Error: {error}")

print("\nKernel inválido:")
success, error = step2_execute(bad_kernel)
print(f"  Success: {success}, Error: {error}")
```

#### Paso 3: Correctitud Numérica

Comparar con la implementación de PyTorch.

```{code-cell} ipython3
:tags: [skip-execution]

def step3_correctness(
    output_triton: torch.Tensor,
    output_torch: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Tuple[bool, Optional[str]]:
    """
    Verifica correctitud numérica comparando con PyTorch.

    Returns:
        (correct, error_message)
    """
    try:
        # Verificar shapes
        if output_triton.shape != output_torch.shape:
            return False, f"Shape mismatch: {output_triton.shape} vs {output_torch.shape}"

        # Verificar valores
        if torch.allclose(output_triton, output_torch, rtol=rtol, atol=atol):
            return True, None
        else:
            # Calcular error máximo para debug
            abs_diff = torch.abs(output_triton - output_torch)
            max_diff = abs_diff.max().item()
            rel_diff = (abs_diff / (torch.abs(output_torch) + 1e-10)).max().item()

            error_msg = f"Valores incorrectos (max_abs_diff={max_diff:.2e}, max_rel_diff={rel_diff:.2e})"
            return False, error_msg

    except Exception as e:
        error_msg = f"Error en comparación: {type(e).__name__}: {str(e)}"
        return False, error_msg

# Test
x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')

output_triton = vector_add(x, y)
output_torch = x + y

print("Comparación correcta:")
correct, error = step3_correctness(output_triton, output_torch)
print(f"  Correct: {correct}, Error: {error}")

print("\nComparación incorrecta:")
output_wrong = output_triton + 0.1  # Agregar error
correct, error = step3_correctness(output_wrong, output_torch)
print(f"  Correct: {correct}, Error: {error}")
```

#### Paso 4: Performance

Medir speedup vs PyTorch con warmup y sincronización.

```{code-cell} ipython3
:tags: [skip-execution]

import time

def step4_benchmark(
    kernel_fn,
    torch_fn,
    args_kernel,
    args_torch,
    n_warmup: int = 10,
    n_iters: int = 100
) -> Tuple[float, float, float]:
    """
    Mide el tiempo de ejecución del kernel vs PyTorch.

    Returns:
        (time_triton_ms, time_torch_ms, speedup)
    """
    # Warmup GPU
    for _ in range(n_warmup):
        _ = kernel_fn(*args_kernel)
        _ = torch_fn(*args_torch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = kernel_fn(*args_kernel)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_triton = (time.perf_counter() - start) / n_iters * 1000  # ms

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch_fn(*args_torch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_torch = (time.perf_counter() - start) / n_iters * 1000  # ms

    speedup = time_torch / time_triton

    return time_triton, time_torch, speedup

# Test
x = torch.randn(100000, device='cuda')
y = torch.randn(100000, device='cuda')

time_triton, time_torch, speedup = step4_benchmark(
    kernel_fn=lambda: vector_add(x, y),
    torch_fn=lambda: x + y,
    args_kernel=(),
    args_torch=(),
    n_warmup=10,
    n_iters=100
)

print(f"Tiempo Triton: {time_triton:.3f} ms")
print(f"Tiempo PyTorch: {time_torch:.3f} ms")
print(f"Speedup: {speedup:.2f}x")
```

### Pipeline Completo

Ahora juntamos todos los pasos en una función unificada.

```{code-cell} ipython3
:tags: [skip-execution]

from dataclasses import dataclass
from typing import Callable, Any, Dict

@dataclass
class EvaluationResult:
    """Resultado de evaluar un kernel."""
    problem_id: str
    compiled: bool
    executed: bool
    correct: bool
    time_triton_ms: float
    time_torch_ms: float
    speedup: float
    score: float
    error_message: Optional[str] = None

def evaluate_kernel(
    problem_id: str,
    kernel_code: str,
    kernel_fn: Callable,
    torch_fn: Callable,
    test_inputs: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    n_warmup: int = 10,
    n_iters: int = 100
) -> EvaluationResult:
    """
    Pipeline completo de evaluación de un kernel.

    Args:
        problem_id: Identificador del problema
        kernel_code: Código fuente del kernel
        kernel_fn: Función wrapper del kernel Triton
        torch_fn: Función de referencia PyTorch
        test_inputs: Diccionario con tensores de entrada
        rtol, atol: Tolerancias para correctitud
        n_warmup, n_iters: Parámetros de benchmark

    Returns:
        EvaluationResult con todos los resultados
    """
    result = EvaluationResult(
        problem_id=problem_id,
        compiled=False,
        executed=False,
        correct=False,
        time_triton_ms=float('inf'),
        time_torch_ms=0.0,
        speedup=0.0,
        score=0.0
    )

    # Paso 1: Compilación
    compiled, error = step1_compile(kernel_code)
    result.compiled = compiled
    if not compiled:
        result.error_message = f"[COMPILE] {error}"
        return result

    # Paso 2: Ejecución
    try:
        output_triton = kernel_fn(**test_inputs)
        result.executed = True
    except Exception as e:
        result.error_message = f"[EXECUTE] {type(e).__name__}: {str(e)}"
        return result

    # Paso 3: Correctitud
    try:
        output_torch = torch_fn(**test_inputs)
        correct, error = step3_correctness(output_triton, output_torch, rtol, atol)
        result.correct = correct
        if not correct:
            result.error_message = f"[CORRECTNESS] {error}"
            return result
    except Exception as e:
        result.error_message = f"[CORRECTNESS] {type(e).__name__}: {str(e)}"
        return result

    # Paso 4: Performance
    try:
        time_triton, time_torch, speedup = step4_benchmark(
            kernel_fn=lambda: kernel_fn(**test_inputs),
            torch_fn=lambda: torch_fn(**test_inputs),
            args_kernel=(),
            args_torch=(),
            n_warmup=n_warmup,
            n_iters=n_iters
        )
        result.time_triton_ms = time_triton
        result.time_torch_ms = time_torch
        result.speedup = speedup
    except Exception as e:
        result.error_message = f"[BENCHMARK] {type(e).__name__}: {str(e)}"
        # No retornamos aquí, el kernel es correcto aunque el benchmark falle

    # Calcular score final
    result.score = calculate_score(result)

    return result

def calculate_score(result: EvaluationResult,
                   weight_correctness: float = 0.5,
                   weight_performance: float = 0.5) -> float:
    """
    Calcula el score agregado de un kernel.

    Score = weight_correctness * correctness + weight_performance * normalized_speedup

    - correctness: 1.0 si pasa, 0.0 si falla
    - normalized_speedup: min(speedup / 2.0, 1.0)  # 2x = 100%
    """
    if not result.correct:
        return 0.0

    correctness_score = 1.0

    # Normalizar speedup: 2x = score máximo de 1.0
    normalized_speedup = min(result.speedup / 2.0, 1.0)
    performance_score = normalized_speedup

    total_score = (
        weight_correctness * correctness_score +
        weight_performance * performance_score
    )

    return total_score

# Test del pipeline completo
print("=== Evaluación de vector_add ===\n")

kernel_code = """
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""

x = torch.randn(100000, device='cuda')
y = torch.randn(100000, device='cuda')

result = evaluate_kernel(
    problem_id="vector_add_L1",
    kernel_code=kernel_code,
    kernel_fn=vector_add,
    torch_fn=lambda x, y: x + y,
    test_inputs={"x": x, "y": y}
)

print(f"Problem: {result.problem_id}")
print(f"Compiled: {result.compiled}")
print(f"Executed: {result.executed}")
print(f"Correct: {result.correct}")
print(f"Time Triton: {result.time_triton_ms:.3f} ms")
print(f"Time PyTorch: {result.time_torch_ms:.3f} ms")
print(f"Speedup: {result.speedup:.2f}x")
print(f"Score: {result.score:.3f}")
if result.error_message:
    print(f"Error: {result.error_message}")
```

### Score Agregado por Nivel

Para evaluar múltiples problemas, agregamos los scores por nivel.

```{code-cell} ipython3
:tags: [skip-execution]

from collections import defaultdict
from typing import List

def aggregate_scores_by_level(results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
    """
    Agrega scores por nivel de dificultad.

    Returns:
        Dict con estadísticas por nivel: {
            "L1": {"mean_score": 0.85, "pass_rate": 0.9, "mean_speedup": 1.5},
            ...
        }
    """
    by_level = defaultdict(list)

    # Agrupar por nivel
    for result in results:
        level = result.problem_id.split("_")[-1]  # Asumimos formato "name_L1"
        by_level[level].append(result)

    # Calcular estadísticas
    stats = {}
    for level, level_results in by_level.items():
        n_total = len(level_results)
        n_correct = sum(1 for r in level_results if r.correct)

        mean_score = sum(r.score for r in level_results) / n_total
        pass_rate = n_correct / n_total

        # Speedup solo de los correctos
        correct_speedups = [r.speedup for r in level_results if r.correct]
        mean_speedup = sum(correct_speedups) / len(correct_speedups) if correct_speedups else 0.0

        stats[level] = {
            "n_problems": n_total,
            "n_correct": n_correct,
            "mean_score": mean_score,
            "pass_rate": pass_rate,
            "mean_speedup": mean_speedup
        }

    return stats

# Test
results = [
    EvaluationResult("vector_add_L1", True, True, True, 0.05, 0.08, 1.6, 0.9),
    EvaluationResult("relu_L1", True, True, True, 0.04, 0.07, 1.75, 0.925),
    EvaluationResult("sum_L1", True, True, False, float('inf'), 0.1, 0.0, 0.0),
    EvaluationResult("transpose_L2", True, True, True, 0.2, 0.3, 1.5, 0.85),
]

stats = aggregate_scores_by_level(results)

print("=== Estadísticas por Nivel ===\n")
for level, stat in sorted(stats.items()):
    print(f"{level}:")
    print(f"  Problemas: {stat['n_problems']}")
    print(f"  Correctos: {stat['n_correct']}")
    print(f"  Pass rate: {stat['pass_rate']:.1%}")
    print(f"  Score promedio: {stat['mean_score']:.3f}")
    print(f"  Speedup promedio: {stat['mean_speedup']:.2f}x")
    print()
```

### Estrategias de Selección de Problemas

Para entrenar o evaluar un agente, necesitamos seleccionar subconjuntos de KernelBench estratégicamente.

```{code-cell} ipython3
:tags: [skip-execution]

from typing import List
import random

def select_by_level(dataset, level: str, n: int = 10, seed: int = 42) -> List[Dict]:
    """Selecciona n problemas aleatorios de un nivel."""
    random.seed(seed)
    problems = [p for p in dataset if p["level"] == level]
    return random.sample(problems, min(n, len(problems)))

def select_by_type(dataset, operation_type: str, n: int = 10, seed: int = 42) -> List[Dict]:
    """Selecciona n problemas de un tipo de operación."""
    random.seed(seed)
    problems = [p for p in dataset if operation_type in p.get("tags", [])]
    return random.sample(problems, min(n, len(problems)))

def select_balanced(dataset, n_per_level: int = 5, seed: int = 42) -> List[Dict]:
    """Selecciona n_per_level problemas de cada nivel (balanceado)."""
    random.seed(seed)
    problems = []
    for level in ["L1", "L2", "L3", "L4"]:
        level_problems = select_by_level(dataset, level, n_per_level, seed)
        problems.extend(level_problems)
    return problems

def select_roi(dataset, completed_problems: set, level_weights: Dict[str, float]) -> List[Dict]:
    """
    Selecciona problemas con mayor ROI (Return on Investment).

    Prioriza:
    - Problemas no resueltos
    - Niveles con mayor peso (por ej., L1 > L2 > L3 > L4 al inicio)
    - Diversidad de tipos de operación
    """
    candidates = [p for p in dataset if p["problem_id"] not in completed_problems]

    # Ordenar por peso del nivel (descendente)
    candidates.sort(key=lambda p: level_weights.get(p["level"], 0.0), reverse=True)

    return candidates

# Test
print("=== Selección Balanceada ===")
selected = select_balanced(dataset, n_per_level=3)
print(f"Total seleccionados: {len(selected)}")
for level in ["L1", "L2", "L3", "L4"]:
    count = sum(1 for p in selected if p["level"] == level)
    print(f"  {level}: {count} problemas")

print("\n=== Selección por ROI ===")
completed = {"vector_add_L1", "relu_L1"}
level_weights = {"L1": 1.0, "L2": 0.7, "L3": 0.4, "L4": 0.2}
roi_candidates = select_roi(dataset, completed, level_weights)
print(f"Top 5 candidatos:")
for p in roi_candidates[:5]:
    print(f"  {p['problem_id']} ({p['level']})")
```

---

## Actividad de Trabajo — Bloque 2 (25 minutos)

**Ejercicio 1: Implementar y evaluar 4 kernels L1 (20 min)**

Elige 4 problemas L1 del Bloque 1 (al menos 1 elementwise, 1 reducción, 1 indexing o
compuesta) e imíple cada uno con:

1. `@triton.jit` kernel + wrapper Python
2. Test de correctitud: `torch.allclose(output_triton, output_torch, rtol=1e-5)`
3. Evaluación completa con `evaluate_kernel()`

Completa esta tabla con tus resultados:

| Problem ID | Compiled | Executed | Correct | Speedup | Score | Notas |
|------------|----------|----------|---------|---------|-------|-------|
| ... | ✓/✗ | ✓/✗ | ✓/✗ | Nx | 0.00 | |

**Consejo**: Empieza por los elementwise (más simples), luego intenta una reducción.
Para la reducción, usa el patrón `tl.sum` + `tl.atomic_add` visto en la sesión 3.

**Ejercicio 2: Diagnosticar un kernel fallido (5 min)**

El siguiente kernel tiene un bug. Ejectuá `evaluate_kernel()` sobre él, lee el
`error_message` del `EvaluationResult`, y corriges sin mirar la solución:

```python
@triton.jit
def buggy_relu_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)       # ¿Problema?
    y = tl.where(x < 0, x, 0.0)        # ¿Lógica correcta?
    tl.store(y_ptr + offsets, y)        # ¿Problema?

def buggy_relu(x): ...
```

¿En qué paso falla el pipeline (compile / execute / correctness)? ¿Cuál es la causa?

---

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: Un kernel pasa `step1_compile` y `step2_execute` pero falla `step3_correctness`
con `max_abs_diff=0.12`. ¿Cuáles son las dos causas más probables?
<details><summary>Respuesta</summary>
(a) Lógica incorrecta en el kernel (ej. operación equivocada, condición invertida
como <code>tl.where(x &lt; 0, x, 0)</code> en vez de ReLU). (b) Problema de precisión numérica
(ej. acumulación en FP16 vs FP32, o falta de eps en división). Ambas producen un
kernel que “corre” pero devuelve valores erróneos.
</details>

**Pregunta 2**: `calculate_score` devuelve 0 para un kernel que mide 0.8x de speedup
(más lento que PyTorch). ¿Eso es correcto? ¿Por qué?
<details><summary>Respuesta</summary>
Sí, si el kernel fue <strong>correcto</strong> pero más lento, el score será
<code>0.5 × 1.0 + 0.5 × min(0.8/2, 1.0) = 0.5 + 0.2 = 0.70</code>, no 0.
El score es 0 solo cuando <code>result.correct == False</code>.
Un speedup de 0.8x aún recibe crédito parcial por ser correcto.
</details>
```

### Para Pensar

> *En las sesiones 6–8 el agente usará `evaluate_kernel()` para decidir si una
> propuesta es válida. ¿Qué pasaría si el agente aprendiera a escribir kernels
> que superan el pipeline de evaluación sin ser realmente rápidos en producción?
> ¿Qué cambiarías en el pipeline para evitarlo?*

---

## Checklist de Implementación

Antes de pasar a la siguiente sesión, verifica que puedas:

- [ ] Cargar KernelBench desde Hugging Face
- [ ] Filtrar problemas por nivel (L1, L2, L3, L4)
- [ ] Identificar el tipo de operación de un problema
- [ ] Compilar código de kernel y detectar errores de sintaxis
- [ ] Ejecutar kernel y detectar crashes
- [ ] Verificar correctitud con `torch.allclose`
- [ ] Medir performance con warmup y sincronización
- [ ] Calcular score agregado (correctness + performance)
- [ ] Evaluar múltiples kernels con el pipeline
- [ ] Generar estadísticas por nivel
- [ ] Seleccionar subconjuntos de problemas estratégicamente

```{admonition} Importancia del Pipeline
:class: important
Este pipeline será el **núcleo de evaluación** del agente en las sesiones 6-8. Cada iteración del agente usará `evaluate_kernel()` para decidir si un kernel es aceptable y si necesita optimizarse más.
```

---

## Resumen

En esta sesión construimos el pipeline de evaluación completo para KernelBench:

**Conceptos clave:**
1. **KernelBench L1-L4**: 80 problemas organizados por dificultad
2. **Tipos de operaciones**: elementwise, reducción, indexing, compuestas
3. **Pipeline de evaluación**: compilación → ejecución → correctitud → performance
4. **Métricas**: correctness (torch.allclose) + speedup vs PyTorch
5. **Score agregado**: weighted sum de correctness y performance
6. **Estrategias de selección**: balanceada, por tipo, ROI

**Flujo del pipeline:**
```
Kernel code
    ↓
[1] Compile → SyntaxError?
    ↓
[2] Execute → RuntimeError?
    ↓
[3] Correctness → torch.allclose?
    ↓
[4] Performance → speedup vs PyTorch
    ↓
Score = 0.5 * correct + 0.5 * min(speedup/2, 1.0)
```

**Próxima sesión:**
En la Sesión 6 construiremos el agente que usa este pipeline para iterar automáticamente sobre un kernel, mejorándolo hasta alcanzar el score objetivo.

---

## Tarea para Casa

### Ejercicio 1: Resolver 10 problemas L1 (★★★)

Amplía la tabla de la clase a 10 problemas L1 diversos:

1. Selecciona al menos 2 elementwise, 2 reducciones, 2 indexing y 2 compuestas.
2. Implementa cada kernel en Triton con wrapper.
3. Evalúa con `evaluate_kernel()`.
4. Para los kernels correctos, agrega `@triton.autotune` y mide el speedup con la
   mejor configuración.
5. Objetivo: speedup > 1.5x en al menos 7 de los 10.

Genera una tabla de resultados agregados usando `aggregate_scores_by_level()`.

### Ejercicio 2: Extender el pipeline (★★☆)

Agrega al `evaluate_kernel()` un paso 0 de validación de shapes:
- Antes de ejecutar, verifica que los tensores de entrada tengan el dtype correcto
  (`float32`) y estén en GPU.
- Si alguno está en CPU, mueve automáticamente y agrega un warning al `error_message`.
- Tests: un caso sin GPU, un caso con `dtype=float16`, un caso correcto.

---

## Referencias

1. **KernelBench Dataset**
   - Hugging Face: `ScalingIntelligence/KernelBench`
   - Paper: "KernelBench: Can LLMs Write Efficient GPU Kernels?" (Ouyang et al., arXiv 2502.10517)
   - Leaderboard: https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/

2. **Triton Documentation**
   - Guía de programación: https://triton-lang.org/main/programming-guide/index.html
   - Tutorials: https://triton-lang.org/main/getting-started/tutorials/index.html

3. **PyTorch Testing**
   - `torch.testing.assert_close()`: https://pytorch.org/docs/stable/testing.html
   - `torch.allclose()`: https://pytorch.org/docs/stable/generated/torch.allclose.html

4. **CUDA Best Practices**
   - Memory coalescing: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
   - Occupancy calculator: https://docs.nvidia.com/cuda/cuda-occupancy-calculator/

5. **Benchmarking**
   - PyTorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
   - CUDA events: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
