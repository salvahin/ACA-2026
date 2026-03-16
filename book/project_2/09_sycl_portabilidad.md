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

# Sesión 9: SYCL, Portabilidad y Multi-target

```{code-cell} ipython3
:tags: [remove-cell]

# Configuración para Google Colab
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Ejecutando en Google Colab")
    # Nota: SYCL requiere hardware específico (Intel GPUs/CPUs)
    # Esta sesión es conceptual - los ejemplos SYCL no se ejecutarán
else:
    print("Ejecutando localmente")
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/project_2/09_sycl_portabilidad.ipynb)

## Introducción

Hasta ahora, hemos trabajado exclusivamente con CUDA y Triton para GPUs NVIDIA. Pero, ¿qué pasa si queremos ejecutar nuestros kernels en hardware Intel, AMD, o incluso en diferentes tipos de aceleradores? La **portabilidad multi-target** es uno de los desafíos más importantes en la programación heterogénea moderna.

En esta sesión exploraremos **SYCL**, un estándar abierto para programación paralela heterogénea, y discutiremos cómo podríamos extender nuestro agente generador de kernels para producir código portable que funcione en múltiples plataformas.

```{admonition} Nota Importante
:class: warning
SYCL requiere hardware específico (Intel GPUs, CPUs con soporte oneAPI, etc.) que no está disponible en este entorno. Esta sesión es **conceptual**: analizaremos código SYCL y discutiremos estrategias de portabilidad, pero no ejecutaremos código SYCL real.
```

---

## Bloque 1: Conceptos Fundamentales de SYCL

### ¿Qué es SYCL?

**SYCL** (pronunciado "sickle") es un estándar abierto desarrollado por el Khronos Group para programación paralela heterogénea. Es C++ moderno puro (sin extensiones del lenguaje) que permite escribir código que se ejecuta en CPUs, GPUs, FPGAs y otros aceleradores.

**Características principales:**
- Estándar abierto (como OpenCL, pero de más alto nivel)
- C++ moderno sin extensiones propietarias
- Single-source: host y device code en el mismo archivo
- Múltiples backends: Intel oneAPI, hipSYCL, ComputeCpp, etc.
- Portabilidad: el mismo código puede ejecutarse en hardware de diferentes vendors

```{admonition} Historia Rápida
:class: info
SYCL surgió como una abstracción de más alto nivel sobre OpenCL. Mientras que OpenCL requiere código separado para host y device, SYCL permite escribir todo en C++ moderno. Intel ha adoptado SYCL como parte central de su oneAPI initiative.
```

### Modelo de Ejecución SYCL

SYCL organiza la ejecución paralela de manera similar a CUDA, pero con terminología diferente:

#### Work-items (equivalente a threads en CUDA)
Un **work-item** es la unidad más básica de ejecución. Cada work-item ejecuta el kernel con un ID único.

```cpp
// Un work-item procesa un elemento
h.parallel_for(range<1>(N), [=](id<1> i) {
    c[i] = a[i] + b[i];  // Este código se ejecuta por cada work-item
});
```

#### Work-groups (equivalente a blocks en CUDA)
Los work-items se organizan en **work-groups**. Los work-items dentro de un work-group pueden:
- Compartir memoria local (local memory)
- Sincronizarse usando barreras
- Cooperar en computaciones

```cpp
// Work-groups explícitos con local memory
h.parallel_for(nd_range<1>(global_size, local_size), [=](nd_item<1> item) {
    int gid = item.get_global_id(0);      // ID global (como threadIdx + blockIdx*blockDim)
    int lid = item.get_local_id(0);        // ID local dentro del work-group
    int group_id = item.get_group_id(0);   // ID del work-group
});
```

#### ND-range (equivalente a grid en CUDA)
Un **ND-range** define el espacio total de ejecución, especificando:
- **Global range**: número total de work-items
- **Local range**: tamaño de cada work-group

```cpp
// Definir un ND-range 2D
nd_range<2> execution_range(
    range<2>(1024, 1024),  // Global: 1024x1024 work-items totales
    range<2>(16, 16)        // Local: work-groups de 16x16
);
```

#### Sub-groups (equivalente a warps en CUDA)
Los **sub-groups** son conjuntos de work-items que se ejecutan juntos en el hardware (típicamente 32 work-items en GPUs, similar a warps CUDA). Permiten operaciones de bajo nivel como shuffles.

```cpp
item.get_sub_group().shuffle(value, source_id);  // Como __shfl_sync en CUDA
```

### Modelo de Memoria SYCL

SYCL ofrece dos paradigmas principales para gestionar memoria:

#### 1. Buffers y Accessors (Enfoque Original)

El modelo tradicional de SYCL usa **buffers** para gestionar datos y **accessors** para acceder a ellos. El runtime gestiona automáticamente las transferencias.

```{code-cell} ipython3
:tags: [skip-execution]

# Pseudocódigo conceptual (no ejecutable)
"""
// Crear buffers
buffer<float> a_buf(a_data, range<1>(N));
buffer<float> b_buf(b_data, range<1>(N));
buffer<float> c_buf(c_data, range<1>(N));

// Submit trabajo
q.submit([&](handler& h) {
    // Crear accessors (especifica read/write)
    auto a = a_buf.get_access<access::mode::read>(h);
    auto b = b_buf.get_access<access::mode::read>(h);
    auto c = c_buf.get_access<access::mode::write>(h);

    h.parallel_for(range<1>(N), [=](id<1> i) {
        c[i] = a[i] + b[i];
    });
});
// Los datos se copian automáticamente de vuelta cuando c_buf sale de scope
"""
```

**Ventajas**: El runtime gestiona transferencias automáticamente, optimizando movimientos de datos.

**Desventajas**: Sintaxis más verbosa, puede ser confuso para principiantes.

#### 2. USM - Unified Shared Memory (Enfoque Moderno)

USM es más similar al modelo de CUDA: asignas memoria y usas punteros directamente.

```{code-cell} ipython3
:tags: [skip-execution]

# Pseudocódigo conceptual (no ejecutable)
"""
// Asignar memoria USM (similar a cudaMalloc)
float* a = malloc_device<float>(N, q);
float* b = malloc_device<float>(N, q);
float* c = malloc_device<float>(N, q);

// Copiar datos (similar a cudaMemcpy)
q.memcpy(a, host_a, N * sizeof(float));
q.memcpy(b, host_b, N * sizeof(float));

// Launch kernel
q.parallel_for(range<1>(N), [=](id<1> i) {
    c[i] = a[i] + b[i];
}).wait();

// Copiar resultados de vuelta
q.memcpy(host_c, c, N * sizeof(float)).wait();

// Liberar memoria
free(a, q);
free(b, q);
free(c, q);
"""
```

**Ventajas**: Más intuitivo para programadores CUDA, control explícito.

**Desventajas**: Debes gestionar transferencias manualmente.

#### Jerarquía de Memoria

| Tipo | Scope | Descripción | Equivalente CUDA |
|------|-------|-------------|------------------|
| **Private** | Work-item | Registros, variables locales | Registros |
| **Local** | Work-group | Compartida dentro del work-group | `__shared__` |
| **Global** | Device | Accesible por todos los work-items | Global memory |
| **Constant** | Device | Read-only, cached | `__constant__` |

```{code-cell} ipython3
:tags: [skip-execution]

# Ejemplo conceptual: uso de local memory
"""
h.parallel_for(nd_range<1>(N, 256), [=](nd_item<1> item) {
    // Local memory (compartida en el work-group)
    auto local_mem = item.get_local_mem<float, 256>();

    int lid = item.get_local_id(0);
    int gid = item.get_global_id(0);

    // Cargar datos a local memory
    local_mem[lid] = global_data[gid];

    // Sincronizar work-group
    item.barrier(access::fence_space::local_space);

    // Usar datos compartidos
    float result = local_mem[lid] + local_mem[(lid + 1) % 256];
});
"""
```

### Comparación: Triton vs CUDA vs SYCL

Veamos cómo se comparan estos tres enfoques para programación GPU:

| Concepto | CUDA | Triton | SYCL |
|----------|------|--------|------|
| **Thread/Work-item** | thread | (automático) | work-item |
| **Block/Work-group** | block | program | work-group |
| **Grid/ND-range** | grid | grid | ND-range |
| **Warp/Sub-group** | warp (32) | (automático) | sub-group |
| **Shared Memory** | `__shared__` | (automático) | local memory |
| **Sincronización** | `__syncthreads()` | (automático) | `barrier()` |
| **Launch** | `<<<grid,block>>>` | `kernel[(grid,)]` | `q.submit()` |
| **Lenguaje** | C++ + extensiones | Python DSL | C++ estándar |
| **Portabilidad** | Solo NVIDIA | Solo NVIDIA | Multi-vendor |
| **Gestión memoria** | Manual | Automática | Buffers o USM |

**Abstracciones:**
- **CUDA**: Control total, bajo nivel, específico de NVIDIA
- **Triton**: Abstracción alta, compilador gestiona optimizaciones, Python
- **SYCL**: C++ estándar, portable, flexibilidad en nivel de abstracción

### Ejemplo Completo: Vector Add en SYCL

Comparemos el mismo kernel de vector addition en los tres lenguajes:

#### CUDA

```cpp
__global__ void vector_add(float* a, float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

// Launch
vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
```

#### Triton

```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# Launch
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=1024)
```

#### SYCL (con USM)

```{code-cell} ipython3
:tags: [skip-execution]

# Pseudocódigo C++ en comentarios
"""
queue q;

// Asignar memoria
float* a = malloc_device<float>(N, q);
float* b = malloc_device<float>(N, q);
float* c = malloc_device<float>(N, q);

// Copiar datos
q.memcpy(a, host_a, N * sizeof(float));
q.memcpy(b, host_b, N * sizeof(float));

// Launch kernel
q.parallel_for(range<1>(N), [=](id<1> i) {
    c[i] = a[i] + b[i];
}).wait();

// Copiar resultados
q.memcpy(host_c, c, N * sizeof(float)).wait();

// Limpiar
free(a, q);
free(b, q);
free(c, q);
"""
```

#### SYCL (con Buffers)

```{code-cell} ipython3
:tags: [skip-execution]

# Pseudocódigo C++ en comentarios
"""
queue q;

{  // Scope de buffers
    // Crear buffers (envuelven datos del host)
    buffer<float> a_buf(host_a, range<1>(N));
    buffer<float> b_buf(host_b, range<1>(N));
    buffer<float> c_buf(host_c, range<1>(N));

    // Submit kernel
    q.submit([&](handler& h) {
        // Accessors declaran cómo se usan los datos
        auto a = a_buf.get_access<access::mode::read>(h);
        auto b = b_buf.get_access<access::mode::read>(h);
        auto c = c_buf.get_access<access::mode::write>(h);

        // Kernel
        h.parallel_for(range<1>(N), [=](id<1> i) {
            c[i] = a[i] + b[i];
        });
    });
}  // Los buffers se destruyen aquí, copiando datos automáticamente al host
"""
```

```{admonition} Observaciones
:class: tip
- **CUDA**: Más explícito, control total sobre threads/blocks
- **Triton**: Más abstracto, el compilador gestiona muchos detalles
- **SYCL**: Flexible - puedes elegir nivel de control (USM vs Buffers)
- **SYCL**: Single-source - todo en un archivo, sin separación host/device
```

### PyTorch en el Ecosistema Multi-target

PyTorch, nuestro framework base, también está evolucionando hacia la portabilidad:

#### torch.compile como Backend Portable

```{code-cell} ipython3
import torch

# torch.compile puede generar código para diferentes backends
@torch.compile
def my_function(x, y):
    return x @ y + x

# El compilador decide el backend óptimo
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
result = my_function(x, y)
```

#### Intel Extension for PyTorch (IPEX)

Intel proporciona extensiones que permiten ejecutar PyTorch en hardware Intel:

```{code-cell} ipython3
:tags: [skip-execution]

# Pseudocódigo conceptual
"""
import torch
import intel_extension_for_pytorch as ipex

# Mover modelo a dispositivo Intel
model = model.to('xpu')  # XPU = Intel GPU
model = ipex.optimize(model)

# El resto del código PyTorch funciona igual
output = model(input)
"""
```

#### Custom Ops Multi-target

Podemos registrar implementaciones diferentes del mismo op para diferentes backends:

```{code-cell} ipython3
:tags: [skip-execution]

# Pseudocódigo conceptual
"""
# Registrar implementación CUDA
@torch.library.impl("mylib::my_op", "CUDA")
def my_op_cuda(x):
    # Llamar a kernel CUDA/Triton
    return triton_kernel(x)

# Registrar implementación CPU
@torch.library.impl("mylib::my_op", "CPU")
def my_op_cpu(x):
    # Implementación CPU
    return cpu_kernel(x)

# Registrar implementación Intel
@torch.library.impl("mylib::my_op", "XPU")
def my_op_xpu(x):
    # Llamar a kernel SYCL
    return sycl_kernel(x)

# PyTorch despacha automáticamente según el device
result = torch.ops.mylib.my_op(tensor)  # Usa el backend correcto
"""
```

## Actividad de Trabajo — Bloque 1 (25 minutos)

**Ejercicio 1: Tabla comparativa CUDA / Triton / SYCL (15 min)**

Usando los tres ejemplos de `vector_add` que acabas de leer, completa la tabla
siguiente. Para cada concepto copia la línea de código exacta de cada lenguaje:

| Concepto | CUDA | Triton | SYCL (USM) |
|---|---|---|---|
| Obtener ID del hilo/work-item | `threadIdx.x + blockIdx.x*blockDim.x` | | |
| Acc. bounds check / máscara | `if (i < n)` | | |
| Leer elemento de entrada | `float val = A[i]` | | |
| Escribir resultado | `C[i] = val` | | |
| Lanzar kernel (host side) | `kernel<<<grid,block>>>(...)` | | |

Añade dos filas más de tu elección (ej: "declarar parámetro compile-time",
"gestionar memoria compartida").

**Ejercicio 2: Buffer vs USM — elección de modelo de memoria (10 min)**

Para cada escenario siguiente, indica si usarías **Buffers/Accessors** o
**USM** y justifica en una oración:

| Escenario | Modelo recomendado | Justificación |
|---|---|---|
| Migrar código CUDA existente que usa punteros raw | | |
| Código portable nuevo donde el compilador gestione dependencias | | |
| Pipeline donde múltiples kernels leen el mismo tensor en secuencia | | |

Después, responde: ¿Cuál modelo de memoria se parece más al modelo de
Triton (`tl.load`/`tl.store` con punteros)? ¿Por qué?

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

1. **Single-source vs split**: SYCL escribe host y device en el mismo archivo
   C++. En CUDA, el host code es `.cpp` y el device code es `.cu`. ¿Qué error
   específico de la práctica diaria elimina el single-source? Nombra un
   escenario donde el split de CUDA sea en cambio una ventaja de tooling.
2. **Máscaras vs bounds**: En Triton usamos `mask = offsets < n_elements` y
   en SYCL `if (i < n)`. Ambas hacen bounds checking, pero el mecanismo es
   distinto. ¿Cuál es más fácil de olvidar? ¿Cuál tiene más impacto en
   divergencia de ejecución en la GPU?
3. **`tl.constexpr` en SYCL**: En Triton, `BLOCK_SIZE: tl.constexpr` permite
   al compilador JIT optimizar el kernel para ese tamaño exacto. ¿Cómo se
   expresa el equivalente en SYCL y en qué momento del ciclo de vida del
   programa ocurre esa especialización?
```

---

## Bloque 2: Portabilidad Multi-target

### El Problema de Portabilidad

Hasta ahora, todo nuestro trabajo ha sido específico de NVIDIA:
- **CUDA**: Solo funciona en GPUs NVIDIA
- **Triton**: Actualmente solo backend CUDA (hay trabajo experimental en otros)
- **Dependencia de vendor**: Si queremos cambiar de hardware, hay que reescribir todo

**Escenarios reales donde esto importa**:

1. **Cloud heterogéneo**: AWS tiene instancias NVIDIA, Intel, AMD
2. **Edge computing**: Dispositivos con diferentes aceleradores
3. **Investigación**: Comparar rendimiento entre vendors
4. **Futuro-proofing**: No depender de un solo vendor

```{admonition} Caso Real: MLPerf
:class: note
Los benchmarks de MLPerf se ejecutan en hardware de múltiples vendors. Los equipos que pueden portar rápidamente tienen ventaja. Muchos usan frameworks portables o generación automática de código.
```

### Enfoques de Portabilidad

#### 1. Reescribir para Cada Target (Costoso)

**Enfoque manual**: Mantener implementaciones separadas.

```
kernels/
  ├── cuda/
  │   ├── matmul.cu
  │   ├── softmax.cu
  ├── sycl/
  │   ├── matmul.cpp
  │   ├── softmax.cpp
  ├── hip/  (AMD)
  │   ├── matmul.cpp
  │   ├── softmax.cpp
```

**Ventajas**:
- Rendimiento óptimo por plataforma
- Control total sobre optimizaciones específicas

**Desventajas**:
- Mucho trabajo (3x-4x el código)
- Difícil mantener sincronizado
- Bugs pueden afectar solo una versión
- Expertise necesario en múltiples plataformas

#### 2. Usar Abstracción Común (SYCL, OpenCL)

**Enfoque portable**: Un código para todas las plataformas.

```cpp
// Un solo kernel SYCL que funciona en NVIDIA, Intel, AMD
q.parallel_for(range<1>(N), [=](id<1> i) {
    c[i] = a[i] + b[i];
});
```

**Ventajas**:
- Un solo código base
- Mantenimiento más fácil
- Portabilidad garantizada (si el backend existe)

**Desventajas**:
- Puede perder optimizaciones específicas de vendor
- Performance puede no ser óptima en todas las plataformas
- Debugging más complejo (múltiples backends)
- Limitado al conjunto común de features

#### 3. Generación Automática Multi-target (Nuestro Agente!)

**Enfoque generativo**: Un spec → múltiples implementaciones.

```
Specification (JSON/Python)
    ↓
Agent Generator
    ├→ CUDA Kernel (para NVIDIA)
    ├→ Triton Kernel (para NVIDIA, más abstracto)
    ├→ SYCL Kernel (para Intel/AMD/NVIDIA)
    └→ HIP Kernel (para AMD)
```

**Ventajas**:
- Mantener solo la especificación
- Optimizaciones específicas por target
- Agregar nuevos targets sin modificar código existente
- El agente puede aprender mejores prácticas por plataforma

**Desventajas**:
- Complejidad en el generador
- Necesita testing exhaustivo en cada plataforma
- Especificación debe ser suficientemente expresiva

```{admonition} Nuestra Oportunidad
:class: tip
¡Este es exactamente el tipo de problema que un agente con LLM puede resolver bien! El modelo puede aprender patterns de traducción entre APIs y generar código idiomático para cada target.
```

### Adaptando el Agente para Multi-target

Recordemos la arquitectura de nuestro agente generador de kernels de sesiones anteriores:

```
User Request → Agent Planner → Kernel Spec → Code Generator → Triton Kernel
```

Para multi-target, extendemos esto:

```
User Request → Agent Planner → Kernel Spec → Target Selector
                                               ├→ CUDA Generator → CUDA Kernel
                                               ├→ Triton Generator → Triton Kernel
                                               └→ SYCL Generator → SYCL Kernel
```

#### Componentes Clave

**1. Kernel Specification (agnóstica de target)**

```python
kernel_spec = {
    "operation": "elementwise_add",
    "inputs": [
        {"name": "a", "dtype": "float32", "shape": ["N"]},
        {"name": "b", "dtype": "float32", "shape": ["N"]}
    ],
    "outputs": [
        {"name": "c", "dtype": "float32", "shape": ["N"]}
    ],
    "compute": "c[i] = a[i] + b[i]",
    "parallelism": "embarrassingly_parallel",
    "memory_pattern": "streaming"
}
```

**2. Target-Specific Templates**

Cada generador tiene templates específicos:

```python
# Template Triton
TRITON_ELEMENTWISE = """
@triton.jit
def {name}_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = {compute}
    tl.store(c_ptr + offsets, c, mask=mask)
"""

# Template SYCL
SYCL_ELEMENTWISE = """
void {name}_kernel(queue& q, float* a, float* b, float* c, size_t N) {{
    q.parallel_for(range<1>(N), [=](id<1> i) {{
        c[i] = {compute};
    }}).wait();
}}
"""
```

**3. Pattern Translation Rules**

El agente necesita saber cómo traducir conceptos:

| Patrón | Triton | SYCL |
|--------|--------|------|
| Get thread ID | `pid * BLOCK + offset` | `i.get(0)` |
| Load data | `tl.load(ptr + off, mask)` | `a[i]` |
| Store data | `tl.store(ptr + off, val, mask)` | `c[i] = val` |
| Block size | `BLOCK_SIZE: tl.constexpr` | `range<1>(local_size)` |
| Sync | (automático) | `item.barrier()` |

**4. Constraint Handling**

Algunas features no están en todas las plataformas:

```python
# Features específicas que no se portan directamente
CUDA_SPECIFIC = ["tensor cores", "warp intrinsics", "dynamic parallelism"]
TRITON_SPECIFIC = ["auto-tuning", "pointer arithmetic abstraction"]
SYCL_SPECIFIC = ["buffer accessors", "hierarchical parallelism"]

# El agente debe advertir o adaptar
if spec.requires("tensor_cores") and target == "SYCL":
    warning("Tensor cores no disponibles en SYCL - usando implementación alternativa")
```

### Template Comparison: SYCL vs Triton

Veamos un kernel más complejo lado a lado: **Reduce Sum**

#### Triton Implementation

```python
@triton.jit
def reduce_sum_kernel(
    input_ptr, output_ptr,
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    # Cada block procesa BLOCK_SIZE elementos
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Cargar datos
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Reduce dentro del block (automático por Triton)
    block_sum = tl.sum(data)

    # Guardar resultado
    tl.store(output_ptr + pid, block_sum)
```

#### SYCL Implementation (Equivalent)

```{code-cell} ipython3
:tags: [skip-execution]

# Pseudocódigo C++
"""
void reduce_sum_kernel(queue& q, float* input, float* output, size_t N) {
    constexpr size_t BLOCK_SIZE = 256;
    size_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q.parallel_for(
        nd_range<1>(num_blocks * BLOCK_SIZE, BLOCK_SIZE),
        [=](nd_item<1> item) {
            size_t gid = item.get_global_id(0);
            size_t lid = item.get_local_id(0);
            size_t group_id = item.get_group_id(0);

            // Local memory compartida
            auto local_mem = item.get_local_mem<float, BLOCK_SIZE>();

            // Cargar datos
            if (gid < N) {
                local_mem[lid] = input[gid];
            } else {
                local_mem[lid] = 0.0f;
            }

            // Sincronizar
            item.barrier(access::fence_space::local_space);

            // Tree reduction (manual en SYCL)
            for (size_t stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
                if (lid < stride) {
                    local_mem[lid] += local_mem[lid + stride];
                }
                item.barrier(access::fence_space::local_space);
            }

            // Thread 0 escribe resultado
            if (lid == 0) {
                output[group_id] = local_mem[0];
            }
        }
    ).wait();
}
"""
```

**Diferencias clave**:

1. **Triton**: Reduce automático con `tl.sum()` - el compilador genera el tree reduction
2. **SYCL**: Tree reduction manual - más control pero más verbose
3. **Triton**: Mask handling automático
4. **SYCL**: Bounds checking explícito
5. **SYCL**: Local memory y barreras explícitas
6. **Triton**: Trabajo en el compilador, SYCL en el programador

```{admonition} Implicaciones para el Agente
:class: important
El agente debe:
- Reconocer que `tl.sum()` en Triton → tree reduction manual en SYCL
- Insertar barreras apropiadas en SYCL
- Gestionar local memory explícitamente en SYCL
- Adaptar el patrón de carga/almacenamiento
```

### Limitaciones de Portabilidad

No todo se puede portar directamente:

#### Features que NO se Portan Fácilmente

**1. Tensor Cores (NVIDIA específico)**

```python
# Triton/CUDA puede usar tensor cores
tl.dot(a, b, allow_tf32=True)  # Usa tensor cores si están disponibles

# SYCL: No hay equivalente directo
# Solución: Implementación manual con GEMM, menos eficiente
```

**2. Warp-level Operations (CUDA)**

```cpp
// CUDA: warp shuffle
float val = __shfl_down_sync(0xffffffff, value, offset);

// SYCL: sub-groups pueden ser diferentes tamaños
// No todos los backends garantizan sub-groups de 32
auto sg = item.get_sub_group();
float val = sg.shuffle_down(value, offset);  // Puede ser diferente tamaño
```

**3. Dynamic Parallelism (CUDA)**

```cpp
// CUDA: kernel puede lanzar otro kernel
__global__ void parent_kernel() {
    child_kernel<<<grid, block>>>();  // Launch desde device
}

// SYCL: No soportado en la mayoría de backends
// Solución: Refactorizar para lanzar desde host
```

#### Estrategias para Manejar Limitaciones

1. **Fallback implementations**: Versión portable pero menos eficiente
2. **Conditional compilation**: Features opcionales si están disponibles
3. **Runtime detection**: Detectar capabilities y adaptar
4. **Degradación gradual**: Mejor esfuerzo en cada plataforma

```{code-cell} ipython3
# Pseudocódigo Python: detección de features
"""
def generate_kernel(spec, target):
    if target == "cuda" and has_tensor_cores():
        return generate_tensorcore_kernel(spec)
    elif target == "sycl" and has_subgroups():
        return generate_subgroup_kernel(spec)
    else:
        return generate_basic_kernel(spec)
"""
```

### Oportunidades: Un Agente Multi-target

**Visión**: Un agente que genera kernels optimizados para cualquier plataforma

```python
# API propuesta
agent = MultiTargetKernelAgent()

# Usuario describe la operación en lenguaje natural
request = """
Necesito un kernel de matrix multiplication que:
- Soporte matrices hasta 4096x4096
- Optimice para throughput sobre latencia
- Funcione en hardware NVIDIA e Intel
"""

# Agente genera para múltiples targets
kernels = agent.generate(request, targets=["cuda", "triton", "sycl"])

# Cada kernel está optimizado para su plataforma
kernels["cuda"]   # Usa tensor cores, shared memory tuning
kernels["triton"] # Usa auto-tuning, abstracciones altas
kernels["sycl"]   # Usa sub-groups, local memory optimizada

# Testing automático
agent.test_all(kernels, test_cases)

# Deployment
agent.deploy(kernels, runtime="adaptive")  # Selecciona kernel según hardware
```

**Capacidades del Agente**:

1. **Traducción entre APIs**: Entiende equivalencias entre CUDA/Triton/SYCL
2. **Optimización por plataforma**: Usa features específicas cuando están disponibles
3. **Fallback automático**: Genera versión portable si feature no está disponible
4. **Testing multi-target**: Verifica corrección en todas las plataformas
5. **Performance tuning**: Experimenta con parámetros por plataforma

```{admonition} Técnicas de LLM Útiles
:class: tip
- **Few-shot learning**: Mostrar ejemplos de traducciones CUDA↔SYCL
- **Chain-of-thought**: "Primero identifico el patrón, luego selecciono el template apropiado..."
- **Code retrieval**: Base de datos de kernels multi-target como referencia
- **Constraint satisfaction**: "Esta feature requiere X, que no está en SYCL, entonces uso Y"
```

### El Futuro de la Programación GPU Heterogénea

**Tendencias actuales**:

1. **Convergencia de APIs**: SYCL, OpenCL, Vulkan Compute, oneAPI
2. **Compiladores multi-target**: LLVM backends para múltiples GPUs
3. **Frameworks portable**: PyTorch, JAX con múltiples backends
4. **AI para generación de código**: GitHub Copilot, AlphaCode, y... ¡nuestros agentes!

**El rol de los agentes**:

- **Democratización**: No necesitas ser experto en cada plataforma
- **Aceleración**: Generación rápida de múltiples versiones
- **Optimización continua**: El agente aprende mejores prácticas con el tiempo
- **Adaptación**: Nuevas plataformas → actualizar el agente, no reescribir código

```{admonition} Reflexión Final
:class: note
En lugar de elegir entre portabilidad y rendimiento, los agentes generativos nos permiten tener ambos: generar código optimizado para cada plataforma desde una especificación común. Esta es una de las aplicaciones más prometedoras de LLMs en computación de alto rendimiento.
```

---

## Actividad de Trabajo — Bloque 2 (25 minutos)

**Ejercicio 1: Traducir `relu_kernel` a SYCL USM (15 min)**

Usa el `relu_kernel` de Triton que ves arriba y rellena la plantilla SYCL
USM siguiente. Está parcialmente completada; completa las 4 líneas marcadas
con `// TODO`:

```cpp
#include <sycl/sycl.hpp>
#include <algorithm>   // std::max

void relu_sycl(float* input, float* out, size_t n, sycl::queue& q) {
    // TODO 1: Definir rango de ejecución (equivalente al grid de CUDA)
    sycl::range<1> global_range{/* TODO 1 */};

    q.parallel_for(global_range, [=](sycl::id<1> idx) {
        size_t i = idx[0];

        // TODO 2: Bounds check (equivalente a mask = offsets < n_elements)
        if (/* TODO 2 */) {

            // TODO 3: Leer valor de input
            float val = /* TODO 3 */;

            // TODO 4: Aplicar ReLU y guardar (equivalente a tl.maximum + tl.store)
            out[i] = /* TODO 4 */;
        }
    }).wait();
}
```

Después responde: ¿en el kernel Triton, `mask` está fuera del loop; en el SYCL,
el bounds check está dentro del `parallel_for`. ¿Hay alguna diferencia de
rendimiento entre los dos enfoques?

**Ejercicio 2: Tabla de mapeo de operaciones y prompt LLM (10 min)**

Completa la tabla de mapeo extendida (mínimo 5 filas; las primeras 3 son de
ejemplo):

| Triton | SYCL (USM) | Notas |
|---|---|---|
| `tl.program_id(0)` | `item.get_global_id(0)` | Requiere `nd_item` |
| `tl.load(ptr, mask=m)` | `ptr[i]` + `if (i<n)` | Mask implícita en guard |
| `tl.maximum(a, b)` | `sycl::max(a, b)` | Header `<sycl/sycl.hpp>` |
| `tl.sum(x, axis=0)` | | |
| `tl.arange(0, BLOCK)` | | |
| ... | | |

Con base en la tabla, escribe en 4-6 líneas el system prompt que usarías
para pedirle a un LLM que traduzca un kernel Triton arbitrario a SYCL.

---

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: En Triton, `BLOCK_SIZE: tl.constexpr = 1024` hace que el
compilador JIT genere código específico para 1024 hilos. En SYCL, el
tamaño de work-group se pasa en el `nd_range` en tiempo de ejecución.
¿Cómo afecta esa diferencia a la estrategia de autotuning del agente?
¿Puede el agente usar el mismo `@triton.autotune` para SYCL?
<details><summary>Respuesta</summary>
No directamente: <code>@triton.autotune</code> explota JIT compile-time
especialización. Para SYCL el agente necesitaría generar múltiples versiones
con distintos <code>nd_range</code> y medirlas en runtime — un loop de
benchmark externo, no un mecanismo del compilador. El agente debe tratar
el tamaño de work-group como un hipérparámetro a explorar con
<code>evaluate_kernel()</code> (de S5).
</details>

**Pregunta 2**: Tu pipeline en S7 clasifica errores como `COMPILE`,
`RUNTIME`, `CORRECTNESS`. Si el agente genera un kernel SYCL que compila
con `icpx` pero produce resultados incorrectos, ¿el clasificador de S8
(`classify_error`) funcionaría sin modificaciones? ¿Qué deberías cambiar?
<details><summary>Respuesta</summary>
El clasificador de S8 detecta patrones en código Python/Triton (ej: regex
sobre <code>tl.load</code> sin máscara). Para SYCL C++ los patrones
cambian: buscar <code>ptr[i]</code> sin <code>if (i&lt;n)</code>,
<code>atomic_ref</code> faltante en reducciones, o indexación column-major
(<code>row + col*height</code>). La categoría <code>CORRECTNESS</code> sigue
funcionando; las subcategorías y sus regexes necesitan versiones SYCL.
</details>
```

### Para Pensar

> *El agente de S7 aprovecha el mensaje de error del compilador Python para
> guiar el reintento. Si el target es SYCL y el compilador es `icpx` (Intel
> C++), ¿cómo cambiaría la estrategia de `error_guided`? ¿Los mensajes de
> error de `icpx` son igual de informativos que los de Python?*

---

## Resumen

En esta sesión exploramos:

### SYCL: Programación Heterogénea Portable
- Estándar abierto de Khronos para C++
- Modelo de ejecución: work-items, work-groups, ND-ranges, sub-groups
- Modelo de memoria: buffers/accessors vs USM
- Comparación con CUDA y Triton

### Portabilidad Multi-target
- El problema: código GPU específico de vendor
- Tres enfoques: reescribir, abstracción común, generación automática
- Cómo adaptar el agente para generar múltiples targets
- Limitaciones: features que no portan

### Oportunidades
- Agentes LLM son naturales para traducción entre APIs
- Generación multi-target desde especificación común
- Optimización por plataforma manteniendo portabilidad
- Futuro: programación heterogénea democratizada

**Conexión con el Proyecto**:
- Tu agente actualmente genera Triton (solo NVIDIA)
- Podrías extenderlo para generar SYCL (Intel/AMD/portable)
- La especificación de kernel puede ser agnóstica de plataforma
- Más targets = mayor impacto y flexibilidad

---

## Tarea para Casa

### Ejercicio 1: `TritonToSYCLTranslator` para 3 patrones (★★☆)

Implementa la clase `TritonToSYCLTranslator` con soporte para tres patrones
(no pseudocódigo — código Python real que manipule strings):

```python
class TritonToSYCLTranslator:
    MAPPING = {
        'tl.program_id(0)':   'idx[0]',
        'tl.maximum':          'sycl::max',
        # TODO: añade las entradas de tu tabla de la clase
    }

    def translate_elementwise(self, triton_src: str) -> str:
        """Reemplaza tokens Triton por equivalentes SYCL USM."""
        # TODO: sustitución simple de tokens con MAPPING
        pass

    def translate_reduction(self, triton_src: str) -> str:
        """
        Detecta `tl.sum` / `tl.max` y genera reducción SYCL
        con `sycl::atomic_ref`.
        """
        # TODO
        pass

    def translate(self, triton_src: str) -> str:
        """Detecta patrón y delega al método correcto."""
        if 'tl.sum' in triton_src or 'tl.max' in triton_src:
            return self.translate_reduction(triton_src)
        return self.translate_elementwise(triton_src)
```

Prueba `translate()` sobre los kernels `relu_kernel` y `vector_add_kernel`
de la sesión. El output no tiene que ser código perfecto, pero debe
contenerse los tokens SYCL correctos (verifícalo con `assert`).

### Ejercicio 2: Prompt LLM para traducción Triton → SYCL (★★★)

Diseña y evalúa un prompt completo para traducción automática:

1. **System prompt** con reglas SYCL USM (≥ 5 reglas; equivalentes a las
   reglas Triton del system prompt de S6).
2. **Few-shot**: incluye la traducción `vector_add` (Triton → SYCL) como
   ejemplo completo.
3. Evalúa el prompt sobre `relu_kernel` y `scale_kernel` usando tu backend LLM.
4. Clasifica cada fallo con la tabla de categorías de S8 adaptada a SYCL C++.
5. Realiza una iteración de mejora y reporta el cambio en pass rate.

---

## Referencias

### SYCL y oneAPI
- [SYCL 2020 Specification](https://www.khronos.org/registry/SYCL/) - Especificación oficial
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/documentation.html)
- [DPC++ (Data Parallel C++)](https://github.com/intel/llvm/tree/sycl) - Implementación open source de Intel
- [SYCL Academy](https://github.com/codeplaysoftware/syclacademy) - Tutoriales interactivos

### Portabilidad GPU
- [hipSYCL](https://github.com/illuhad/hipSYCL) - SYCL sobre CUDA y ROCm
- [Kokkos](https://github.com/kokkos/kokkos) - Framework portable C++ para HPC
- [RAJA](https://github.com/LLNL/RAJA) - Abstracción portable de LLNL
- [Alpaka](https://github.com/alpaka-group/alpaka) - Header-only C++ para aceleradores

### Comparaciones
- "A Comparative Study of SYCL, OpenCL, and CUDA" - Zhai et al. (2022)
- "Performance Portability on Heterogeneous Architectures" - Deakin et al. (2021)
- [GPU Performance Portability](https://performanceportability.org/)

### Herramientas
- [SYCLomatic](https://github.com/oneapi-src/SYCLomatic) - Herramienta de migración CUDA→SYCL
- [SYCL-Bench](https://github.com/unisa-hpc/sycl-bench) - Suite de benchmarks
- [ComputeCpp](https://developer.codeplay.com/products/computecpp/) - Compilador SYCL de Codeplay

### Artículos Relevantes
- Reyes, R., & Lomüller, V. (2020). "SYCL: Single-source C++ on OpenCL"
- Thoman, P., et al. (2023). "Towards Performance Portability for Heterogeneous Systems"
- Pennycook, J., et al. (2021). "Achieving Performance Portability for GPUs"

---

**Próxima sesión**: Integraremos todo lo aprendido en una demostración end-to-end del agente generador de kernels, incluyendo evaluación, debugging, y deployment en producción.
