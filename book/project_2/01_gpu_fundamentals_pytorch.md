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

# GPU Fundamentals y PyTorch en GPU

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton datasets evaluate
```

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/01_gpu_fundamentals_pytorch.ipynb)
```

> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 1
> **Tiempo de lectura:** ~40 minutos

---

## Introducción

Si eres nuevo en PyTorch, es posible que ya hayas entrenado un modelo sencillo con algo como `model = nn.Linear(10, 1); loss.backward(); optimizer.step()`. Eso funciona en CPU, pero en cuanto escalas el modelo o el dataset, la CPU se convierte en el cuello de botella. El mismo código en GPU puede correr **10x a 100x más rápido** sin cambiar casi nada — solo `.to('cuda')`. ¿Por qué? Eso es lo que aprenderás hoy.

¿Por qué entrenar un modelo de deep learning es órdenes de magnitud más rápido en GPU? ¿Cómo PyTorch gestiona tensores en estos dispositivos tan distintos a una CPU tradicional? La respuesta está en arquitecturas fundamentalmente diferentes diseñadas para resolver problemas distintos, y en abstracciones cuidadosamente construidas para aprovechar ese poder.

Antes de optimizar código para GPU, necesitas entender **por qué** las GPUs son tan poderosas para ciertos tipos de cómputo, **cómo** su arquitectura dicta las estrategias de optimización, y **cómo** PyTorch abstrae estos conceptos para hacerlos accesibles desde Python. Sin este conocimiento, estarías optimizando a ciegas.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta sesión podrás:

**Bloque 1 (Arquitectura GPU):**
- Explicar las diferencias fundamentales entre CPUs y GPUs (latencia vs throughput)
- Describir el modelo de ejecución SIMT y la jerarquía Grid→Blocks→Threads→Warps
- Comprender la jerarquía de memoria GPU y calcular latencias relativas
- Identificar cuándo usar GPU vs CPU basándote en arithmetic intensity y paralelismo

**Bloque 2 (PyTorch en GPU):**
- Gestionar tensores en GPU usando `.to(device)`, `.cuda()`, y `device='cuda'`
- Minimizar transferencias CPU↔GPU y usar pinned memory
- Monitorear uso de memoria con `torch.cuda.memory_allocated()` y family
- Usar `torch.profiler` para identificar bottlenecks de CUDA time y memoria
```

---

## Bloque 1: Arquitectura GPU y Modelo de Ejecución

### La Filosofía Fundamental: CPU vs GPU

#### CPUs: Pocos Trabajadores Muy Inteligentes

Los procesadores tradicionales están optimizados para **ejecutar tareas complejas y secuenciales rápidamente**. Imagina una CPU como un chef experto en una cocina pequeña:

- **Pocos cores** (8-64 típicamente)
- **Cores muy potentes** con predicción de branches, ejecución fuera de orden
- **Caches grandes** para minimizar latencia de memoria
- **Clock alto** (3-5 GHz)
- **Latencia baja**: Optimizado para responder rápidamente a una tarea individual

Las CPUs están diseñadas para ejecutar **una tarea muy rápido**.

#### GPUs: Miles de Trabajadores Simples

Las GPUs son máquinas de **paralelismo masivo**. Imagina una GPU como una fábrica con miles de empleados haciendo la misma tarea repetitiva:

- **Miles de cores** (10,000+ en GPUs de datacenter)
- **Cores simples** sin predicción de branches sofisticada
- **Caches pequeños** pero **ancho de banda masivo**
- **Clock moderado** (1-2 GHz)
- **Latencia alta, throughput gigantesco**: Lento para una tarea individual, pero procesa muchísimas en paralelo

Las GPUs están diseñadas para ejecutar **muchas tareas en paralelo**.

:::{figure} diagrams/cpu_vs_gpu_architecture.png
:name: fig-cpu-gpu-arch
:alt: Comparación de arquitecturas de CPU (pocos cores complejos) vs GPU (miles de cores simples)
:align: center
:width: 90%

**Figura 1:** Arquitectura CPU vs GPU - Las CPUs optimizan latencia con cores complejos; las GPUs optimizan throughput con miles de cores simples.
:::

```{admonition} Modelo Mental
:class: hint
Piensa en los threads de GPU como trabajadores en una línea de ensamblaje:
- **CPU**: Un chef experto que prepara comidas gourmet una a la vez (pocos cores, alta complejidad)
- **GPU**: 10,000 empleados de comida rápida preparando hamburguesas simultáneamente (muchos cores, tarea simple)

Si necesitas 1 comida gourmet → CPU
Si necesitas 10,000 hamburguesas → GPU
```

```{admonition} Concepto clave
:class: note
La GPU sacrifica eficiencia en tareas individuales para maximizar el throughput total. Para deep learning, donde tenemos millones de operaciones independientes, esto es exactamente lo que necesitamos.
```

---

### El Modelo SIMT (Single Instruction, Multiple Threads)

Este es el secreto de las GPUs: **SIMT** (Single Instruction, Multiple Threads). Es similar a SIMD pero con más flexibilidad.

```
Código GPU:
int result = threadData[threadId] * 2;

Con 4 threads:
Thread 0: threadData[0] * 2
Thread 1: threadData[1] * 2
Thread 2: threadData[2] * 2
Thread 3: threadData[3] * 2
```

Todos los threads ejecutan **exactamente el mismo código**, pero con **datos diferentes**. Esto es masivamente más eficiente que tener que especificar el mismo código 10,000 veces.

#### ¿Por qué es esto tan poderoso?

1. **Simplicidad lógica**: El hardware es más simple si todos hacen lo mismo
2. **Escalabilidad**: Podemos agregar más cores sin cambiar el código
3. **Ancho de banda**: Con miles de threads, podemos mantener la memoria constantemente ocupada

---

### Jerarquía de Ejecución: Grids, Blocks, Threads, Warps

Las GPUs organizan el paralelismo en una jerarquía clara:

```
Grid (todo el kernel)
  └── Blocks (grupos de threads)
        └── Threads (unidades de ejecución)
              └── Warps (32 threads ejecutando la misma instrucción)
```

#### Grid

El **Grid** representa todo el trabajo a realizar. Cuando lanzas un kernel, especificas el tamaño del grid.

```
GRID (Todo tu trabajo)
├─ Block 0 (256 threads)
│  ├─ Thread 0
│  ├─ Thread 1
│  └─ ... (256 threads total)
├─ Block 1 (256 threads)
│  ├─ Thread 0
│  └─ ...
└─ Block N
   └─ ...
```

#### Blocks

Los **Blocks** son grupos de threads que:
- Pueden comunicarse entre sí via shared memory
- Se sincronizan con `__syncthreads()`
- Se ejecutan en un solo Streaming Multiprocessor (SM)
- Típicamente 128-512 threads por bloque

#### Threads

Cada **Thread** es una unidad de ejecución que:
- Tiene su propio program counter
- Tiene acceso a registros privados
- Ejecuta el mismo código pero con diferentes datos
- Se identifica con `threadIdx` y `blockIdx`

#### Warps: La Unidad Crítica

Un **Warp** es un grupo de **32 threads** que:
- Ejecutan **exactamente la misma instrucción** al mismo tiempo
- Son la unidad fundamental de scheduling en la GPU

```{code-cell} ipython3
:tags: [skip-execution]

# Si tienes un if/else en tu código:
if condition:
    # Todos los threads del warp donde condition=True ejecutan esto
    do_something()
else:
    # Threads donde condition=False esperan, luego ejecutan esto
    do_other()
# Esto se llama "warp divergence" y es costoso!
```

```{code-cell} ipython3
# Visualización de Warps y Threads
import numpy as np
import plotly.graph_objects as go

fig = go.Figure()

# Grid de threads (simplificado: 4 warps de 8 threads para visualizar)
n_warps = 4
threads_per_warp = 8

for w in range(n_warps):
    for t in range(threads_per_warp):
        x = t * 0.1 + 0.05
        y = 0.9 - w * 0.2
        color = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D'][w]
        fig.add_shape(type="rect", x0=x, y0=y-0.05, x1=x+0.08, y1=y+0.05,
                      fillcolor=color, opacity=0.7, line=dict(color='black'))
        fig.add_annotation(x=x+0.04, y=y, text=str(w*threads_per_warp + t),
                           showarrow=False, font=dict(size=8))

    fig.add_annotation(x=0.95, y=0.9-w*0.2, text=f'Warp {w}',
                       showarrow=False, font=dict(size=12, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D'][w]))

fig.update_layout(
    title="Organización de Threads en Warps (32 threads reales = 1 warp)",
    xaxis=dict(visible=False, range=[0, 1.1]),
    yaxis=dict(visible=False, range=[0, 1]),
    height=350
)
fig.show()
```

### Divergencia de Warp: El Enemigo del Performance

```cuda
// Divergencia de warp (MAL)
if (threadIdx.x % 2 == 0) {
    // 16 threads ejecutan esto
    resultado = sumaCompleja();
} else {
    // 16 threads ESPERAN
    resultado = multiplicacionSimple();
}
// Los 16 threads que terminaron rápido ESPERAN a los otros
```

```{admonition} Antipatrón: Divergencia de Warp
:class: warning
**Problema**: Branches dentro de un warp causan serialización - los threads que toman un camino esperan a los que toman el otro.

**Causa común**: `if (threadIdx.x % 2 == 0)` hace que mitad del warp ejecute código A y mitad ejecute código B.

**Impacto**: Rendimiento se divide por 2 (o peor si hay múltiples branches).

**Solución**: Reorganiza datos para que threads consecutivos tomen el mismo camino, o usa operaciones predicate-based como `tl.where()`.
```

**Ejemplo numérico de warps:** Bloque de 256 threads = 8 warps (256/32). Con 100 bloques y GPU de 80 SMs ejecutando 4 warps simultáneos: 800 warps totales ejecutan en 3 waves (ceil(800/320)).

### Distribución de Bloques en SMs

Si tu GPU tiene 80 SMs y lanzas 1000 bloques:

```
Ejecución dinámica de bloques:
┌────────────┬────────────┬────────────┐
│  SM 0      │  SM 1      │  SM 2      │
├────────────┼────────────┼────────────┤
│ Block 0    │ Block 1    │ Block 2    │
├────────────┼────────────┼────────────┤
│ Block 80   │ Block 81   │ Block 82   │
├────────────┼────────────┼────────────┤
│ ...        │ ...        │ ...        │
└────────────┴────────────┴────────────┘

Los bloques se lanzan dinámicamente a los SMs disponibles
```

:::{figure} diagrams/thread_hierarchy.png
:name: fig-thread-hierarchy
:alt: Jerarquía de ejecución: Grid contiene Blocks, que contienen Threads, agrupados en Warps de 32 threads
:align: center
:width: 90%

**Figura 2:** Jerarquía de ejecución - Grid subdivido en Blocks, cada Block contiene Threads, y cada 32 Threads consecutivos forman un Warp (unidad de scheduling).
:::

---

### Simulando el Grid en Python Puro

Antes de ver código compilado para GPU (CUDA o Triton), construyamos el modelo mental de cómo los identificadores mágicos (`blockIdx`, `threadIdx`) mapean tu posición 3D a un arreglo lineal (1D) en memoria. En Python, esto es simple aritmética modular que todo thread de GPU ejecuta en hardware.

```{code-cell} ipython3
# Simulación en Python puro de la indexación GPU
def simulate_gpu_grid_mapping(grid_dim_x, block_dim_x):
    print(f"\nLanzando Kernel simulado: Grid de {grid_dim_x} bloques, {block_dim_x} threads por bloque")
    print(f"Total de threads que se ejecutarán en paralelo: {grid_dim_x * block_dim_x}")
    print("="*70)

    # Simulamos lo que cada thread calcula internamente
    for blockIdx_x in range(grid_dim_x):
        for threadIdx_x in range(block_dim_x):

            # FÓRMULA MÁGICA DE CUDA/TRITON (De 2D a 1D)
            global_id = (blockIdx_x * block_dim_x) + threadIdx_x

            print(f"Bloque [{blockIdx_x}], Thread local [{threadIdx_x}] -> Accederá al índice Global de Memoria: [{global_id}]")

# Lanzar 3 bloques, cada uno con 4 threads
simulate_gpu_grid_mapping(grid_dim_x=3, block_dim_x=4)
```

---

## Jerarquía de Memoria GPU

Si bien las GPUs tienen miles de cores, su verdadero superpoder es el **ancho de banda de memoria**.

### Tipos de Memoria

| Tipo | Tamaño | Latencia | Scope | Uso |
|------|--------|----------|-------|-----|
| **Registers** | ~256KB/SM | 0 cycles | Thread | Variables locales |
| **Shared Memory** | ~48-100KB/SM | ~20 cycles | Block | Comunicación entre threads |
| **L1 Cache** | ~128KB/SM | ~30 cycles | SM | Cache automático |
| **L2 Cache** | ~1-40MB | ~200 cycles | GPU | Cache global |
| **Global Memory** | 16-80GB | ~400 cycles | GPU | Datos principales |

### Visualización de la Jerarquía

```
┌─────────────────────────────────────────────────────┐
│                   GPU MEMORY HIERARCHY              │
├─────────────────────────────────────────────────────┤
│ Registros (Registers)                               │
│ ├─ Privado de cada thread                           │
│ ├─ Más rápido (0 ciclos de latencia)                │
│ └─ Muy limitado (~256 bytes por thread)             │
├─────────────────────────────────────────────────────┤
│ Shared Memory (Memoria Compartida)                  │
│ ├─ Compartida entre threads del mismo bloque        │
│ ├─ ~48-96 KB por bloque                             │
│ ├─ Mucho más rápido que memoria global              │
│ └─ Requiere sincronización                          │
├─────────────────────────────────────────────────────┤
│ L1/L2 Cache                                         │
│ ├─ L1: ~128 KB por SM                               │
│ └─ L2: ~1-40 MB global                              │
├─────────────────────────────────────────────────────┤
│ Memoria Global (HBM)                                │
│ ├─ Toda la memoria del dispositivo (16-80 GB)       │
│ ├─ Mucha latencia (cientos de ciclos)               │
│ └─ Ancho de banda gigantesco (2-3 TB/s)             │
└─────────────────────────────────────────────────────┘
```

:::{figure} diagrams/gpu_memory_hierarchy.png
:name: fig-gpu-memory-hierarchy
:alt: Jerarquía de memoria GPU: Registros (privado), Shared Memory (per-block), L1/L2 Caches, Memoria Global
:align: center
:width: 90%

**Figura 3:** Jerarquía de memoria GPU - desde Registros ultrarrápidos pero limitados hasta Memoria Global masiva pero lenta.
:::

### Ancho de Banda vs Latencia

| Dispositivo | Latencia | Ancho de Banda |
|-------------|----------|----------------|
| CPU (DDR5) | Baja (~100ns) | ~50 GB/s |
| GPU A100 | Alta (~400 cycles) | ~2 TB/s |
| GPU H100 | Alta (~400 cycles) | ~3.35 TB/s |

```{admonition} Modelo Mental: Latencia vs Throughput
:class: hint
Piensa en la memoria como un sistema de entregas:

**CPU (Latencia baja)**: Como un servicio de mensajería express - llega rápido (100ns) pero solo puedes enviar pocas cajas a la vez (50 GB/s).

**GPU (Throughput alto)**: Como un tren de carga - tarda más en llegar (400 cycles), pero cuando llega trae TONELADAS de datos (2-3 TB/s).

¿Cómo compensa la GPU la alta latencia? **Teniendo miles de threads esperando** - mientras unos esperan su tren, otros procesan el que acaba de llegar. Es como una fábrica donde mientras algunos empleados esperan materiales, otros continúan trabajando.
```

```{admonition} Tip de Performance
:class: important
Para maximizar throughput de memoria GPU:
1. Lanza suficientes threads para ocultar latencia (occupancy > 50%)
2. Accede memoria en patrones coalesced (threads consecutivos → direcciones consecutivas)
3. Mantén datos en cache/shared memory cuando sea posible
```

---

### Cuándo Usar GPU vs CPU

#### Usa GPU si:

- ✓ Tienes mucho paralelismo (miles de operaciones independientes)
- ✓ La relación entre trabajo y movimiento de datos es alta (arithmetic intensity)
- ✓ Puedes tolerar divergencia de warp limitada
- ✓ El problema es regular (código similar para todos los threads)

#### Usa CPU si:

- ✓ El problema es muy secuencial o tiene muchas ramificaciones
- ✓ Necesitas latencia baja para operaciones individuales
- ✓ El código es muy complicado o variable por dato
- ✓ Movimiento de datos es el cuello de botella

---

### Por Qué Importa Optimizar

#### Impacto Económico

```
Costo de GPU A100 en cloud: ~$2-3/hora

Kernel no optimizado: 100ms por inferencia
Kernel optimizado:     10ms por inferencia

A escala (1M inferencias/día):
- No optimizado: 27.7 horas GPU → ~$70/día
- Optimizado:    2.77 horas GPU → ~$7/día

Ahorro anual: ~$23,000 por modelo
```

#### Impacto Energético

Las GPUs de datacenter consumen 300-700W. Kernels eficientes:
- Reducen tiempo de ejecución
- Reducen energía total consumida
- Reducen huella de carbono

---

````{admonition} Actividad de Trabajo - Bloque 1 (25 minutos)
:class: note

**Ejercicio 1: Cálculo de Warps y Waves (10 min)**

Calcula a mano los siguientes valores para cada configuración, luego verifica con
`analyze_kernel_launch` (definida en la sección "Ejercicio Práctico" más abajo):

1. Kernel con 512 threads/bloque, 150 bloques, GPU de 80 SMs, 4 warps/SM
2. Kernel con 128 threads/bloque, 1000 bloques, GPU de 108 SMs, 64 warps/SM

Calcula: warps por bloque, total warps, warps concurrentes, waves necesarias.

**Ejercicio 2: Identificar Candidatos GPU (10 min)**

Para cada problema, decide GPU o CPU y justifica basándote en paralelismo y arithmetic intensity:

1. Filtrar outliers de un dataset de 10 millones de registros
2. Ejecutar un compilador con análisis complejo
3. Multiplicación de matrices 10,000 x 10,000
4. Parsing de JSON con estructura variable
5. Aplicar filtro de convolución a imágenes 4K

**Ejercicio 3: Análisis de Divergencia (5 min)**

De los dos kernels siguientes, identifica cuál tiene divergencia de warp, explica por qué
y propón una versión corregida:

```cuda
// kernel_A
__global__ void kernel_A(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0) {
        data[idx] = sqrt(data[idx]);
    } else {
        data[idx] = data[idx] * 2.0f;
    }
}

// kernel_B
__global__ void kernel_B(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}
```
````

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

Preguntas para discutir en clase antes de continuar:

1. **¿Cuándo NO conviene usar GPU?** Revisen juntos las respuestas del Ejercicio 2 e identifiquen el patrón: ¿qué hace que un problema sea "malo" para GPU?
2. **Divergencia en la práctica**: ¿Cómo puede aparecer warp divergence en un transformer que procesa secuencias de distinto largo en el mismo batch? ¿Qué hace PyTorch para mitigarlo con el padding?
3. **Occupancy y waves**: Si una GPU tiene 108 SMs y lanzas 109 bloques en vez de 108, ¿qué pasa con la última wave? ¿Cómo afecta a la utilización de la GPU?
```

---

## Bloque 2: PyTorch en GPU

Ahora que entiendes la arquitectura GPU, veamos cómo PyTorch abstrae estos conceptos para hacerlos accesibles desde Python.

### Tensores en GPU: Primeros Pasos

PyTorch hace la gestión de GPU sorprendentemente simple:

```{code-cell} ipython3
:tags: [skip-execution]

import torch

# Crear tensor en CPU
x_cpu = torch.randn(1000, 1000)

# Mover a GPU - Tres formas equivalentes
x_gpu = x_cpu.to('cuda')        # Método recomendado
x_gpu = x_cpu.cuda()            # Forma corta
x_gpu = x_cpu.to(torch.device('cuda'))  # Forma explícita

# Crear directamente en GPU (más eficiente)
x_gpu = torch.randn(1000, 1000, device='cuda')

# Verificar dispositivo
print(x_gpu.device)  # cuda:0
print(x_gpu.is_cuda)  # True
```

````{admonition} Tip de Performance
:class: important
**Siempre crea tensores directamente en GPU cuando sea posible.**

```python
# MALO: Crea en CPU, luego mueve a GPU
x = torch.randn(1000, 1000).cuda()  # 2 allocaciones

# BUENO: Crea directamente en GPU
x = torch.randn(1000, 1000, device='cuda')  # 1 allocación
```

Esto evita allocación temporal en CPU y transferencia PCIe innecesaria.
````

---

### El Costo del Movimiento CPU↔GPU

Aquí es donde la arquitectura GPU se vuelve crítica para tu código PyTorch:

```{admonition} Modelo Mental: CPU↔GPU Transfer
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

```{code-cell} ipython3
:tags: [skip-execution]

import time

# Ejemplo del impacto de transferencias
x_cpu = torch.randn(10000, 10000)  # 400 MB
x_gpu = torch.randn(10000, 10000, device='cuda')

# Operación en GPU (rápido)
start = time.time()
y_gpu = x_gpu @ x_gpu.t()
torch.cuda.synchronize()
print(f"GPU compute: {(time.time() - start)*1000:.2f} ms")

# Transferencia CPU→GPU (lento)
start = time.time()
x_transferred = x_cpu.cuda()
torch.cuda.synchronize()
print(f"CPU→GPU transfer: {(time.time() - start)*1000:.2f} ms")
```

```{admonition} Concepto clave
:class: note
El bandwidth PCIe es 60x menor que el HBM interno. Mantén los datos en GPU el mayor tiempo posible. Una regla simple: si tu kernel tarda <30ms pero mueves 1GB de datos, la transferencia es tu bottleneck real.
```

---

### Memory Layout y Contiguidad

PyTorch requiere que los tensores sean **contiguos** en memoria para operaciones eficientes:

```{code-cell} ipython3
:tags: [skip-execution]

# Tensor contiguo (óptimo para GPU)
x = torch.randn(100, 100)
print(f"Contiguo: {x.is_contiguous()}")  # True

# Tensor no contiguo (después de transpose)
y = x.t()
print(f"Contiguo: {y.is_contiguous()}")  # False

# Hacer contiguo (copia los datos)
y_contig = y.contiguous()
print(f"Contiguo: {y_contig.is_contiguous()}")  # True
```

**¿Por qué importa?** Los tensores no contiguos causan accesos de memoria que violan **coalescing** - threads consecutivos acceden direcciones no consecutivas, resultando en múltiples transacciones de memoria en vez de una.

```{admonition} Tip de Performance
:class: important
Antes de operaciones intensivas en GPU:
1. Verifica contiguidad: `tensor.is_contiguous()`
2. Si no es contiguo: `tensor = tensor.contiguous()`
3. Considera si el transpose es realmente necesario o puede evitarse
```

---

### Caching Allocator de PyTorch

PyTorch usa un **caching allocator** inteligente para evitar llamadas frecuentes a `cudaMalloc` (que son lentas):

```{code-cell} ipython3
:tags: [skip-execution]

# Primera allocación: llama a cudaMalloc
x = torch.randn(1000, 1000, device='cuda')
print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

# Liberar tensor
del x
print(f"Allocated after del: {torch.cuda.memory_allocated() / 1e6:.2f} MB")  # 0 MB
print(f"Reserved (cached): {torch.cuda.memory_reserved() / 1e6:.2f} MB")  # >0 MB

# Segunda allocación del mismo tamaño: reutiliza memoria cacheada
y = torch.randn(1000, 1000, device='cuda')  # No llama cudaMalloc, reutiliza
```

**¿Por qué cachear?** Porque `cudaMalloc` y `cudaFree` son operaciones costosas (microsegundos). El caching allocator:
- Reduce overhead de allocación/liberación
- Mejora performance en loops con allocaciones temporales
- Puede causar memoria "reservada" pero no "activa"

---

### Monitoreo de Memoria GPU

```{code-cell} ipython3
:tags: [skip-execution]

# Estado de memoria GPU
allocated = torch.cuda.memory_allocated()      # Memoria activamente usada
reserved = torch.cuda.memory_reserved()        # Memoria en cache del allocator
max_allocated = torch.cuda.max_memory_allocated()  # Pico histórico

print(f"Allocated:  {allocated / 1e9:.2f} GB")
print(f"Reserved:   {reserved / 1e9:.2f} GB")
print(f"Peak:       {max_allocated / 1e9:.2f} GB")

# Liberar memoria cacheada (si necesitas memoria para otro proceso)
torch.cuda.empty_cache()

# Resetear contadores de pico
torch.cuda.reset_peak_memory_stats()
```

```{admonition} Debugging de OOM (Out of Memory)
:class: warning
Si ves `RuntimeError: CUDA out of memory`:

1. **Verifica pico de memoria**: `torch.cuda.max_memory_allocated()` antes del crash
2. **Reduce batch size**: La causa #1 de OOM
3. **Libera variables grandes**: `del tensor` + `torch.cuda.empty_cache()`
4. **Usa gradient checkpointing**: Trade compute por memoria
5. **Considera usar AMP (Automatic Mixed Precision)**: FP16 usa 50% menos memoria
```

---

### torch.profiler: Identificar Bottlenecks

El profiler es tu herramienta principal para optimización:

```{code-cell} ipython3
:tags: [skip-execution]

from torch.profiler import profile, ProfilerActivity

model = torch.nn.Linear(1000, 1000).cuda()
x = torch.randn(100, 1000, device='cuda')

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(10):
        y = model(x)
        y.sum().backward()

# Ver top operaciones por CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Exportar para TensorBoard
prof.export_chrome_trace("trace.json")
```

**Output típico:**
```
Name                    CPU Time  CUDA Time  # Calls   Memory
aten::addmm              100us     800us       10      4MB
aten::mm                  80us     600us       10      4MB
aten::sum                 20us     100us       10      0B
```

**¿Qué buscar?**
- **CUDA time > CPU time**: Good - GPU está haciendo el trabajo
- **CUDA time << CPU time**: Bad - Overhead de lanzamiento domina
- **High memory**: Identifica operaciones que consumen mucha memoria

---

### Patrones de Uso Ineficientes

```{code-cell} ipython3
:tags: [skip-execution]

# MALO: Muchas operaciones pequeñas (kernel launch overhead)
for i in range(1000):
    x = x + 1  # 1000 kernel launches

# BUENO: Una operación grande (batch)
x = x + 1000  # 1 kernel launch

# ===================================

# MALO: Sincronización innecesaria
for i in range(100):
    x = model(x)
    print(x[0, 0].item())  # .item() sincroniza cada vez!

# BUENO: Batch las sincronizaciones
results = []
for i in range(100):
    x = model(x)
    results.append(x[0, 0])
torch.cuda.synchronize()
print([r.item() for r in results])
```

```{admonition} Antipatrón: Sincronización Implícita
:class: warning
Operaciones que fuerzan sincronización GPU→CPU:
- `.item()` - convierte tensor a Python scalar
- `.numpy()` - convierte a NumPy array
- `print(tensor)` - muestra valores
- Cualquier comparación Python: `if tensor > 0:`

**Impacto**: Cada sincronización detiene la GPU pipeline asíncrona, añadiendo 10-100μs de overhead.

**Solución**: Acumula resultados en GPU, sincroniza una vez al final.
```

---

```{admonition} Lectura Opcional — Pinned Memory y CUDA Streams
:class: seealso

Los dos temas siguientes van más allá de los 25 minutos del bloque. Revísalos por tu cuenta
si quieres profundizar en transferencias asíncronas y paralelismo de operaciones.
```

### Pinned Memory para Transferencias Rápidas *(Opcional)*

```{code-cell} ipython3
:tags: [skip-execution]

# Sin pinned memory (transferencia síncrona lenta)
x_cpu = torch.randn(1000, 1000)
x_gpu = x_cpu.cuda()  # Bloquea hasta que termina

# Con pinned memory (transferencia asíncrona rápida)
x_pinned = torch.randn(1000, 1000).pin_memory()
x_gpu = x_pinned.cuda(non_blocking=True)  # No bloquea

# Continuar con otro trabajo mientras se transfiere
y = torch.randn(1000, 1000, device='cuda')
z = y @ y.t()

# Sincronizar cuando necesites x_gpu
torch.cuda.synchronize()
```

**¿Por qué funciona?** Pinned memory (también llamada page-locked) es memoria que el OS garantiza que no moverá. Esto permite DMA (Direct Memory Access) - la GPU puede leer directamente desde RAM sin pasar por CPU.

**Cuándo usar:**
- DataLoaders para training: `DataLoader(..., pin_memory=True)`
- Transferencias grandes y frecuentes CPU→GPU
- Pipelines donde puedes overlap compute con transfer

---

### CUDA Streams: Paralelismo de Operaciones *(Opcional)*

```{code-cell} ipython3
:tags: [skip-execution]

# Stream por defecto (secuencial)
x = torch.randn(1000, 1000, device='cuda')
y = x * 2  # Ejecuta en stream por defecto
z = y + 3  # Espera a que termine y

# Crear streams custom (paralelo)
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    a = torch.randn(1000, 1000, device='cuda')
    b = a @ a.t()

with torch.cuda.stream(stream2):
    c = torch.randn(1000, 1000, device='cuda')
    d = c @ c.t()

# Los dos matmuls pueden ejecutar en paralelo si hay recursos disponibles
torch.cuda.synchronize()  # Esperar a ambos streams
```

---

## Ejercicio Práctico: Análisis de Kernel Launch

```{code-cell} ipython3
import math

def analyze_kernel_launch(threads_per_block, num_blocks, num_SMs, max_warps_per_SM):
    """Analiza un kernel launch para entender occupancy."""

    warps_per_block = math.ceil(threads_per_block / 32)
    total_warps = warps_per_block * num_blocks

    # Warps que pueden ejecutar concurrentemente
    concurrent_warps = num_SMs * max_warps_per_SM

    # Waves necesarias
    waves = math.ceil(total_warps / concurrent_warps)

    return {
        "warps_per_block": warps_per_block,
        "total_warps": total_warps,
        "concurrent_warps": concurrent_warps,
        "waves": waves,
        "theoretical_occupancy": min(1.0, total_warps / concurrent_warps)
    }

# Configuración de ejemplo (A100-like)
config = {
    "threads_per_block": 256,
    "num_blocks": 200,
    "num_SMs": 108,
    "max_warps_per_SM": 64
}

result = analyze_kernel_launch(**config)

print("=== Análisis de Kernel Launch ===")
print(f"Threads/bloque: {config['threads_per_block']}")
print(f"Bloques: {config['num_blocks']}")
print(f"Warps/bloque: {result['warps_per_block']}")
print(f"Total warps: {result['total_warps']}")
print(f"Warps concurrentes max: {result['concurrent_warps']}")
print(f"Waves necesarias: {result['waves']}")
print(f"Occupancy teórico: {result['theoretical_occupancy']:.1%}")

# ¿Qué pasa si duplicamos threads/bloque?
config2 = {**config, "threads_per_block": 512}
result2 = analyze_kernel_launch(**config2)
print(f"\nCon 512 threads/bloque: {result2['waves']} waves")
```

---

````{admonition} Actividad de Trabajo - Bloque 2 (25 minutos)
:class: note

**Ejercicio 4: Comparación CPU vs GPU (10 min)**

Implementa y mide el tiempo de una multiplicación de matrices:

```python
import torch
import time

# CPU
x_cpu = torch.randn(5000, 5000)
start = time.time()
y_cpu = x_cpu @ x_cpu.t()
cpu_time = time.time() - start

# GPU — warm-up obligatorio: la primera llamada compila kernels de cuBLAS
x_gpu = torch.randn(5000, 5000, device='cuda')
_ = x_gpu @ x_gpu.t(); torch.cuda.synchronize()   # warm-up

torch.cuda.synchronize()
start = time.time()
y_gpu = x_gpu @ x_gpu.t()
torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

Varía el tamaño de matriz (1000, 2000, 5000, 10000). ¿A partir de qué tamaño vale la pena GPU?

**Ejercicio 5: Profiling y Memory (10 min)**

Usa `torch.profiler` para analizar un forward pass de un modelo pequeño.
Completa el bloque `# TODO` y ejecuta el análisis:

```python
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(1000, 2000),
    torch.nn.ReLU(),
    torch.nn.Linear(2000, 1000)
).cuda()

x = torch.randn(32, 1000, device='cuda')

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(10):
        # TODO: hacer el forward pass del modelo con x
        pass

# TODO: imprimir la tabla ordenada por cuda_time_total (top 10 ops)
# Identifica: ¿Cuál operación consume más CUDA time? ¿Y más memoria?
```

**Ejercicio 6: Antipatrón Detection (5 min)**

Identifica qué está mal en este código y corrígelo:

```python
# Código problemático
results = []
for i in range(100):
    x = torch.randn(100, 100).cuda()
    y = x @ x.t()
    results.append(y[0, 0].item())
```
````

---

## Cierre de Sesión *(10 min)*

## Resumen

```{admonition} Resumen de la Sesión
:class: important

**Bloque 1 - Arquitectura GPU:**
- **CPU vs GPU**: Latencia vs throughput (chef vs fábrica)
- **SIMT**: Todos los threads ejecutan la misma instrucción con datos diferentes
- **Jerarquía de ejecución**: Grid→Blocks→Threads→Warps (32 threads/warp)
- **Jerarquía de memoria**: Registers (rápido) → Global HBM (lento pero masivo)
- **Divergencia de warp**: El antipatrón #1 - evita branches dentro de warps
- **Por qué optimizar**: 10x mejora = $23K/año de ahorro

**Bloque 2 - PyTorch en GPU:**
- **Creación de tensores**: `torch.randn(..., device='cuda')` es más eficiente que `.cuda()`
- **Transferencias CPU↔GPU**: PCIe (32 GB/s) es 60x más lento que HBM (2 TB/s)
- **Memory layout**: Tensores no contiguos violan coalescing → usa `.contiguous()`
- **Caching allocator**: PyTorch cachea memoria para evitar `cudaMalloc` costosos
- **Monitoreo**: `memory_allocated()` (activo) vs `memory_reserved()` (cache)
- **Profiling**: `torch.profiler` para identificar bottlenecks de CUDA time y memoria
- **Antipatrón**: `.item()` en loops causa sincronización costosa

**Checklist de optimización básica:**
- [ ] ¿Tienes suficiente paralelismo? (miles de operaciones independientes)
- [ ] ¿Evitas divergencia de warp? (threads consecutivos toman mismo camino)
- [ ] ¿Minimizas transferencias CPU↔GPU? (crea tensores directamente en GPU)
- [ ] ¿Verificas contiguidad? (`.is_contiguous()` antes de ops intensivas)
- [ ] ¿Evitas sincronizaciones implícitas? (no `.item()` en loops)
```

```{admonition} Verifica tu comprensión
:class: note

**Pregunta 1**: ¿Cuántos warps caben en un bloque de 256 threads?
<details><summary>Respuesta</summary>
256 / 32 = 8 warps. Cada warp tiene 32 threads, así que un bloque de 256 threads contiene 8 warps que se ejecutan en paralelo.
</details>

**Pregunta 2**: GPU con 80 SMs ejecutando 4 warps/SM. Lanzas 100 bloques de 256 threads. ¿Cuántas waves?
<details><summary>Respuesta</summary>
- Warps por bloque: 256/32 = 8
- Warps totales: 100 bloques × 8 = 800 warps
- Warps concurrentes: 80 SMs × 4 = 320 warps
- Waves: ceil(800/320) = 3 waves
</details>
```

### Para Pensar

> *Si las GPUs tienen miles de cores pero alta latencia de memoria, ¿por qué no simplemente agregamos más memoria caché para reducir latencia en vez de depender de throughput?*

---

## Próximos Pasos

En la siguiente sesión exploraremos **CUDA Kernels y Triton**: escribirás tu primer kernel custom, entenderás indexación 2D/3D, memory coalescing, shared memory, y el modelo de programación Triton que simplifica todo esto.

---

## Tarea para Casa

### Ejercicio 1: Costo de Optimización

Una empresa ejecuta 10M inferencias/día con un kernel que toma 50ms. Si optimizas a 10ms:
- ¿Cuántas horas GPU ahorras por día?
- ¿Cuál es el ahorro anual a $3/hora?

### Ejercicio 2: Análisis de Código

¿Por qué este código es ineficiente en GPU y cómo lo mejorarías?

```python
model = torch.nn.Linear(1000, 1000).cuda()
for i in range(1000):
    x = torch.randn(10, 1000).cuda()
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(f"Iteration {i}: loss = {loss.item()}")
```

---

## Referencias

- NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). NVIDIA Developer.
- PyTorch. [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html). PyTorch Docs.
- Harris, M. (2013). [How to Optimize Data Transfers in CUDA](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/). NVIDIA Blog.
- Hwu, W., Kirk, D., & El Hajj, I. (2022). Programming Massively Parallel Processors (4th ed.). Morgan Kaufmann.

---

## Lecturas Recomendadas

- **D2L: GPUs** - [Capítulo 6.7](https://d2l.ai/chapter_builders-guide/use-gpu.html). Conceptos básicos de gestión de memoria y arrays en GPUs con PyTorch.
- **D2L: Hardware** - [Capítulo 13.4](https://d2l.ai/chapter_computational-performance/hardware.html). Una excelente base para entender la jerarquía de memoria y arquitectura de cómputo.
- **PyTorch Performance Tuning Guide** - [Official Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html). Patrones de optimización específicos de PyTorch.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*
