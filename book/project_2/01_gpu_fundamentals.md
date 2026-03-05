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

# GPU Fundamentals: Arquitectura y Paralelismo Masivo


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/01_gpu_fundamentals.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 1
> **Tiempo de lectura:** ~35 minutos

---

## Introducción

¿Por qué los videojuegos necesitan una GPU dedicada? ¿Por qué entrenar un modelo de deep learning es órdenes de magnitud más rápido en GPU? La respuesta está en arquitecturas fundamentalmente diferentes diseñadas para resolver problemas distintos.

Antes de optimizar código para GPU, necesitas entender **por qué** las GPUs son tan poderosas para ciertos tipos de cómputo y **cómo** su arquitectura dicta las estrategias de optimización. Sin este conocimiento, estarías optimizando a ciegas.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Explicar las diferencias fundamentales entre CPUs y GPUs (latencia vs throughput)
- Describir el modelo de ejecución SIMT y la jerarquía Grid→Blocks→Threads→Warps
- Comprender la jerarquía de memoria GPU y calcular latencias relativas
- Identificar cuándo usar GPU vs CPU para un problema dado basándote en arithmetic intensity
- Justificar económicamente por qué optimizar kernels (ahorro de costos y energía)
```

---

## La Filosofía Fundamental: CPU vs GPU

### CPUs: Pocos Trabajadores Muy Inteligentes

Los procesadores tradicionales están optimizados para **ejecutar tareas complejas y secuenciales rápidamente**. Imagina una CPU como un chef experto en una cocina pequeña:

- **Pocos cores** (8-64 típicamente)
- **Cores muy potentes** con predicción de branches, ejecución fuera de orden
- **Caches grandes** para minimizar latencia de memoria
- **Clock alto** (3-5 GHz)
- **Latencia baja**: Optimizado para responder rápidamente a una tarea individual

Las CPUs están diseñadas para ejecutar **una tarea muy rápido**.

### GPUs: Miles de Trabajadores Simples

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

```{admonition} 🧠 Modelo Mental
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

## El Modelo SIMT (Single Instruction, Multiple Threads)

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

### ¿Por qué es esto tan poderoso?

1. **Simplicidad lógica**: El hardware es más simple si todos hacen lo mismo
2. **Escalabilidad**: Podemos agregar más cores sin cambiar el código
3. **Ancho de banda**: Con miles de threads, podemos mantener la memoria constantemente ocupada

---

## Jerarquía de Ejecución: Grids, Blocks, Threads, Warps

Las GPUs organizan el paralelismo en una jerarquía clara:

```
Grid (todo el kernel)
  └── Blocks (grupos de threads)
        └── Threads (unidades de ejecución)
              └── Warps (32 threads ejecutando la misma instrucción)
```

### Grid

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

### Blocks

Los **Blocks** son grupos de threads que:
- Pueden comunicarse entre sí via shared memory
- Se sincronizan con `__syncthreads()`
- Se ejecutan en un solo Streaming Multiprocessor (SM)
- Típicamente 128-512 threads por bloque

### Threads

Cada **Thread** es una unidad de ejecución que:
- Tiene su propio program counter
- Tiene acceso a registros privados
- Ejecuta el mismo código pero con diferentes datos
- Se identifica con `threadIdx` y `blockIdx`

### Warps: La Unidad Crítica

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
// Divergencia de warp (¡MAL!)
if (threadIdx.x % 2 == 0) {
    // 16 threads ejecutan esto
    resultado = sumaCompleja();
} else {
    // 16 threads ESPERAN
    resultado = multiplicacionSimple();
}
// Los 16 threads que terminaron rápido ESPERAN a los otros
```

```{admonition} ⚠️ Antipatrón: Divergencia de Warp
:class: warning
**Problema**: Branches dentro de un warp causan serialización - los threads que toman un camino esperan a los que toman el otro.

**Causa común**: `if (threadIdx.x % 2 == 0)` hace que mitad del warp ejecute código A y mitad ejecute código B.

**Impacto**: Rendimiento se divide por 2 (o peor si hay múltiples branches).

**Solución**: Reorganiza datos para que threads consecutivos tomen el mismo camino, o usa operaciones predicate-based como `tl.where()`.
```

```{admonition} 🎯 En tu proyecto
:class: note
Cuando implementes kernels de filtrado o condicionales, asegúrate de que los threads dentro de un warp (grupos de 32) tomen decisiones similares. Por ejemplo, si filtras por umbral, ordena datos primero.
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

### ACA Framework: Simulando el Grid en Python Puro

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

```{admonition} 🧠 Modelo Mental: Latencia vs Throughput
:class: hint
Piensa en la memoria como un sistema de entregas:

**CPU (Latencia baja)**: Como un servicio de mensajería express - llega rápido (100ns) pero solo puedes enviar pocas cajas a la vez (50 GB/s).

**GPU (Throughput alto)**: Como un tren de carga - tarda más en llegar (400 cycles), pero cuando llega trae TONELADAS de datos (2-3 TB/s).

¿Cómo compensa la GPU la alta latencia? **Teniendo miles de threads esperando** - mientras unos esperan su tren, otros procesan el que acaba de llegar. Es como una fábrica donde mientras algunos empleados esperan materiales, otros continúan trabajando.
```

```{admonition} ⚡ Tip de Performance
:class: important
Para maximizar throughput de memoria GPU:
1. Lanza suficientes threads para ocultar latencia (ocupancy > 50%)
2. Accede memoria en patrones coalesced (threads consecutivos → direcciones consecutivas)
3. Mantén datos en cache/shared memory cuando sea posible
```

---

## Métricas de Performance GPU

### Occupancy

**Occupancy** = Warps activos / Warps máximos posibles

- Depende del uso de registros y shared memory por thread
- Más occupancy ≠ siempre mejor performance
- Pero muy bajo occupancy = GPU infrautilizada

### Throughput

Medido en:
- **FLOPS**: Operaciones de punto flotante por segundo
- **GB/s**: Ancho de banda efectivo de memoria

### Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes transferidos
```

- **Memory bound**: Bajo arithmetic intensity, limitado por ancho de banda
- **Compute bound**: Alto arithmetic intensity, limitado por FLOPS

---

## Comparación Práctica: Suma de Vectores

### En CPU (Secuencial)

```c
void suma_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
// Tiempo: O(n) secuencial
```

### En GPU (Paralelo)

```cuda
__global__ void suma_gpu(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
// Todos los threads hacen esto simultáneamente
// Tiempo: O(n / num_threads)
```

¿La diferencia? Con 10,000 threads, el kernel GPU es **~10,000 veces más rápido** (teóricamente). En la práctica, movimiento de datos y overhead reducen esto, pero aun así es órdenes de magnitud más rápido.

---

## Cuándo Usar GPU vs CPU

### Usa GPU si:

- ✓ Tienes mucho paralelismo (miles de operaciones independientes)
- ✓ La relación entre trabajo y movimiento de datos es alta (aritmética intensiva)
- ✓ Puedes tolerar divergencia de warp limitada
- ✓ El problema es regular (código similar para todos los threads)

### Usa CPU si:

- ✓ El problema es muy secuencial o tiene muchas ramificaciones
- ✓ Necesitas latencia baja para operaciones individuales
- ✓ El código es muy complicado o variable por dato
- ✓ Movimiento de datos es el cuello de botella

---

## Por Qué Importa Optimizar

### Impacto Económico

```
Costo de GPU A100 en cloud: ~$2-3/hora

Kernel no optimizado: 100ms por inferencia
Kernel optimizado:     10ms por inferencia

A escala (1M inferencias/día):
- No optimizado: 27.7 horas GPU → ~$70/día
- Optimizado:    2.77 horas GPU → ~$7/día

Ahorro anual: ~$23,000 por modelo
```

### Impacto Energético

Las GPUs de datacenter consumen 300-700W. Kernels eficientes:
- Reducen tiempo de ejecución
- Reducen energía total consumida
- Reducen huella de carbono

### Impacto en Experiencia de Usuario

Para aplicaciones interactivas:
- 100ms de latencia = usuario nota el delay
- 10ms de latencia = se siente instantáneo

---

## Resumen

```{admonition} Resumen
:class: important
**Conceptos clave cubiertos:**
- **CPU vs GPU**: Latencia vs throughput (chef vs fábrica)
- **SIMT**: Todos los threads ejecutan la misma instrucción con datos diferentes
- **Jerarquía de ejecución**: Grid→Blocks→Threads→Warps (32 threads/warp)
- **Jerarquía de memoria**: Registers (rápido) → Global (lento pero masivo)
- **Divergencia de warp**: El antipatrón #1 - evita branches dentro de warps
- **Por qué optimizar**: 10x mejora = $23K/año de ahorro

**Checklist de optimización básica:**
- [ ] ¿Tienes suficiente paralelismo? (miles de operaciones independientes)
- [ ] ¿Evitas divergencia de warp? (threads consecutivos toman mismo camino)
- [ ] ¿Minimizas transferencias CPU↔GPU?
- [ ] ¿Usas el tipo correcto de memoria? (shared > global)
```

```{admonition} 📊 Cómo verificar
:class: tip
Para verificar que entiendes los conceptos:
1. Calcula warps: `num_warps = ceil(threads_per_block / 32)`
2. Estima ocupancy: `threads_activos / threads_max_GPU`
3. Identifica divergencia: ¿Hay `if (threadIdx % algo)` en tu código?
```

```{admonition} 🎯 En tu proyecto
:class: note
Usarás estos conceptos cuando:
- Diseñes el grid/block layout para tus kernels
- Debuggees performance (¿por qué mi kernel es lento?)
- Optimizes memoria (¿shared o global?)
- Evalúes si un problema vale la pena GPUficar
```

---

## Ejercicios y Reflexión

### Ejercicio 1: Identificar Oportunidades

Para cada problema, ¿sería mejor GPU o CPU? Justifica:
1. Filtrar outliers de un dataset de 10 millones de registros
2. Ejecutar un compilador con análisis complejo
3. Multiplicación de matrices 10,000 x 10,000
4. Parsing de JSON con estructura variable
5. Procesamiento de imágenes (convoluciones, filtros)

### Ejercicio 2: Divergencia de Warp

¿Por qué este código es malo en GPU?
```cuda
__global__ void kernel_malo(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx % 2 == 0) {
        data[idx] = sqrt(data[idx]);  // Operación lenta
    } else {
        data[idx] = data[idx] * 2;    // Operación rápida
    }
}
```
¿Cómo lo mejorarías?

### Ejercicio 3: Cálculo de Impacto

Una empresa ejecuta 10M inferencias/día con un kernel que toma 50ms. Si optimizas a 10ms:
- ¿Cuántas horas GPU ahorras por día?
- ¿Cuál es el ahorro anual a $3/hora?

```{admonition} ✅ Verifica tu comprensión
:class: note
**Pregunta 1**: ¿Cuántos warps caben en un bloque de 256 threads?
<details><summary>Respuesta</summary>
256 / 32 = 8 warps. Cada warp tiene 32 threads, así que un bloque de 256 threads contiene 8 warps que se ejecutan en paralelo.
</details>

**Pregunta 2**: ¿Por qué stride access (saltar posiciones de memoria) es malo?
<details><summary>Respuesta</summary>
Porque viola coalescing. Si thread 0 lee addr[0] y thread 1 lee addr[1000], están en diferentes bloques de 128 bytes, requiriendo múltiples transacciones en vez de una.
</details>

**Pregunta 3**: GPU con 80 SMs ejecutando 4 warps/SM. Lanzas 100 bloques de 256 threads. ¿Cuántas waves?
<details><summary>Respuesta</summary>
- Warps por bloque: 256/32 = 8
- Warps totales: 100 bloques × 8 = 800 warps
- Warps concurrentes: 80 SMs × 4 = 320 warps
- Waves: ceil(800/320) = 3 waves
</details>

**Pregunta 4**: ¿Qué tipo de workload se beneficia más de GPU?
a) Sorting de array de 100 elementos
b) Multiplicación de matrices 10000×10000
<details><summary>Respuesta</summary>
(b) MatMul grande. Tiene suficiente paralelismo (10000² operaciones) y alta arithmetic intensity. Sorting de 100 elementos tiene demasiado poco paralelismo para saturar GPU.
</details>
```

### Para Pensar

> *Si las GPUs tienen miles de cores, ¿por qué no simplemente ponemos más cores en las CPUs y eliminamos la necesidad de programación GPU especializada?*

---

## Próximos Pasos

En la siguiente semana, exploraremos **CUDA y PyTorch Internals**: conceptos fundamentales de CUDA, cómo PyTorch gestiona tensores en GPU, memory allocation, CUDA streams, y cómo usar el profiler para identificar bottlenecks.

---

## Errores Comunes

```{admonition} Errores frecuentes en GPU
:class: warning

1. **Divergencia de warp**: Branches dentro de un warp causan serialización. Evita `if (threadIdx.x < 16)`.
2. **Olvidar que un warp = 32 threads**: Siempre calcula `num_warps = ceil(threads / 32)`.
3. **Confundir bloques con warps**: Un bloque puede tener múltiples warps.
```

## Ejercicio Práctico: Calcular Warps y Waves

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

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/). NVIDIA Developer.
- Harris, M. (2013). [How to Optimize Data Transfers in CUDA](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/). NVIDIA Blog.
- Hwu, W., Kirk, D., & El Hajj, I. (2022). Programming Massively Parallel Processors (4th ed.). Morgan Kaufmann.

---

## Lecturas Recomendadas

- **D2L: GPUs** - [Capítulo 6.7](https://d2l.ai/chapter_builders-guide/use-gpu.html). Conceptos básicos de gestión de memoria y arrays en GPUs con PyTorch.
- **D2L: Hardware** - [Capítulo 13.4](https://d2l.ai/chapter_computational-performance/hardware.html). Una excelente base para entender la jerarquía de memoria y arquitectura de cómputo antes de entrar a Triton.
