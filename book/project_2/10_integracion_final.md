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

# Evaluación Final y Presentaciones

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q torch triton datasets
```

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/project_2/10_integracion_final.ipynb)
```

> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 10
> **Tiempo de lectura:** ~40 minutos

---

## Introducción

Llegamos a la sesión final del proyecto. Has construido un agente capaz de generar kernels GPU automáticamente, desde prompts básicos hasta sistemas con feedback y mejora iterativa. Esta sesión cierra el círculo: evaluación final en KernelBench, presentaciones de equipo, y entrega del reporte.

**Formato de la sesión:**
- **Bloque 1 (60 min):** Evaluación final del agente + gramáticas como restricción
- **Bloque 2 (60 min):** Presentaciones de equipos + discusión grupal + cierre

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Ejecutar evaluación final del agente en KernelBench (L1-L3) con targets claros
- Explicar cómo gramáticas capturan restricciones GPU para validez estructural
- Identificar límites de generación automática (qué es generable vs qué requiere innovación)
- Estructurar presentación técnica efectiva (arquitectura, demo, resultados, lecciones)
- Preparar reporte final con código, experimentos, resultados y análisis completos
```

---

## Bloque 1: Evaluación Final del Agente

### Evaluación Final en KernelBench

#### Protocolo de Evaluación Final

Tu agente debe ser evaluado exhaustivamente en KernelBench para determinar su capacidad real de generación de kernels. El protocolo final incluye:

**Objetivos por Nivel:**

| Nivel | Descripción | Target Pass Rate | Complejidad |
|-------|-------------|-----------------|-------------|
| **L1** | Operaciones elementwise básicas | >90% | Baja |
| **L2** | Reducciones y operaciones con memoria | >60% | Media |
| **L3** | Kernels complejos (matmul, softmax) | >20% | Alta |

```{code-cell} ipython3
:tags: [skip-execution]

# Evaluación final completa
import json
from pathlib import Path
from typing import Dict, List
import torch

class FinalEvaluator:
    """Evaluador final para KernelBench."""

    def __init__(
        self,
        agent,
        kernelbench_path: str = "./kernelbench",
        output_dir: str = "./results/final_eval"
    ):
        self.agent = agent
        self.kernelbench_path = Path(kernelbench_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_evaluation(self) -> Dict[str, any]:
        """
        Ejecuta evaluación completa en todos los niveles.

        Returns:
            Diccionario con resultados por nivel y métricas agregadas.
        """
        results = {
            "L1": self._evaluate_level("L1"),
            "L2": self._evaluate_level("L2"),
            "L3": self._evaluate_level("L3"),
        }

        # Calcular métricas agregadas
        results["aggregate"] = self._compute_aggregate_metrics(results)

        # Guardar resultados
        self._save_results(results)

        return results

    def _evaluate_level(self, level: str) -> Dict[str, any]:
        """Evalúa un nivel específico de KernelBench."""
        problems = self._load_problems(level)

        level_results = {
            "total": len(problems),
            "passed": 0,
            "failed": 0,
            "timeout": 0,
            "problems": {}
        }

        for problem in problems:
            print(f"Evaluando {level}/{problem['name']}...")

            try:
                # Generar kernel
                kernel_code = self.agent.generate(
                    problem["description"],
                    max_attempts=3
                )

                # Verificar correctitud
                is_correct = self._verify_correctness(
                    kernel_code,
                    problem["test_cases"]
                )

                # Benchmark si es correcto
                if is_correct:
                    perf_metrics = self._benchmark_kernel(
                        kernel_code,
                        problem["inputs"]
                    )
                    level_results["passed"] += 1
                    status = "passed"
                else:
                    perf_metrics = None
                    level_results["failed"] += 1
                    status = "failed"

                level_results["problems"][problem["name"]] = {
                    "status": status,
                    "kernel_code": kernel_code,
                    "performance": perf_metrics
                }

            except TimeoutError:
                level_results["timeout"] += 1
                level_results["problems"][problem["name"]] = {
                    "status": "timeout"
                }

        # Calcular pass rate
        level_results["pass_rate"] = (
            level_results["passed"] / level_results["total"]
        )

        return level_results

    def _compute_aggregate_metrics(
        self,
        results: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calcula métricas agregadas."""
        total_problems = sum(r["total"] for r in results.values() if "total" in r)
        total_passed = sum(r["passed"] for r in results.values() if "passed" in r)

        return {
            "overall_pass_rate": total_passed / total_problems,
            "L1_pass_rate": results["L1"]["pass_rate"],
            "L2_pass_rate": results["L2"]["pass_rate"],
            "L3_pass_rate": results["L3"]["pass_rate"],
            "total_evaluated": total_problems,
            "total_passed": total_passed
        }

    def _save_results(self, results: Dict):
        """Guarda resultados en JSON."""
        output_path = self.output_dir / "final_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Resultados guardados en {output_path}")

        # Generar reporte en Markdown
        self._generate_report(results)

    def _generate_report(self, results: Dict):
        """Genera reporte Markdown de resultados."""
        report_path = self.output_dir / "FINAL_REPORT.md"

        with open(report_path, 'w') as f:
            f.write("# Evaluación Final - KernelBench\n\n")

            # Resumen
            agg = results["aggregate"]
            f.write("## Resumen Ejecutivo\n\n")
            f.write(f"- **Pass Rate Global**: {agg['overall_pass_rate']:.1%}\n")
            f.write(f"- **Problemas Evaluados**: {agg['total_evaluated']}\n")
            f.write(f"- **Problemas Resueltos**: {agg['total_passed']}\n\n")

            # Por nivel
            f.write("## Resultados por Nivel\n\n")
            for level in ["L1", "L2", "L3"]:
                level_data = results[level]
                f.write(f"### {level}\n\n")
                f.write(f"- Pass Rate: {level_data['pass_rate']:.1%}\n")
                f.write(f"- Passed: {level_data['passed']}/{level_data['total']}\n")
                f.write(f"- Failed: {level_data['failed']}\n")
                f.write(f"- Timeout: {level_data['timeout']}\n\n")

            # Análisis
            f.write("## Análisis\n\n")
            self._write_analysis(f, results)

        print(f"Reporte generado en {report_path}")

    def _write_analysis(self, f, results: Dict):
        """Escribe sección de análisis."""
        agg = results["aggregate"]

        # Comparar con targets
        targets = {"L1": 0.90, "L2": 0.60, "L3": 0.20}

        f.write("### Comparación con Targets\n\n")
        f.write("| Nivel | Target | Actual | Status |\n")
        f.write("|-------|--------|--------|--------|\n")

        for level, target in targets.items():
            actual = results[level]["pass_rate"]
            status = "✅" if actual >= target else "❌"
            f.write(f"| {level} | {target:.0%} | {actual:.1%} | {status} |\n")


# Uso en evaluación final
evaluator = FinalEvaluator(agent=my_agent)
final_results = evaluator.run_full_evaluation()

print(f"Pass Rate L1: {final_results['L1']['pass_rate']:.1%}")
print(f"Pass Rate L2: {final_results['L2']['pass_rate']:.1%}")
print(f"Pass Rate L3: {final_results['L3']['pass_rate']:.1%}")
```

#### Interpretación de Resultados

**Pass Rate L1 >90%:**
- Indica que el agente domina operaciones básicas elementwise
- Expected: suma, multiplicación, ReLU, sigmoid funcionan correctamente
- Si <90%: revisar templates básicos y manejo de máscaras

**Pass Rate L2 >60%:**
- Indica capacidad de reducciones y operaciones con memoria compartida
- Expected: sum, max, softmax básico funcionan
- Si <60%: revisar patrones de reducción y sincronización

**Pass Rate L3 >20%:**
- Indica que el agente puede abordar problemas complejos ocasionalmente
- Expected: algunos matmul, layernorm funcionan
- Si <20%: normal para primeras iteraciones del agente

```{admonition} Umbral de Éxito
:class: note
Un agente que logra **>90% en L1, >60% en L2, y >20% en L3** demuestra capacidad práctica de generación automática de kernels GPU. Esto supera significativamente la capacidad de LLMs base sin especialización.
```

---

## Actividad de Trabajo — Bloque 1 (25 minutos)

**Ejercicio 1: Ejecutar `FinalEvaluator` e interpretar resultados (15 min)**

Instancia `FinalEvaluator` con tu agente y ejecuta la evaluación completa.
Para acelerar, limita a **5 L1 + 3 L2 + 1 L3** (usa `problems[:5]` por nivel).
Completa la tabla de resultados:

| Nivel | Problems tested | Passed | Failed | Timeout | Pass rate | Target | ¿Logrado? |
|-------|----------------|--------|--------|---------|-----------|--------|----------|
| L1 | 5 | | | | | >90% | |
| L2 | 3 | | | | | >60% | |
| L3 | 1 | | | | | >20% | |

Revisa el `FINAL_REPORT.md` generado. ¿El análisis automático coincide con
tu observación manual de los errores?

**Ejercicio 2: Clasificar errores de L2 con la taxonomía de S8 (10 min)**

Para los problemas L2 que fallaron, usa `enhanced_classify_error` de S8 y
clasifica cada fallo:

| Problem ID | Error type | Subcategoría | Hint generado |
|------------|-----------|--------------|---------------|
| ... | RUNTIME | OUT_OF_BOUNDS | "Añade máscara..." |

¿La categoría más frecuente en L2 es la misma que en L1? ¿Qué pasos del
pipeline de S5 (`evaluate_kernel`) fallan más: `step1_compile`, `step2_execute`,
o `step3_correctness`?

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

1. **El patrón L1 → L2 → L3**: Tu pass rate baja drásticamente de L1 a L2.
   ¿Cuál es la razón fundamental que explica esa caída? (Pista: no es solo
   líneas de código — piensa en el tipo de operación y la interacción
   entre threads.)
2. **Timeout vs failed**: `FinalEvaluator` reporta timeouts separados de fallos.
   ¿Por qué importa esa distinción para el loop de S7? ¿Usarías la misma
   estrategia de retry para un timeout que para un `RuntimeError`?
3. **Validez estructural vs correctitud semántica**: El kernel de softmax
   generado por gramática en el Bloque 2 pasa la gramática pero es ineficiente.
   ¿Cómo diseñarías un test adicional en `evaluate_kernel()` que detecte kernels
   válidos pero más de 10x más lentos que PyTorch?
```

---

## Bloque 2: Límites, Gramáticas y Presentación Final

### Gramáticas como Restricción del Generador

#### De Proyecto 1 a Proyecto 2: Gramáticas en GPU

En Project 1 S6-S7, que cursaste en paralelo a este módulo, aprendiste cómo **gramáticas estructuran la salida** de LLMs para garantizar validez sintáctica. Este concepto es aún más crítico en GPU porque:

1. **Hardware no perdona errores**: Acceso fuera de límites → crash
2. **Sincronización incorrecta**: Race conditions → resultados incorrectos
3. **Patrones específicos**: Solo ciertos patrones de acceso son eficientes

#### Restricciones GPU en Gramática

**Restricción 1: Indexación Válida**

```python
# Gramática simplificada para indexación en Triton
IndexExpr ::= BlockId "*" BlockSize "+" ThreadOffset
            | "tl.program_id(axis=" Integer ")" "*" Constant "+" Offset

# Ejemplo válido generado
offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# La gramática PREVIENE esto (indexación indirecta):
# offsets = indices[offsets]  # ❌ No está en la gramática
```

**Restricción 2: Máscaras Obligatorias**

```python
# Gramática fuerza máscaras en loads/stores
LoadExpr ::= "tl.load(" PointerExpr "," "mask=" MaskExpr "," "other=" DefaultValue ")"

# Ejemplo válido
x = tl.load(x_ptr + offsets, mask=offsets < n, other=0.0)

# La gramática PREVIENE esto:
# x = tl.load(x_ptr + offsets)  # ❌ Sin máscara
```

**Restricción 3: Operaciones Atómicas para Reducciones Globales**

```python
# Gramática distingue reducciones locales vs globales
GlobalReduction ::= LocalReduction ThenAtomic
LocalReduction ::= "tl.sum(" Vector "," "axis=" Integer ")"
ThenAtomic ::= "tl.atomic_add(" GlobalPtr "," LocalResult ")"

# Ejemplo válido (suma global)
local_sum = tl.sum(x, axis=0)  # Reducción dentro del bloque
tl.atomic_add(result_ptr, local_sum)  # Acumular globalmente

# La gramática PREVIENE race conditions:
# result[0] = result[0] + local_sum  # ❌ No atómico
```

#### Implicaciones Prácticas

```{admonition} Validez Estructural vs Correctitud Semántica
:class: important
**Gramática garantiza:**
- Sintaxis válida de Triton
- Máscaras presentes en accesos de memoria
- Operaciones atómicas para sincronización global

**Gramática NO garantiza:**
- Algoritmo correcto (puede calcular la operación equivocada)
- Performance óptima (puede ser válido pero lento)
- Ausencia de deadlocks sutiles
```

**Ejemplo: Gramática genera código válido pero ineficiente**

```{code-cell} ipython3
:tags: [skip-execution]

# Código generado por gramática (VÁLIDO pero INEFICIENTE)
@triton.jit
def softmax_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load válido (con máscara)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute (algoritmo correcto)
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum

    # Store válido (con máscara)
    tl.store(y_ptr + offsets, y, mask=mask)

# Problema: INEFICIENTE porque no usa memoria compartida para x_max y x_sum
# Un kernel óptimo haría reducción en dos pasadas
```

La gramática **asegura validez** pero no optimización. Para performance, necesitas:
1. Patrones conocidos en el corpus (flash attention, tiling)
2. Feedback de benchmarking
3. Técnicas de búsqueda (beam search, evolutionary)

---

### Límites de la Generación Automática

#### ¿Qué Kernels Son Generables?

**Tier 1: Operaciones Elementwise (Altamente Generables)**

Características:
- Patrón regular y conocido
- No requiere coordinación compleja
- Template directo aplicable

```python
# Ejemplos generables con alta confianza
- Vector addition: x + y
- ReLU: max(x, 0)
- Sigmoid: 1 / (1 + exp(-x))
- Multiplicación elemento-elemento: x * y
- Dropout: x * mask / (1 - p)
```

**Tier 2: Reducciones y Operaciones con Memoria (Moderadamente Generables)**

Características:
- Patrones conocidos pero más complejos
- Requiere sincronización dentro del bloque
- Templates con variaciones

```python
# Ejemplos generables con confianza media
- Sum reduction: tl.sum(x, axis=0)
- LayerNorm básico: (x - mean) / std
- Softmax: exp(x) / sum(exp(x))
- Max pooling: tl.max(x, axis=1)
```

**Tier 3: Operaciones Complejas (Difícilmente Generables)**

Características:
- Requieren optimizaciones no obvias
- Múltiples patrones entrelazados
- Creatividad algorítmica

```python
# Ejemplos difíciles de generar automáticamente
- Flash Attention: tiling sofisticado + numerics cuidadosos
- Matmul optimizado: tiling 2D + swizzling + prefetching
- Sparse operations: indexación indirecta compleja
- Custom kernels con heurísticas específicas del dominio
```

#### El Gap entre Correcto y Óptimo

```{admonition} Validez vs Performance
:class: warning
**Kernel Correcto:**
- Produce output matemáticamente correcto
- No crashes
- Puede ser 10x más lento que optimal

**Kernel Óptimo:**
- Aprovecha coalescing de memoria
- Usa tiling para maximizar reuso
- Minimiza sincronización
- Balances ocupancy vs recursos

**Gap:** La diferencia puede ser 5-10x en performance.
```

**Ejemplo: MatMul Básico vs Óptimo**

```{code-cell} ipython3
:tags: [skip-execution]

# Versión CORRECTA (generable automáticamente)
@triton.jit
def matmul_naive(A_ptr, B_ptr, C_ptr, M, N, K):
    # Versión ingenua: cada thread calcula un elemento de C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = 0.0
    for k in range(K):
        a = tl.load(A_ptr + pid_m * K + k)
        b = tl.load(B_ptr + k * N + pid_n)
        acc += a * b

    tl.store(C_ptr + pid_m * N + pid_n, acc)

# Performance: ~100 GFLOPS en A100

# Versión ÓPTIMA (requiere expertise humano)
@triton.jit
def matmul_optimized(A_ptr, B_ptr, C_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    # Tiling 2D con prefetching
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Bloques de A y B en SRAM
    A_block = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    B_block = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    C_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop con tiling
    for k_tile in range(0, K, BLOCK_K):
        # Load tiles con coalescing
        # ... (código complejo de indexación)

        # Multiply-accumulate
        C_block += tl.dot(A_block, B_block)

    # Store resultado
    # ... (con manejo de boundaries)

# Performance: ~15 TFLOPS en A100 (150x más rápido)
```

**Gap:** 150x diferencia entre versión correcta y óptima.

#### ¿Qué NO Es Generable Automáticamente (Aún)?

**1. Algoritmos Novedosos**

Ejemplo: Flash Attention (Dao et al., 2022)
- Requiere insight matemático: reformulación del cálculo de attention
- Tiling específico para maximizar reuso
- Numerics cuidadosos para evitar overflow
- No es derivable de templates existentes

**2. Optimizaciones Específicas del Hardware**

Ejemplo: Kernel para Tensor Cores
- Requiere conocimiento de instrucciones específicas (wmma)
- Layout de datos específico (row-major vs column-major)
- Alineación de memoria crítica

**3. Heurísticas del Dominio**

Ejemplo: Sparse MatMul
- Depende de patrón de sparsity (random, structured, block)
- Heurísticas para balancear carga entre threads
- Estrategias diferentes por densidad

```{admonition} El Límite Actual
:class: note
**Estado del Arte (2026):**
- LLMs pueden generar ~90% de kernels Tier 1 (elementwise)
- ~60% de Tier 2 (reducciones básicas)
- ~20% de Tier 3 (operaciones complejas)

**No generables aún:**
- Optimizaciones que requieren "creatividad algorítmica"
- Kernels que combinan múltiples técnicas avanzadas
- Adaptación a características específicas del hardware (más allá de CUDA)
```

#### Futuro de la Generación Automática

**Dirección 1: Corpus Más Rico**

- Más ejemplos de kernels optimizados → mejores templates
- Patrones emergentes de la comunidad (GitHub, papers)
- Transfer learning desde otros frameworks (TVM, MLIR)

**Dirección 2: Búsqueda Guiada**

- Evolutionary algorithms para explorar variaciones
- Reinforcement learning con reward = performance
- Neural architecture search aplicado a kernels

**Dirección 3: Verificación Formal**

- Probar correctitud matemáticamente
- Verificar ausencia de race conditions
- Garantizar bounds de memoria

---

### Presentaciones Finales

#### Formato de Presentación de Equipo

**Duración:** 10 minutos por equipo
- 8 min presentación
- 2 min Q&A

**Estructura Recomendada:**

```{admonition} Template de Presentación
:class: tip
**Slide 1: Título (30 seg)**
- Nombre del equipo
- Título del proyecto
- Integrantes

**Slides 2-3: Arquitectura del Agente (2 min)**
- Diagrama de componentes
- Flow de generación (prompt → código → validación → feedback)
- Decisiones clave de diseño

**Slides 4-5: Demo en Vivo (3 min)**
- Seleccionar 1-2 problemas representativos
- Mostrar: input (description) → output (kernel code)
- Si es posible: ejecutar y mostrar output correcto

**Slide 6: Resultados Cuantitativos (2 min)**
- Tabla de pass rates por nivel (L1, L2, L3)
- Comparación con baseline (LLM sin especialización)
- Speedups si aplicable

**Slide 7: Lecciones Aprendidas (1 min)**
- ¿Qué funcionó bien?
- ¿Qué no funcionó y por qué?
- ¿Qué harías diferente?

**Slide 8: Preguntas (2 min)**
- Q&A con audiencia
```

#### Ejemplo de Arquitectura (Slide Visual)

```{code-cell} ipython3
:tags: [skip-execution]

# Diagrama ASCII para presentación
"""
┌──────────────────────────────────────────────────────────┐
│                   KERNEL GENERATION AGENT                 │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │    1. Prompt Engineering         │
        │  - Few-shot examples             │
        │  - Grammar constraints           │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │    2. LLM Generation             │
        │  - Model: CodeLlama-7B           │
        │  - Temperature: 0.2              │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │    3. Validation                 │
        │  - Syntax check                  │
        │  - Execution test                │
        └────────────┬─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────┐
        │    4. Feedback Loop              │
        │  - Error analysis                │
        │  - Retry with corrections        │
        │  - Max 3 attempts                │
        └──────────────────────────────────┘
                     │
                     ▼
              Kernel Code ✓
"""
```

#### Ejemplo de Resultados (Slide con Tabla)

```markdown
## Resultados: Pass Rates en KernelBench

| Nivel | Target | Nuestro Agente | Baseline (GPT-4) |
|-------|--------|----------------|------------------|
| L1    | >90%   | 92.3%          | 68.4%           |
| L2    | >60%   | 64.7%          | 32.1%           |
| L3    | >20%   | 23.5%          | 8.9%            |

**Conclusión:** El agente supera significativamente baseline sin especialización.
```

#### Consejos para Demo en Vivo

```{admonition} Best Practices para Demos
:class: note
**DO:**
- Preparar ejemplos que funcionan (no improvisar)
- Tener plan B si algo falla (video grabado)
- Seleccionar problemas representativos (1 fácil, 1 medio)
- Mostrar código generado (syntax highlighting)

**DON'T:**
- Intentar problemas muy complejos (alto riesgo de fallo)
- Depender de conexión de red en vivo
- Mostrar todo el proceso (solo highlights)
- Leer código línea por línea (aburrido)
```

---

## Actividad de Trabajo — Bloque 2 (25 minutos)

**Ejercicio 1: Preparar y presentar (20 min)**

Usando el template de esta sesión (8 slides), prepara tu presentación de equipo:

1. **Slides** (5 min): completa las secciones del template priorizando:
   - Slide 3: Tu arquitectura (diagrama del agente con S6–S8 integrados)
   - Slide 5: Resultados en KernelBench — la tabla del Ejercicio 1 de hoy
   - Slide 6: El error más interesante que encontraste y cómo lo resolviste
2. **Demo en vivo** (5 min): prepara 2 kernels que funcionen y 1 fallo interesante
   con diagnóstico del `classify_error`.
3. **Presentación** (10 min): 8 min + 2 min Q&A.

**Ejercicio 2: Peer review estructurado (5 min)**

Mientras cada equipo presenta, completa esta tabla (una fila por equipo):

| Equipo | Técnica más efectiva | Resultado sorprendente | Pregunta para ellos |
|--------|---------------------|----------------------|---------------------|
| ... | | | |

---

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: Tu agente logró >90% en L1 pero ~50% en L2. De los tres cambios
posibles — (a) añadir 2 few-shot de reducciones al system prompt, (b) aumentar
`max_attempts` de 5 a 10, o (c) extender `enhanced_classify_error` con un
detector de `RACE_CONDITION` específico para L2 — ¿cuál predices que tendrá
el mayor impacto? Justifica basándote en los errores que clasificaste hoy.
<details><summary>Respuesta</summary>
Depende de la distribución de fallos: si domina <strong>logic errors</strong>
(semántica incorrecta de reducción), la opción (a) — más few-shot — ayuda más
porque el LLM no tiene suficientes ejemplos del patrón. Si domina
<strong>race conditions</strong>, la opción (c) es mejor porque el feedback
dirigido cambia qué ve el LLM en el reintento. Aumentar solo max_attempts (b)
sin mejorar el feedback produce los mismos errores repetidos.
</details>

**Pregunta 2**: El `FinalEvaluator` corre con `max_attempts=3`. Si reejecutas
el mismo agente mañana sobre los mismos 9 problemas con `temperature=0.3`,
¿obtendrás exactamente el mismo FINAL_REPORT.md?
<details><summary>Respuesta</summary>
Probablemente <strong>no</strong>: con temperatura > 0, el LLM produce
diferentes tokens en cada llamada. Algunos problemas que pasaron hoy pueden
fallar mañana y viceversa. Para resultados reproducibles: usar
<code>temperature=0.0</code> (greedy decoding) + fijar la seed de PyTorch.
Esto importa para comparar versiones del agente: el benchmark debe ser
determinístico o promediar múltiples corridas.
</details>
```

### Para Pensar

> *El `matmul_naive` correcto pero 150x más lento que el óptimo. En los
> próximos 3 años, ¿crees que los LLMs cerrarán ese gap automáticamente?
> ¿Qué debería cambiar: el corpus de entrenamiento, la estrategia de agente,
> el reward signal, o el hardware? Defiende con un argumento concreto.*

---

## Entrega del Reporte Final

### Checklist de Entrega

```{admonition} Checklist Completo
:class: important
**Código (40% de la nota):**
- [ ] Agente funcional (puede generar kernels end-to-end)
- [ ] Tests passing (>80% coverage si aplicable)
- [ ] README.md con:
  - [ ] Descripción del proyecto
  - [ ] Instrucciones de instalación
  - [ ] Quick start con ejemplo
  - [ ] Arquitectura del sistema
- [ ] requirements.txt o environment.yml
- [ ] Código bien documentado (docstrings en funciones clave)

**Experimentos (30% de la nota):**
- [ ] Resultados en KernelBench (L1, L2, L3)
- [ ] Baseline comparativo (LLM sin especialización)
- [ ] Análisis estadístico (significancia si aplicable)
- [ ] Configuraciones documentadas (configs/*.yaml)
- [ ] Scripts de reproducción (reproduce_experiments.sh)
- [ ] Resultados guardados (results/*.json)

**Reporte (30% de la nota):**
- [ ] Metodología clara (arquitectura del agente, prompts, feedback)
- [ ] Resultados con visualizaciones (tablas, gráficos)
- [ ] Análisis de errores (taxonomía aplicada: sintaxis, semántica, runtime)
- [ ] Discusión (limitaciones, lecciones aprendidas)
- [ ] Conclusiones y trabajo futuro
- [ ] Referencias completas (papers, repos, documentación)
```

### Estructura del Reporte Final

```markdown
# Grammar-Constrained GPU Kernel Generation
## Project 2 - Final Report

### Abstract
[150-200 palabras: problema, enfoque, resultados principales]

### 1. Introducción
- Motivación: ¿por qué generación automática de kernels?
- Objetivos del proyecto
- Contribución: qué aporta nuestro agente

### 2. Background
- GPU Architecture (breve resumen)
- Triton Programming Model
- Grammar-Constrained Generation (P1 S6-S7, aplicado en P2)
- KernelBench (framework de evaluación)

### 3. Metodología
#### 3.1 Arquitectura del Agente
- Componentes: prompt engineering, LLM, validación, feedback
- Decisiones de diseño
- Diagrama de flujo

#### 3.2 Prompts y Few-Shot Examples
- Estructura de prompts
- Ejemplos incluidos
- Restricciones gramaticales

#### 3.3 Feedback Loop
- Tipos de errores detectados
- Estrategia de corrección
- Criterios de parada

### 4. Implementación
- Modelo LLM usado (ej. CodeLlama-7B)
- Framework (Triton, PyTorch)
- Optimizaciones aplicadas
- Desafíos técnicos

### 5. Experimentos
#### 5.1 Setup
- Hardware (GPU, CUDA version)
- Software (versiones de librerías)
- Configuración del agente

#### 5.2 Baselines
- LLM sin especialización (GPT-4, Claude, etc.)
- Templates simples

#### 5.3 Resultados
- Pass rates por nivel (L1, L2, L3)
- Tiempo de generación
- Comparación con baselines

### 6. Análisis
#### 6.1 Performance por Tipo de Kernel
- Elementwise: 92% pass rate
- Reducciones: 65% pass rate
- Complejos: 24% pass rate

#### 6.2 Análisis de Errores
- Taxonomía: sintaxis, semántica, runtime
- Errores más comunes
- Casos de éxito y fracaso

#### 6.3 Ablations
- ¿Qué pasa sin few-shot examples?
- ¿Qué pasa sin feedback loop?
- ¿Qué pasa sin restricciones gramaticales?

### 7. Discusión
- Hallazgos principales
- Limitaciones del enfoque
- Comparación con estado del arte
- Lecciones aprendidas

### 8. Trabajo Futuro
- Mejoras al agente (más ejemplos, mejor feedback)
- Extensión a otros backends (SYCL, Metal)
- Integración con PyTorch (torch.compile)
- Verificación formal

### 9. Conclusión
[Resumen de contribuciones y resultados]

### Referencias
[Papers, repos, documentación]

### Apéndices
- A: Especificación completa de la gramática
- B: Ejemplos de prompts
- C: Resultados completos por problema
- D: Código de kernels generados (selección)
```

---

## Progresión Completa del Proyecto

```{admonition} Resumen Visual: 10 Semanas de Progreso
:class: important
**S1-S2: Fundamentos GPU + Kernels Básicos**
```
GPU Architecture → CUDA/PyTorch → Primer Triton kernel
Output: Entender SIMT, memoria, escribir kernel simple
```

**S3-S4: Optimización + Benchmarking**
```
Tiling, coalescing → Autotuning → Roofline analysis
Output: Kernel optimizado con 5-10x speedup
```

**S5: Pipeline de Evaluación**
```
KernelBench integration → Test correctness → Measure performance
Output: Framework para evaluar kernels automáticamente
```

**S6: Primer Agente (Prompting)**
```
Few-shot prompts → LLM generation → Basic validation
Output: ~50% pass rate en L1, ~10% en L2
```

**S7: Agente con Loop**
```
Feedback on errors → Retry logic → Grammar constraints
Output: ~80% pass rate en L1, ~40% en L2
```

**S8: Agente con Feedback Avanzado**
```
Error taxonomy → Targeted corrections → Ablations
Output: ~90% pass rate en L1, ~60% en L2, ~20% en L3
```

**S9: Extensión Conceptual (SYCL)**
```
Portability → oneAPI → Multi-backend thinking
Output: Entender cómo abstracciones generalizan
```

**S10: Evaluación Final + Demos**
```
Full KernelBench eval → Presentations → Final report
Output: Sistema completo documentado y evaluado
```
```

### Integración con el Ecosistema PyTorch

Tu agente no existe en aislamiento. Para ser útil en producción, debe integrarse con herramientas existentes:

```{code-cell} ipython3
:tags: [skip-execution]

# Integración con torch.compile
import torch
import triton

class CustomOp(torch.autograd.Function):
    """Operación custom con kernel Triton generado."""

    @staticmethod
    def forward(ctx, x, kernel_code):
        # Compilar kernel generado
        kernel = triton.compile(kernel_code)

        # Ejecutar
        output = kernel(x)

        # Guardar para backward
        ctx.save_for_backward(x)
        ctx.kernel_code = kernel_code

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Generar kernel backward (si aplicable)
        x, = ctx.saved_tensors
        # ... implementar backward
        return grad_x, None

# Uso con torch.compile
@torch.compile
def my_model(x):
    # torch.compile detecta CustomOp y puede fusionar
    return CustomOp.apply(x, generated_kernel_code)

# Beneficio: Fusión automática con otras operaciones
```

**Ventajas de Integración:**
1. **Fusión automática**: torch.compile fusiona kernels custom con operaciones nativas
2. **Autograd support**: Backward pass automático si defines gradiente
3. **Producción-ready**: Compatible con torch.jit, ONNX export

---

## Resumen Final del Módulo

```{admonition} Lo que Lograste en 10 Semanas
:class: tip
**Conocimientos Técnicos:**
- ✓ GPU Architecture: SIMT, warps, jerarquía de memoria, coalescing
- ✓ CUDA/Triton: Escribir kernels desde cero, optimizarlos
- ✓ Benchmarking: Medir performance correctamente (baselining, estadística)
- ✓ Evaluación: KernelBench, correctitud vs performance
- ✓ Gramáticas: Restringir generación para validez estructural
- ✓ LLMs para Código: Prompting, few-shot, feedback loops

**Habilidades Prácticas:**
- ✓ Debugging de kernels GPU (errores sintácticos, race conditions, OOB)
- ✓ Profiling (nsys, ncu, roofline analysis)
- ✓ Documentación técnica (README, API docs, reproducibilidad)
- ✓ Experimentación rigurosa (baselines, ablations, significancia)

**Proyecto Completo:**
- ✓ Agente funcional que genera kernels GPU automáticamente
- ✓ Evaluado en benchmark estándar (KernelBench L1-L3)
- ✓ Documentado y reproducible
- ✓ Presentado profesionalmente

**Impacto:**
Has demostrado que es posible **automatizar la generación de kernels GPU** para operaciones comunes, reduciendo significativamente el esfuerzo manual. Esto abre puertas a:
- Democratización de GPU programming
- Aceleración de prototipado en ML
- Portabilidad entre backends (CUDA, SYCL, Metal)
```

```{admonition} Próximos Pasos: Más Allá del Curso
:class: note
**Para Profundizar:**
1. **Implementa más kernels complejos**: Flash Attention, Sparse MatMul
2. **Contribuye a proyectos open source**: Triton, PyTorch, TVM
3. **Explora verificación formal**: Probar correctitud matemáticamente
4. **Experimenta con nuevos backends**: SYCL, Metal, Vulkan
5. **Investiga optimización automática**: AutoTVM, Neural Architecture Search

**Para Aplicar:**
1. **Integra en tus proyectos de ML**: Custom layers con kernels Triton
2. **Benchmarking de modelos**: Identifica cuellos de botella, optimiza
3. **Enseña a otros**: Comparte conocimiento (blogs, tutoriales, talks)

**Para Investigar:**
1. **Generación guiada por performance**: RL donde reward = speedup
2. **Transfer learning entre backends**: CUDA → SYCL automático
3. **Verificación de correctitud**: Formal methods para kernels
```

---

## Referencias

**Papers Fundamentales:**
- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.
- Tillet, P., et al. (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL.
- Chen, T., et al. (2018). TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. OSDI.

**Herramientas y Frameworks:**
- Triton. [Triton Language and Compiler](https://github.com/triton-lang/triton). GitHub.
- PyTorch. [PyTorch Documentation](https://pytorch.org/docs/). PyTorch.
- CUDA. [CUDA Programming Guide](https://docs.nvidia.com/cuda/). NVIDIA.
- KernelBench. [GPU Kernel Benchmarking Suite](https://github.com/kernel-bench/kernelbench). GitHub.

**Grammar-Constrained Generation:**
- XGrammar. [Efficient, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar). GitHub.
- NVIDIA. [Outlines: Structured Text Generation](https://github.com/outlines-dev/outlines). GitHub.

**Benchmarking y Profiling:**
- NVIDIA. [Nsight Systems](https://developer.nvidia.com/nsight-systems). NVIDIA Developer.
- NVIDIA. [Nsight Compute](https://developer.nvidia.com/nsight-compute). NVIDIA Developer.

**Portabilidad:**
- Intel. [oneAPI and SYCL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html). Intel.
- Khronos Group. [SYCL Specification](https://www.khronos.org/sycl/). Khronos.

---

*Esta lectura concluye el Proyecto 2: Grammar-Constrained GPU Kernel Generation. Has completado un viaje desde fundamentos de GPU hasta generación automática de kernels con LLMs. ¡Felicitaciones!*

---

## Trabajo Final: Reporte y Código

**Fecha de entrega:** [Definida por instructor]

**Entregables:**

1. **Código completo** en repositorio Git:
   - `src/`: Código del agente
   - `tests/`: Tests unitarios
   - `configs/`: Configuraciones de experimentos
   - `scripts/`: Scripts de reproducción
   - `README.md`: Documentación principal

2. **Resultados** en `results/`:
   - `final_results.json`: Resultados de KernelBench
   - `figures/`: Visualizaciones
   - `FINAL_REPORT.md`: Reporte técnico completo

3. **Presentación**:
   - `presentation.pdf`: Slides de la presentación
   - Opcional: `demo_video.mp4` si el demo fue grabado

**Formato de entrega:** Enlace a repositorio GitHub/GitLab con README indicando dónde encontrar cada componente.

---

```{admonition} Mensaje Final
:class: tip
Has recorrido un camino desafiante: desde entender cómo funciona un GPU hasta construir un sistema que genera kernels automáticamente. Este conocimiento es valioso no solo para GPU programming, sino para cualquier área de sistemas de alto rendimiento.

**Recuerda:**
- La generación automática no reemplaza expertise, la democratiza
- Los mejores sistemas combinan automatización con revisión humana
- La documentación y reproducibilidad son tan importantes como el código

**¡Éxito en tus proyectos futuros!**
```
