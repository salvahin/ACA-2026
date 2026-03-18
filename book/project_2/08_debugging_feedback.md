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

# Debugging como Feedback del Agente

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
    !pip install -q torch triton datasets
```

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/project_2/08_debugging_feedback.ipynb)
```

> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 8
> **Tiempo de lectura:** ~45 minutos
> **Formato:** 25' clase + 25' trabajo + 10' discusión + 25' clase + 25' trabajo + 10' cierre

---

## Introducción

En la sesión anterior construimos un agente LLM que genera kernels GPU. Pero como vimos, muchos kernels fallan en compilación, runtime, o correctitud. En lugar de tratar el debugging como una habilidad separada, esta sesión lo refunda como **feedback estructurado para el agente**.

La idea clave: cada tipo de error le dice al agente QUÉ específicamente debe corregir en su siguiente intento.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta sesión podrás:
- Clasificar errores GPU en taxonomía de 4 categorías para el agente
- Implementar `classify_error()` que extrae feedback estructurado
- Integrar clasificación de errores en el loop del agente (S7)
- Usar test suites como verificadores automáticos
- Mejorar pass rate del agente con error-guided feedback (~60% L2)
```

---

## Bloque 1: Taxonomía de Errores GPU para el Agente

### Taxonomía de 4 Categorías

Los errores GPU caen en 4 categorías que el agente debe distinguir:

```
1. COMPILACIÓN
   - Sintaxis Python/Triton malformada
   - Tipos incompatibles (e.g., tl.dot(float32, int32))
   - constexpr incorrecto (BLOCK = variable_runtime)
   - API mal usada (tl.load sin pointer válido)

2. RUNTIME
   - Out of bounds (acceso fuera de límites)
   - Race conditions (resultados no determinísticos)
   - División por zero (inf/nan silencioso)
   - Deadlock (raro en Triton)

3. CORRECTITUD
   - Lógica incorrecta (algoritmo no implementa spec)
   - Off-by-one (offsets < n vs offsets <= n)
   - Broadcasting incorrecto (shapes mal alineados)
   - Reducción incompleta (no procesar todos los elementos)

4. PERFORMANCE
   - Memory coalescing violado (accesos no coalescidos)
   - Bank conflicts (shared memory)
   - Occupancy bajo (demasiados registros)
   - Trabajo redundante (recalcular innecesariamente)
```

```{code-cell} ipython3
from dataclasses import dataclass
from typing import Optional

@dataclass
class ErrorFeedback:
    """Feedback estructurado para el agente."""
    category: str  # COMPILATION, RUNTIME, CORRECTNESS, PERFORMANCE
    subcategory: str  # e.g., "OUT_OF_BOUNDS", "SYNTAX"
    message: str  # Mensaje de error original
    suggestion: str  # Sugerencia específica para el agente
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None

    def to_prompt(self) -> str:
        """Convierte feedback a texto para el LLM."""
        prompt = f"""ERROR DETECTED:
Category: {self.category}
Type: {self.subcategory}

{self.message}

SUGGESTION FOR NEXT ATTEMPT:
{self.suggestion}"""

        if self.code_snippet:
            prompt += f"\n\nPROBLEMATIC CODE:\n{self.code_snippet}"

        return prompt
```

### Top-5 Errores Más Comunes

Estos 5 errores cubren ~80% de fallos en kernels generados:

```{code-cell} ipython3
TOP_5_ERRORS = {
    "1_NO_MASK": {
        "symptoms": "Crash en runtime, resultados incorrectos al final del array",
        "cause": "tl.load/tl.store sin máscara cuando n no divide BLOCK_SIZE",
        "example": """
# MALO
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets)  # ¿Qué si offsets >= n?
""",
        "fix": """
# BUENO
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
""",
        "agent_hint": "ALWAYS add mask=offsets<n to tl.load and tl.store"
    },

    "2_WRONG_INDEXING_2D": {
        "symptoms": "Resultados transpuestos o permutados",
        "cause": "Confundir row-major (PyTorch) con column-major",
        "example": """
# MALO: column-major
idx = h + w * height

# BUENO: row-major (PyTorch)
idx = h * width + w
""",
        "agent_hint": "PyTorch uses row-major: idx = row*width + col"
    },

    "3_RACE_CONDITION": {
        "symptoms": "Resultados no determinísticos entre runs",
        "cause": "Múltiples threads escriben al mismo lugar sin atomic",
        "example": """
# MALO
tl.store(result_ptr, suma)  # Todos los bloques escriben a result_ptr[0]

# BUENO
tl.atomic_add(result_ptr, suma)
""",
        "agent_hint": "Use tl.atomic_add when multiple blocks write to same location"
    },

    "4_WRONG_DTYPE": {
        "symptoms": "Overflow, underflow, pérdida de precisión",
        "cause": "Tipo de datos incorrecto para la operación",
        "example": """
# MALO: int32 overflow en suma
x = tl.load(x_ptr + offsets).to(tl.int32)
suma = tl.sum(x)  # Puede overflow

# BUENO: usar float32 o int64
x = tl.load(x_ptr + offsets).to(tl.float32)
""",
        "agent_hint": "Use float32 for reductions, int64 for indices"
    },

    "5_WARP_DIVERGENCE": {
        "symptoms": "Kernel mucho más lento de lo esperado",
        "cause": "Branches que causan divergencia dentro del warp",
        "example": """
# MALO: divergencia
if tid % 2 == 0:
    y = x * 2
else:
    y = x * 3

# BUENO: usar tl.where
y = tl.where(tid % 2 == 0, x * 2, x * 3)
""",
        "agent_hint": "Use tl.where instead of if/else for conditional operations"
    }
}

def print_top5_guide():
    """Guía rápida de los top 5 errores."""
    print("=" * 70)
    print("TOP-5 ERRORES GPU PARA EL AGENTE")
    print("=" * 70)

    for error_id, info in TOP_5_ERRORS.items():
        print(f"\n{error_id}:")
        print(f"  Síntoma: {info['symptoms']}")
        print(f"  Causa: {info['cause']}")
        print(f"  Hint para agente: {info['agent_hint']}")

    print("\n" + "=" * 70)

print_top5_guide()
```

### El Agente Usa la Clasificación

```{code-cell} ipython3
def classify_error(code: str, error: Exception) -> ErrorFeedback:
    """
    Clasifica error y genera feedback estructurado para el LLM.

    Esta función es el CORAZÓN del debugging como feedback.
    Toma un error genérico y lo convierte en instrucciones específicas.
    """
    error_str = str(error)
    error_type = type(error).__name__

    # 1. ERRORES DE COMPILACIÓN
    if isinstance(error, SyntaxError):
        return ErrorFeedback(
            category="COMPILATION",
            subcategory="SYNTAX",
            message=error_str,
            suggestion="Fix Python syntax: check parentheses, indentation, and decorators. Ensure @triton.jit decorator is present.",
            line_number=getattr(error, 'lineno', None)
        )

    if isinstance(error, TypeError):
        return ErrorFeedback(
            category="COMPILATION",
            subcategory="TYPE_MISMATCH",
            message=error_str,
            suggestion="Check type compatibility in operations. Common: tl.dot requires matching types (both float32 or both float16).",
        )

    if "constexpr" in error_str.lower():
        return ErrorFeedback(
            category="COMPILATION",
            subcategory="CONSTEXPR",
            message=error_str,
            suggestion="Parameters like BLOCK_SIZE must be compile-time constants (tl.constexpr). Don't use runtime variables.",
        )

    # 2. ERRORES DE RUNTIME
    if "out of bounds" in error_str.lower() or "invalid memory" in error_str.lower():
        # Detectar si falta máscara
        has_mask = "mask=" in code
        suggestion = "Add mask to ALL tl.load and tl.store operations:\n"
        suggestion += "  mask = offsets < n\n"
        suggestion += "  x = tl.load(x_ptr + offsets, mask=mask, other=0.0)\n"
        suggestion += "  tl.store(y_ptr + offsets, result, mask=mask)"

        return ErrorFeedback(
            category="RUNTIME",
            subcategory="OUT_OF_BOUNDS",
            message=error_str,
            suggestion=suggestion,
            code_snippet=_extract_load_store_lines(code)
        )

    if "race" in error_str.lower() or "non-deterministic" in error_str.lower():
        return ErrorFeedback(
            category="RUNTIME",
            subcategory="RACE_CONDITION",
            message=error_str,
            suggestion="Use tl.atomic_add when multiple blocks write to the same memory location. Replace:\n  tl.store(ptr, value)\nwith:\n  tl.atomic_add(ptr, value)",
        )

    # 3. ERRORES DE CORRECTITUD
    if "nan" in error_str.lower():
        return ErrorFeedback(
            category="CORRECTNESS",
            subcategory="NAN_RESULT",
            message=error_str,
            suggestion="Check for: division by zero, log of negative numbers, or overflow. Add numerical stability (e.g., subtract max before exp in softmax).",
        )

    if "incorrect result" in error_str.lower() or "mismatch" in error_str.lower():
        return ErrorFeedback(
            category="CORRECTNESS",
            subcategory="LOGIC_ERROR",
            message=error_str,
            suggestion="Verify algorithm logic:\n  1. Check indexing (row-major: row*width+col)\n  2. Verify reduction is complete\n  3. Check boundary conditions (< vs <=)\n  4. Ensure all elements are processed",
        )

    # 4. ERRORES DE PERFORMANCE (detectados por análisis)
    if "slow" in error_str.lower() or "performance" in error_str.lower():
        return ErrorFeedback(
            category="PERFORMANCE",
            subcategory="SLOW_KERNEL",
            message=error_str,
            suggestion="Optimize memory access:\n  1. Ensure coalesced access (contiguous offsets)\n  2. Avoid bank conflicts (stride != 32)\n  3. Use appropriate BLOCK_SIZE (64, 128, 256)\n  4. Check if compute-bound or memory-bound",
        )

    # Default: error no clasificado
    return ErrorFeedback(
        category="UNKNOWN",
        subcategory="UNCLASSIFIED",
        message=error_str,
        suggestion="Review the error message and kernel code carefully. Common issues: missing mask, wrong indexing, type mismatches.",
    )

def _extract_load_store_lines(code: str) -> str:
    """Extrae líneas con tl.load/tl.store para debugging."""
    lines = code.split('\n')
    relevant = []
    for i, line in enumerate(lines, 1):
        if 'tl.load' in line or 'tl.store' in line:
            relevant.append(f"Line {i}: {line.strip()}")
    return '\n'.join(relevant) if relevant else "No load/store found"
```

## Actividad de Trabajo — Bloque 1 (25 minutos)

**Ejercicio 1: Completar `enhanced_classify_error` con 2 detectores nuevos (15 min)**

A partir del esqueleto de `enhanced_classify_error`, implementa dos de los
cuatro detectores propuestos:

1. **Indexación 2D column-major**: Detecta el patrón `_ + _ * height` con regex
   y añade el mensaje corrector en `base_feedback.suggestion`.
2. **Race condition por determinismo**: Si `test_result['determinism_failed']` es
   `True`, clasifica como `RUNTIME / RACE_CONDITION` y sugiere `tl.atomic_add`.

```{code-cell} ipython3
:tags: [skip-execution]

def enhanced_classify_error(code: str, error: Exception, test_result: dict) -> ErrorFeedback:
    base_feedback = classify_error(code, error)

    # TODO 1: Detectar indexación 2D column-major (regex sobre code)
    import re
    if re.search(r'\w+\s*\+\s*\w+\s*\*\s*height', code):
        base_feedback.suggestion += "\nWARNING: column-major detected. Use row-major: idx = row*width + col"

    # TODO 2: Detectar race condition via non-determinism flag
    if test_result.get('determinism_failed'):
        base_feedback.category = "RUNTIME"
        base_feedback.subcategory = "RACE_CONDITION"
        base_feedback.suggestion = "Non-deterministic results. Use tl.atomic_add for global reductions."

    return base_feedback
```

Verifica con los dos test cases:

```{code-cell} ipython3
:tags: [skip-execution]

# Test 1: column-major
code_2d = "idx = h + w * height"
fb = enhanced_classify_error(code_2d, RuntimeError("incorrect result"), {})
assert "row-major" in fb.suggestion.lower(), f"Falta hint row-major: {fb.suggestion}"

# Test 2: race condition
fb2 = enhanced_classify_error("", None, {'determinism_failed': True})
assert fb2.subcategory == "RACE_CONDITION"
assert "atomic" in fb2.suggestion.lower()

print("✓ Ambos detectores funcionan")
```

**Ejercicio 2: Probar el clasificador con un kernel buggy (10 min)**

Escribe un kernel Triton que tenga indexación column-major (introduce el bug
deliberadamente en un transpose 2D). Pasa el código a `enhanced_classify_error`
y verifica que el `suggestion` generado sea suficientemente informativo
como para guiar a un LLM a corregirlo. Imprime el `feedback.to_prompt()` completo.

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

1. **Categorizar vs describir**: `classify_error` devuelve una categoría
   (`RUNTIME/RACE_CONDITION`) y un `suggestion` textual. ¿Cuál de los dos le
   sirve más al LLM para corregir el código? ¿Por qué tener ambos?
2. **Falsos positivos**: La detección de column-major usa una regex. Describe
   un kernel 2D válido que dispare la regex incorrectamente (falso positivo).
   ¿Cómo refinar la detección para evitarlo?
3. **Lazó entre Bloque 1 y S7**: En S7 el agente tomaba el `error_message`
   crudo de Python y lo pasaba directo al LLM. ¿Qué aporta el paso de
   clasificación del Bloque 1? ¿En qué caso el error crudo sería igual o mejor?
```

---

## Bloque 2: Feedback Estructurado en el Agente

### Integración con el Agente (de S7)

Ahora conectamos el clasificador de errores con el agente de generación:

```{code-cell} ipython3
:tags: [skip-execution]

import triton
import triton.language as tl
from typing import Callable, Tuple

class ErrorGuidedKernelAgent:
    """
    Agente que usa feedback de errores para mejorar generación.
    Versión extendida del KernelAgent de S7.
    """

    def __init__(self, llm_generate_fn: Callable, max_attempts: int = 5):
        self.llm_generate = llm_generate_fn
        self.max_attempts = max_attempts
        self.error_history = []  # Log de errores para análisis

    def generate_kernel(self, task_spec: str, reference_fn: Callable) -> Tuple[str, dict]:
        """
        Genera kernel con error-guided feedback loop.

        Returns:
            (kernel_code, metadata) donde metadata incluye:
            - success: bool
            - attempts: int
            - errors: list[ErrorFeedback]
            - final_test_results: dict
        """

        prompt = self._build_initial_prompt(task_spec)
        metadata = {"success": False, "attempts": 0, "errors": []}

        for attempt in range(self.max_attempts):
            metadata["attempts"] = attempt + 1

            # Generar código
            kernel_code = self.llm_generate(prompt)

            # Compilar y testear
            try:
                kernel_fn = self._compile_kernel(kernel_code)
                test_results = self._run_test_suite(kernel_fn, reference_fn)

                if test_results["all_passed"]:
                    metadata["success"] = True
                    metadata["final_test_results"] = test_results
                    return kernel_code, metadata

                # Tests fallaron: clasificar error de correctitud
                error = Exception(f"Test failures: {test_results['failures']}")
                feedback = classify_error(kernel_code, error)

            except Exception as e:
                # Error de compilación o runtime
                feedback = classify_error(kernel_code, e)

            # Guardar error en historial
            metadata["errors"].append(feedback)
            self.error_history.append({
                "attempt": attempt + 1,
                "code": kernel_code,
                "feedback": feedback
            })

            # Actualizar prompt con feedback
            prompt = self._build_feedback_prompt(task_spec, kernel_code, feedback)

        # Max attempts alcanzado sin éxito
        return kernel_code, metadata

    def _build_initial_prompt(self, task_spec: str) -> str:
        """Prompt inicial con task spec."""
        return f"""Generate a Triton kernel for the following task:

{task_spec}

Requirements:
- Use @triton.jit decorator
- Add mask to ALL tl.load and tl.store operations
- Use row-major indexing (idx = row*width + col)
- Include proper type conversions
- Return complete, executable code

Generate the kernel:"""

    def _build_feedback_prompt(self, task_spec: str, failed_code: str,
                               feedback: ErrorFeedback) -> str:
        """Prompt con feedback del error anterior."""
        return f"""The previous kernel attempt FAILED with this error:

{feedback.to_prompt()}

PREVIOUS ATTEMPT:
```python
{failed_code}
```

Please generate a CORRECTED version that fixes the error.

Original task:
{task_spec}

Generate the corrected kernel:"""

    def _compile_kernel(self, code: str):
        """Compila el kernel generado."""
        # Ejecutar código para definir la función
        namespace = {"triton": triton, "tl": tl}
        exec(code, namespace)

        # Encontrar la función decorada con @triton.jit
        for name, obj in namespace.items():
            if callable(obj) and hasattr(obj, '__name__') and not name.startswith('_'):
                return obj

        raise ValueError("No triton.jit function found in generated code")

    def _run_test_suite(self, kernel_fn, reference_fn) -> dict:
        """Ejecuta suite completa de tests."""
        suite = KernelTestSuite(kernel_fn, reference_fn)
        return suite.run_all()

    def analyze_error_patterns(self) -> dict:
        """Analiza patrones en el historial de errores."""
        categories = {}
        for entry in self.error_history:
            cat = entry["feedback"].category
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_category": categories,
            "most_common": max(categories.items(), key=lambda x: x[1]) if categories else None
        }
```

### Test Suites como Verificadores del Agente

```{code-cell} ipython3
:tags: [skip-execution]

class KernelTestSuite:
    """
    Suite de tests que sirve como verificador automático para el agente.
    Adaptado de S7 (debugging taxonomía).
    """

    def __init__(self, kernel_fn, reference_fn):
        self.kernel = kernel_fn
        self.reference = reference_fn
        self.results = []

    def run_all(self) -> dict:
        """Ejecuta todos los tests y retorna resultados."""
        self.test_compilation()
        self.test_boundary_cases()
        self.test_determinism()
        self.test_correctness()

        return self.summarize()

    def test_compilation(self):
        """Verifica que el kernel compila y ejecuta."""
        try:
            x = torch.randn(128, device='cuda')
            _ = self._run_kernel(x)
            self.results.append(("compilation", True, None))
        except Exception as e:
            self.results.append(("compilation", False, str(e)))

    def test_boundary_cases(self):
        """Prueba casos límite importantes."""
        test_sizes = [1, 31, 32, 33, 255, 256, 257]

        for size in test_sizes:
            try:
                x = torch.randn(size, device='cuda')
                output = self._run_kernel(x)
                expected = self.reference(x)

                passed = torch.allclose(output, expected, rtol=1e-5, atol=1e-6)
                self.results.append((f"boundary_{size}", passed, None))
            except Exception as e:
                self.results.append((f"boundary_{size}", False, str(e)))

    def test_determinism(self, num_runs: int = 10):
        """Detecta race conditions mediante múltiples ejecuciones."""
        try:
            x = torch.randn(1000, device='cuda')
            outputs = [self._run_kernel(x).clone() for _ in range(num_runs)]

            reference_output = outputs[0]
            all_equal = all(torch.equal(out, reference_output) for out in outputs[1:])

            if not all_equal:
                self.results.append(("determinism", False, "Non-deterministic results detected"))
            else:
                self.results.append(("determinism", True, None))
        except Exception as e:
            self.results.append(("determinism", False, str(e)))

    def test_correctness(self):
        """Compara contra referencia PyTorch."""
        test_shapes = [(128,), (1000,), (256, 256)]

        for shape in test_shapes:
            try:
                x = torch.randn(shape, device='cuda')
                output = self._run_kernel(x)
                expected = self.reference(x)

                if output.shape != expected.shape:
                    self.results.append((f"correctness_{shape}", False, f"Shape mismatch: {output.shape} vs {expected.shape}"))
                    continue

                passed = torch.allclose(output, expected, rtol=1e-4, atol=1e-5)
                if not passed:
                    max_diff = (output - expected).abs().max().item()
                    self.results.append((f"correctness_{shape}", False, f"Max diff: {max_diff:.2e}"))
                else:
                    self.results.append((f"correctness_{shape}", True, None))
            except Exception as e:
                self.results.append((f"correctness_{shape}", False, str(e)))

    def _run_kernel(self, x):
        """Wrapper para ejecutar kernel con configuración estándar."""
        n = x.numel()
        output = torch.empty_like(x)

        # Configuración típica
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        self.kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output

    def summarize(self) -> dict:
        """Resume resultados de tests."""
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        failures = [(name, error) for name, p, error in self.results if not p]

        return {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0,
            "all_passed": len(failures) == 0,
            "failures": failures,
            "determinism_failed": any("determinism" in name and not p for name, p, _ in self.results)
        }

# Tests específicos como funciones standalone
def boundary_test(kernel_fn, reference_fn, sizes=[1, 31, 32, 33, 255, 256, 257]):
    """Test de casos límite."""
    print(f"Testing boundary cases: {sizes}")
    for size in sizes:
        x = torch.randn(size, device='cuda')
        try:
            output = kernel_fn(x)
            expected = reference_fn(x)
            passed = torch.allclose(output, expected, rtol=1e-5)
            print(f"  n={size:4d}: {'✓' if passed else '✗'}")
        except Exception as e:
            print(f"  n={size:4d}: ✗ ({type(e).__name__})")

def determinism_test(kernel_fn, x, num_runs=10):
    """Test de determinismo (detecta race conditions)."""
    outputs = [kernel_fn(x).clone() for _ in range(num_runs)]
    reference = outputs[0]

    for i, out in enumerate(outputs[1:], 1):
        if not torch.equal(out, reference):
            print(f"✗ Run {i} differs from run 0 - RACE CONDITION DETECTED")
            return False

    print(f"✓ All {num_runs} runs identical")
    return True

def reference_test(kernel_fn, reference_fn, x):
    """Test de correctitud vs referencia."""
    output = kernel_fn(x)
    expected = reference_fn(x)

    if not torch.allclose(output, expected, rtol=1e-5, atol=1e-6):
        diff = (output - expected).abs()
        print(f"✗ Max diff: {diff.max().item():.2e}")
        print(f"  Mean diff: {diff.mean().item():.2e}")

        # Mostrar primeros 5 elementos incorrectos
        wrong_mask = diff > 1e-4
        wrong_indices = torch.where(wrong_mask.flatten())[0][:5]
        for idx in wrong_indices:
            idx = idx.item()
            print(f"  [{idx}]: got {output.flatten()[idx]:.6f}, expected {expected.flatten()[idx]:.6f}")
        return False

    print("✓ Output matches reference")
    return True
```

### Ejemplo: Agente Con y Sin Error Feedback

```{code-cell} ipython3
:tags: [skip-execution]

# Comparación de pass rates

def dummy_llm_without_feedback(prompt: str) -> str:
    """LLM sin feedback - siempre genera el mismo código buggy."""
    return """
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # BUG: sin máscara
    y = x * 2.0
    tl.store(y_ptr + offsets, y)  # BUG: sin máscara
"""

def dummy_llm_with_feedback(prompt: str) -> str:
    """LLM con feedback - aprende del error y corrige."""
    if "Add mask" in prompt:
        # Segunda iteración: aprendió del feedback
        return """
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
"""
    else:
        # Primera iteración: mismo código buggy
        return dummy_llm_without_feedback(prompt)

# Simular generación
task = "Multiply each element by 2"
reference = lambda x: x * 2.0

print("=" * 70)
print("COMPARACIÓN: Con vs Sin Error Feedback")
print("=" * 70)

# Sin feedback
agent_no_feedback = ErrorGuidedKernelAgent(dummy_llm_without_feedback, max_attempts=3)
code1, meta1 = agent_no_feedback.generate_kernel(task, reference)
print(f"\nSIN feedback: success={meta1['success']}, attempts={meta1['attempts']}")
print(f"Errores: {[e.subcategory for e in meta1['errors']]}")

# Con feedback
agent_with_feedback = ErrorGuidedKernelAgent(dummy_llm_with_feedback, max_attempts=3)
code2, meta2 = agent_with_feedback.generate_kernel(task, reference)
print(f"\nCON feedback: success={meta2['success']}, attempts={meta2['attempts']}")
print(f"Errores: {[e.subcategory for e in meta2['errors']]}")

print("\nRESULTADO: El agente CON feedback aprende del primer error y corrige en intento 2")
```

### Error-to-Prompt Mapping

Cada categoría de error mapea a instrucciones específicas:

````{code-cell} ipython3
ERROR_TO_PROMPT_MAP = {
    "OUT_OF_BOUNDS": """
CRITICAL FIX REQUIRED: Out of bounds memory access detected.

SOLUTION:
1. Calculate mask: mask = offsets < n
2. Add mask to EVERY tl.load: tl.load(ptr + offsets, mask=mask, other=0.0)
3. Add mask to EVERY tl.store: tl.store(ptr + offsets, data, mask=mask)

Example:

offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
result = x * 2.0
tl.store(y_ptr + offsets, result, mask=mask)

""",

    "RACE_CONDITION": """
CRITICAL FIX REQUIRED: Race condition detected (non-deterministic results).

SOLUTION:
Replace tl.store with tl.atomic_add when multiple blocks write to the same location.

BAD:

suma = tl.sum(x)
tl.store(output_ptr, suma)  # Race: todos escriben a output_ptr[0]


GOOD:

suma = tl.sum(x)
tl.atomic_add(output_ptr, suma)  # Atómico: seguro

""",

    "WRONG_INDEXING": """
FIX REQUIRED: Incorrect 2D indexing detected.

PyTorch uses ROW-MAJOR layout:
  idx = row * width + col

NOT column-major:
  idx = row + col * height  # WRONG

Example:

h_idx = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
w_idx = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
h_2d = h_idx[:, None]
w_2d = w_idx[None, :]
linear_idx = h_2d * width + w_2d  # row-major ✓

""",

    "TYPE_MISMATCH": """
FIX REQUIRED: Type compatibility issue.

RULES:
1. Use .to(tl.float32) for most operations
2. Use .to(tl.int64) for indices
3. tl.dot requires matching types
4. Reductions should use float32 to avoid overflow

Example:

x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
suma = tl.sum(x)  # float32 evita overflow

"""
}

def get_prompt_for_error(feedback: ErrorFeedback) -> str:
    """Obtiene prompt específico según tipo de error."""
    key = feedback.subcategory
    return ERROR_TO_PROMPT_MAP.get(key, "Review error message and fix accordingly.")
````

### Trabajo Alumnos 2: Mejorar Agente con Error Feedback

```{code-cell} ipython3
:tags: [skip-execution]

# EJERCICIO: Implementar agente mejorado para L2 (~60% pass rate)

def build_l2_agent(llm_api_key: str) -> ErrorGuidedKernelAgent:
    """
    TAREA: Construir agente que alcance ~60% pass rate en KernelBench L2

    Requerimientos:
    1. Integrar classify_error en el loop
    2. Agregar memoria de errores previos (aprender de errores pasados)
    3. Usar test suite completo (boundary + determinism + correctness)
    4. Implementar estrategia de reintentos inteligente:
       - Intento 1: prompt base
       - Intento 2: prompt + feedback de error
       - Intento 3: prompt + feedback + ejemplos de corrección
       - Intento 4+: prompt + feedback + todos los errores previos

    Pistas:
    - Usa una LLM API real (OpenAI, Anthropic, etc)
    - Aumenta max_attempts a 8-10 para L2
    - Agrega ejemplos de kernels correctos en el prompt
    - Usa temperatura baja (0.3) para generación más determinística
    """

    def llm_generate_with_api(prompt: str) -> str:
        # TODO: Implementa tu backend LLM (elige uno):
        #
        # OpenAI:    return openai.chat.completions.create(model="gpt-4o-mini", ...).choices[0].message.content
        # Anthropic: return anthropic.Anthropic().messages.create(model="claude-3-haiku-...", ...).content[0].text
        # Ollama:    return requests.post("http://localhost:11434/api/generate", json={...}).json()["response"]
        # HF:        return pipeline("text-generation", model="codellama/...")(prompt)[0]["generated_text"]
        raise NotImplementedError("Conecta tu backend LLM aquí")

    agent = ErrorGuidedKernelAgent(
        llm_generate_fn=llm_generate_with_api,
        max_attempts=10
    )

    return agent

# Test en benchmarks L2
def evaluate_l2_pass_rate(agent: ErrorGuidedKernelAgent, l2_benchmarks: list) -> float:
    """
    Evalúa pass rate del agente en L2.

    L2 incluye:
    - Reducciones simples (sum, max, min)
    - Broadcasting básico
    - Operaciones element-wise con máscaras
    """
    successes = 0

    for benchmark in l2_benchmarks:
        _, metadata = agent.generate_kernel(
            task_spec=benchmark['description'],
            reference_fn=benchmark['reference']
        )

        if metadata['success']:
            successes += 1

        print(f"{benchmark['name']}: {'✓' if metadata['success'] else '✗'} ({metadata['attempts']} attempts)")

    pass_rate = successes / len(l2_benchmarks)
    print(f"\n{'='*50}")
    print(f"L2 Pass Rate: {pass_rate*100:.1f}% ({successes}/{len(l2_benchmarks)})")
    print(f"{'='*50}")

    return pass_rate

# Meta: alcanzar ~60% en L2
# agent = build_l2_agent("your-api-key")
# pass_rate = evaluate_l2_pass_rate(agent, load_l2_benchmarks())
# assert pass_rate >= 0.55, f"Pass rate {pass_rate:.2%} below target 55%"
```

---

## Actividad de Trabajo — Bloque 2 (25 minutos)

**Ejercicio 1: Conectar `ErrorGuidedKernelAgent` con `enhanced_classify_error` (15 min)**

Modifica `ErrorGuidedKernelAgent` para:
1. En cada intento fallido, llamar a `enhanced_classify_error(code, error, test_result)`
   en lugar de pasar el error crudo al prompt.
2. Usar `feedback.to_prompt()` como parte del mensaje de reintento.
3. Evaluar el agente mejorado sobre los 3 problemas L1:
   - `vector_add`, `relu`, `transpose_2d` (el que tiene el bug column-major del Bloque 1)

Completa la tabla comparativa:

| Problema | Sin `classify_error` (raw error) | Con `enhanced_classify_error` | Reducción de intentos |
|---|---|---|---|
| vector_add | | | |
| relu | | | |
| transpose_2d | | | |

¿Para cuál problema ayuda más clasificar el error?

**Ejercicio 2: Diagnóstico del `ERROR_TO_PROMPT_MAP` (10 min)**

Revisar el `ERROR_TO_PROMPT_MAP` definido en la sección de teoría:

1. ¿Qué subcategoría tiene el hint más específico? ¿Cuál tiene el más genérico?
2. Añade una entrada nueva para `"WRONG_2D_INDEXING"` con un hint que incluya
   el patrón correcto (`row*width + col`) y un ejemplo de 1 línea.
3. ¿Por qué el mapa usa `subcategory` como clave y no `category`?
   Da un ejemplo donde dos subcategorías distintas de `RUNTIME` necesitarían
   hints completamente diferentes.

---

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: Un kernel falla con `RuntimeError: CUDA error: device-side assert`
en la línea `tl.store(out_ptr + offsets, result)` (sin máscara). `classify_error`
lo categoriza como `RUNTIME/OUT_OF_BOUNDS`. El hint dice "añade máscara".
¿Por qué ese hint es suficiente para que el LLM corrija el bug, pero `"Fix the error"`
no lo sería aunque ambos aparecieran en el mismo prompt?
<details><summary>Respuesta</summary>
"Fix the error" no aporta <strong>estructura cognitiva</strong>: el LLM no sabe
si el fallo es de sintaxis, lógica, o memoria. "Añade máscara" le da tanto
el <em>componente a cambiar</em> como el <em>patrón correcto</em> que ya aparece
en los few-shot de S6 — anclando la corrección al conocimiento existente del modelo.
</details>

**Pregunta 2**: El agente lanza `enhanced_classify_error` en cada intento fallido.
Después del intento 3, los tres hints generados son distintos aunque todos sean
para el mismo kernel. ¿Eso es una señal de bug en el clasificador, en el LLM,
o en ambos? ¿Cómo lo diagnosticarías?
<details><summary>Respuesta</summary>
Serial de hints distintos en el mismo kernel indica que el LLM produce código
distinto en cada intento (por temperatura > 0), por lo que cada versión
dispará un error diferente. El clasificador está funcionando correctamente;
el problema puede ser temperatura alta o un prompt demasiado abierto. Bajar
temperatura a 0.1 y añadir más restricciones en el system prompt estabiliza
los intentos.
</details>
```

### Para Pensar

> *Si el agente comete el mismo error 3 veces seguidas (`RUNTIME/RACE_CONDITION`
> en reducciones), el `classify_error` siempre devuelve el mismo hint. ¿Qué
> cambiarías en la estrategia después del 2.º fallo repetido? ¿Puedes diseñar
> un mecanismo que escale el hint automáticamente (hint corto → hint con
> ejemplo de código completo)?*

---

## Resumen

```{admonition} Resumen
:class: important

**Debugging como Feedback para el Agente:**

1. **Taxonomía de 4 categorías**:
   - Compilación: sintaxis, tipos, API
   - Runtime: OOB, race conditions, división por zero
   - Correctitud: lógica, off-by-one, broadcasting
   - Performance: coalescing, occupancy, redundancia

2. **Top-5 errores (80% de casos)**:
   1. Sin máscara en load/store → OOB
   2. Indexación 2D incorrecta → resultados incorrectos
   3. Race conditions → no determinístico
   4. Dtype incorrecto → overflow/underflow
   5. Warp divergence → lento

3. **classify_error()**: Convierte exceptions genéricas en feedback estructurado
   - Detecta categoría automáticamente
   - Genera sugerencias específicas
   - Extrae snippets problemáticos

4. **Test Suites como Verificadores**:
   - Boundary tests (n=1,31,32,33,255,256,257)
   - Determinism tests (10 runs)
   - Reference tests (vs PyTorch)

5. **Error-Guided Agent**:
   - Loop: generar → compilar → testear → clasificar error → feedback → regenerar
   - Aprende de errores previos
   - Meta: ~60% pass rate en L2 con feedback
```

````{admonition} Antipatrón: Error Messages Sin Estructura
:class: warning

**Malo**: Pasar error message directo al LLM
```python
prompt = f"This failed: {str(exception)}. Fix it."
```

**Bueno**: Clasificar y estructurar feedback
```python
feedback = classify_error(code, exception)
prompt = feedback.to_prompt()  # Instrucciones específicas
```

El LLM necesita contexto estructurado, no solo el mensaje de error crudo.
````

```{admonition} Cómo el Feedback Mejora el Pass Rate
:class: tip

**Sin feedback estructurado**: ~20-30% pass rate en L2
- El LLM no entiende QUÉ exactamente falló
- Reintentos son aleatorios
- Mismos errores se repiten

**Con feedback estructurado**: ~50-60% pass rate en L2
- El LLM recibe instrucciones específicas
- Cada reintento es más informado
- Aprende patrones de corrección

**Diferencia clave**: Convertir "compilation error" en "Add mask=offsets<n to all tl.load operations"
```

```{admonition} En tu Proyecto
:class: note

Implementa el agente error-guided para L2:

1. **Integra classify_error** en tu loop de generación
2. **Crea test suite** con boundary, determinism, reference tests
3. **Mide mejora**: compara pass rate con/sin feedback
4. **Documenta errores**: guarda historial para análisis
5. **Itera prompts**: usa patrones de errores para mejorar prompts base

**Meta realista**: 55-65% pass rate en L2 con 8-10 intentos por kernel.
```

---

## Tarea para Casa

### Ejercicio 1: `classify_error` completo con 4 detectores (★★☆)

Extiende `enhanced_classify_error` añadiendo los dos detectores que no
implementaste en clase:

3. **Dtype incorrecto en reducciones**: Detecta `.to(tl.int32)` dentro de una
   sección de reducción (`tl.sum`, `tl.max`) y sugiere mantener FP32 durante
   el cómputo.
4. **Warp divergence**: Detecta el patrón `if \w+ % \d+` (branch por tid)
   y sugiere sustituir por operación vectorial con máscara.

Escribe un test unitario por cada detector y verifica que los 4 pasan.

### Ejercicio 2: Agente en 10 problemas L2 con análisis de patrones (★★★)

Usa `ErrorGuidedKernelAgent` con `enhanced_classify_error` integrado:

1. Evalúa sobre los **10 primeros problemas L2** de KernelBench.
2. Implementa `analyze_error_history(agent)` que calcule:
   - Frecuencia de fallo por categoría (`COMPILE`, `RUNTIME`, `CORRECTNESS`, `PERFORMANCE`)
   - Intentos promedio hasta corrección por subcategoría
   - Subcategorías que **nunca** se corrigen dentro de `max_attempts`
3. Reporta el pass rate final y grafica la distribución de errores.
4. **Pregunta**: ¿Hay alguna subcategoría con tasa de corrección < 20%?
   Si la hay, ¿qué cambiarías en el `ERROR_TO_PROMPT_MAP` para esa subcategoría?

---

## Errores Comunes

```{admonition} Errores frecuentes al implementar feedback
:class: warning

1. **Feedback muy genérico**: "Fix the error" no ayuda. Sé específico: "Add mask=offsets<n to line 15"

2. **No testear determinismo**: Race conditions son silenciosas. Siempre ejecuta 10+ veces.

3. **Ignorar boundary cases**: n=257 (no múltiplo de 256) expone bugs de máscara.

4. **No clasificar correctamente**: Confundir runtime OOB con correctitud lógica cambia la estrategia de corrección.

5. **Demasiados intentos**: Si el agente falla 5+ veces, el problema está en el prompt base, no en el feedback.
```

---

## Referencias

- NVIDIA. [CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/). NVIDIA Developer.
- NVIDIA. [Nsight Systems](https://developer.nvidia.com/nsight-systems). NVIDIA Developer.
- OpenAI. [Best Practices for Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering). OpenAI Documentation.
- Anthropic. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). ArXiv 2022.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA 2026*
