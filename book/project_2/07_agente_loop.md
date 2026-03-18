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

# Sesión 7: Agente con Loop de Retroalimentación

```{admonition} Objetivos de la sesión
:class: tip
- Identificar las limitaciones del enfoque de prompt único de S6
- Diseñar e implementar un loop agéntico con herramientas especializadas
- Aplicar estrategias de reintento: simple, error-guided y self-reflection
- Construir un agente completo que mejora automáticamente kernels iterativamente
- Comparar pass rates: single-shot vs loop agent
```

## Configuración de Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/project_2/07_agente_loop.ipynb)

```{code-cell} ipython3
:tags: [skip-execution]

# Verificar GPU
!nvidia-smi

# Instalar dependencias
!pip install torch triton anthropic openai -q
```

```{code-cell} ipython3
:tags: [skip-execution]

import torch
import triton
import triton.language as tl
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass
import time
import traceback
```

## Bloque 1: De Prompt Único a Loop Agéntico

### Limitaciones del Prompt Único

En la Sesión 6 construimos un pipeline de prompt único que genera kernels Triton a partir de especificaciones. Este enfoque tiene limitaciones críticas:

```{code-cell} ipython3
:tags: [skip-execution]

# Enfoque de S6: un solo intento
def generate_kernel_single_shot(spec: str, llm_fn: Callable) -> str:
    """
    Genera un kernel con un único intento.
    Si falla... no hay segunda oportunidad.
    """
    prompt = f"""Generate a Triton kernel for:
{spec}

Return only the Python code with the kernel implementation."""

    response = llm_fn(prompt)
    return response
```

**Problemas identificados:**

1. **Sin retroalimentación**: Si el kernel no compila o falla en runtime, no hay mecanismo de corrección
2. **Sin verificación**: No sabemos si el output es correcto sin ejecutar y comparar
3. **Sin aprendizaje**: Errores comunes se repiten en cada ejecución
4. **Baja robustez**: Pass rate típico ~40-60% en kernels L1, <20% en L2

```{admonition} Reflexión
:class: note
En la práctica, los desarrolladores no escriben código perfecto al primer intento. Iteran basándose en errores del compilador, resultados incorrectos y métricas de performance. Un agente efectivo debe replicar este proceso.
```

### El Loop Agéntico: Generate → Execute → Verify → Regenerate

Un agente con loop de retroalimentación sigue este ciclo:

```
┌─────────────────────────────────────────────┐
│  1. GENERATE: LLM produce código inicial    │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  2. EXECUTE: Compilar y ejecutar kernel     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  3. VERIFY: Comparar vs referencia PyTorch  │
└──────────────┬──────────────────────────────┘
               │
         ┌─────┴─────┐
         │  Success? │
         └─────┬─────┘
               │
       ┌───────┴────────┐
       │                │
      YES               NO
       │                │
       ▼                ▼
   [RETURN]   ┌──────────────────────┐
              │ 4. CLASSIFY ERROR     │
              │ - Compilation error   │
              │ - Runtime error       │
              │ - Incorrect output    │
              │ - Performance issue   │
              └──────────┬────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ 5. FEEDBACK          │
              │ Actualizar contexto  │
              │ con info del error   │
              └──────────┬────────────┘
                         │
                         └─────► (volver a GENERATE)
```

### Diseño de Herramientas del Agente

Esta sesión introduce patrones de sistemas agénticos que también cubre P1 S10 (tool use, ReAct). P2 S7 y P1 S10 se desarrollan en paralelo y se complementan: uno desde el dominio GPU y otro desde LLMs generales. Si tu equipo ya ha cubierto P1 S10, puedes conectar directamente esos patrones con lo que implementas aquí; si no, construirás la intuición antes de formalizarla en P1. Diseñamos tres herramientas especializadas:

#### Tool 1: Code Executor

```{code-cell} ipython3
:tags: [skip-execution]

@dataclass
class ExecutionResult:
    """Resultado de ejecutar un kernel."""
    success: bool
    output: Optional[torch.Tensor] = None
    error_type: Optional[str] = None  # 'compilation', 'runtime', 'shape_mismatch'
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

def code_executor(code: str, inputs: Dict[str, torch.Tensor]) -> ExecutionResult:
    """
    Tool 1: Compila y ejecuta el kernel generado.

    Args:
        code: Código Python con el kernel Triton
        inputs: Diccionario de tensores de entrada

    Returns:
        ExecutionResult con el output o información del error
    """
    try:
        # Crear namespace para ejecutar el código
        namespace = {
            'torch': torch,
            'triton': triton,
            'tl': tl,
        }

        # Compilar el código
        exec(code, namespace)

        # Buscar la función del kernel (asumimos que termina en _kernel)
        kernel_fn = None
        for name, obj in namespace.items():
            if callable(obj) and name.endswith('_kernel') and hasattr(obj, 'run'):
                kernel_fn = obj
                break

        if kernel_fn is None:
            return ExecutionResult(
                success=False,
                error_type='compilation',
                error_message='No kernel function found ending with _kernel'
            )

        # Ejecutar el kernel
        start_time = time.time()

        # El código debe definir también una función wrapper
        # que prepara los tensores de salida y llama al kernel
        if 'run_kernel' in namespace:
            output = namespace['run_kernel'](**inputs)
        else:
            return ExecutionResult(
                success=False,
                error_type='compilation',
                error_message='No run_kernel wrapper function found'
            )

        execution_time = time.time() - start_time

        return ExecutionResult(
            success=True,
            output=output,
            execution_time=execution_time
        )

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        # Clasificar el tipo de error
        if 'Compilation' in error_trace or 'TritonError' in error_trace:
            error_type = 'compilation'
        elif 'shape' in error_msg.lower() or 'size' in error_msg.lower():
            error_type = 'shape_mismatch'
        else:
            error_type = 'runtime'

        return ExecutionResult(
            success=False,
            error_type=error_type,
            error_message=f"{error_msg}\n\n{error_trace}"
        )
```

#### Tool 2: Correctness Tester

```{code-cell} ipython3
:tags: [skip-execution]

@dataclass
class CorrectnessResult:
    """Resultado de verificar correctitud."""
    is_correct: bool
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None
    error_message: Optional[str] = None

def correctness_tester(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> CorrectnessResult:
    """
    Tool 2: Compara el output del kernel vs referencia PyTorch.

    Args:
        output: Tensor producido por el kernel
        reference: Tensor de referencia (PyTorch)
        rtol: Tolerancia relativa
        atol: Tolerancia absoluta

    Returns:
        CorrectnessResult indicando si el output es correcto
    """
    try:
        # Verificar que las formas coincidan
        if output.shape != reference.shape:
            return CorrectnessResult(
                is_correct=False,
                error_message=f"Shape mismatch: output {output.shape} vs reference {reference.shape}"
            )

        # Comparar valores con torch.allclose
        is_correct = torch.allclose(output, reference, rtol=rtol, atol=atol)

        # Calcular métricas de diferencia
        diff = torch.abs(output - reference)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if not is_correct:
            error_message = (
                f"Output incorrect: max_diff={max_diff:.2e}, "
                f"mean_diff={mean_diff:.2e}, rtol={rtol}, atol={atol}"
            )
        else:
            error_message = None

        return CorrectnessResult(
            is_correct=is_correct,
            max_diff=max_diff,
            mean_diff=mean_diff,
            error_message=error_message
        )

    except Exception as e:
        return CorrectnessResult(
            is_correct=False,
            error_message=f"Comparison failed: {str(e)}"
        )
```

#### Tool 3: Profiler

```{code-cell} ipython3
:tags: [skip-execution]

@dataclass
class ProfileResult:
    """Resultado de perfilar un kernel."""
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    speedup_vs_pytorch: Optional[float] = None

def profiler(
    kernel_fn: Callable,
    pytorch_fn: Callable,
    inputs: Dict[str, torch.Tensor],
    n_warmup: int = 10,
    n_iters: int = 100
) -> ProfileResult:
    """
    Tool 3: Mide la performance del kernel generado.

    Args:
        kernel_fn: Función que ejecuta el kernel Triton
        pytorch_fn: Función equivalente en PyTorch (baseline)
        inputs: Inputs para ambas funciones
        n_warmup: Iteraciones de warmup
        n_iters: Iteraciones para medir

    Returns:
        ProfileResult con métricas de performance
    """
    # Warmup para Triton
    for _ in range(n_warmup):
        _ = kernel_fn(**inputs)

    # Medir kernel Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = kernel_fn(**inputs)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / n_iters * 1000  # ms

    # Warmup para PyTorch
    for _ in range(n_warmup):
        _ = pytorch_fn(**inputs)

    # Medir PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = pytorch_fn(**inputs)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / n_iters * 1000  # ms

    speedup = pytorch_time / triton_time

    return ProfileResult(
        mean_time_ms=triton_time,
        std_time_ms=0.0,  # Simplificado
        min_time_ms=triton_time,
        speedup_vs_pytorch=speedup
    )
```

### Estrategias de Reintento

#### 1. Retry Simple

La estrategia más básica: si falla, volver a generar desde cero.

```{code-cell} ipython3
:tags: [skip-execution]

def retry_simple(spec: str, llm_fn: Callable, max_attempts: int = 3) -> str:
    """
    Estrategia 1: Retry simple sin feedback.
    Genera desde cero en cada intento.
    """
    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")

        prompt = f"""Generate a Triton kernel for:
{spec}

Return only the Python code."""

        code = llm_fn(prompt)

        # Intentar ejecutar (simplificado)
        result = try_execute(code)
        if result.success:
            print(f"Success on attempt {attempt + 1}")
            return code

    raise Exception(f"Failed after {max_attempts} attempts")
```

**Limitaciones**: No aprende de errores anteriores. Pass rate mejora solo ligeramente vs single-shot.

#### 2. Error-Guided Retry

Incluir el mensaje de error en el siguiente prompt.

```{code-cell} ipython3
:tags: [skip-execution]

def retry_error_guided(
    spec: str,
    llm_fn: Callable,
    max_attempts: int = 5
) -> str:
    """
    Estrategia 2: Error-guided retry.
    Incluye información del error anterior en el siguiente prompt.
    """
    previous_code = None
    previous_error = None

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")

        if attempt == 0:
            # Primera intento: prompt estándar
            prompt = f"""Generate a Triton kernel for:
{spec}

Return only the Python code with the kernel implementation."""
        else:
            # Intentos subsecuentes: incluir error feedback
            prompt = f"""The previous kernel attempt failed with this error:

Error type: {previous_error['type']}
Error message: {previous_error['message']}

The problematic code was:
```python
{previous_code}
```

Please analyze the error and generate a corrected kernel for:
{spec}

Return only the corrected Python code."""

        code = llm_fn(prompt)
        result = try_execute_and_verify(code, spec)

        if result.success:
            print(f"Success on attempt {attempt + 1}")
            return code

        # Guardar para el siguiente intento
        previous_code = code
        previous_error = {
            'type': result.error_type,
            'message': result.error_message
        }

    raise Exception(f"Failed after {max_attempts} attempts")
```

**Mejora**: Pass rate aumenta ~15-25% vs retry simple. El LLM puede corregir errores de compilación y lógica simple.

#### 3. Self-Reflection

Pedir al LLM que analice su propio error antes de regenerar.

```{code-cell} ipython3
:tags: [skip-execution]

def retry_self_reflection(
    spec: str,
    llm_fn: Callable,
    max_attempts: int = 5
) -> str:
    """
    Estrategia 3: Self-reflection.
    El LLM analiza su error antes de generar la corrección.
    """
    previous_code = None
    previous_error = None

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")

        if attempt == 0:
            prompt = f"""Generate a Triton kernel for:
{spec}

Return only the Python code."""
        else:
            # Paso 1: Self-reflection
            reflection_prompt = f"""You generated this Triton kernel:
```python
{previous_code}
```

It failed with this error:
{previous_error['type']}: {previous_error['message']}

Analyze the error and explain:
1. What went wrong?
2. What's the root cause?
3. How should it be fixed?

Provide a brief analysis."""

            analysis = llm_fn(reflection_prompt)
            print(f"Self-analysis: {analysis[:200]}...")

            # Paso 2: Generate correction basada en la reflexión
            prompt = f"""Based on this analysis:
{analysis}

Generate a corrected Triton kernel for:
{spec}

Return only the corrected Python code."""

        code = llm_fn(prompt)
        result = try_execute_and_verify(code, spec)

        if result.success:
            print(f"Success on attempt {attempt + 1}")
            return code

        previous_code = code
        previous_error = {
            'type': result.error_type,
            'message': result.error_message
        }

    raise Exception(f"Failed after {max_attempts} attempts")
```

**Mejora**: Pass rate aumenta ~25-35% vs retry simple en kernels L2 complejos. El paso de reflexión ayuda a errores de lógica más profundos.

### Cuándo Parar el Loop

Criterios para terminar el loop de retroalimentación:

```{code-cell} ipython3
:tags: [skip-execution]

@dataclass
class StopCondition:
    """Condiciones para detener el loop agéntico."""
    max_attempts: int = 5
    timeout_seconds: float = 300.0
    success_threshold: float = 0.99  # Correctitud requerida

def should_stop(
    attempt: int,
    elapsed_time: float,
    result: Optional[CorrectnessResult],
    condition: StopCondition
) -> tuple[bool, str]:
    """
    Determina si el loop debe detenerse.

    Returns:
        (should_stop, reason)
    """
    # Condición 1: Éxito
    if result is not None and result.is_correct:
        return True, "success"

    # Condición 2: Max intentos alcanzado
    if attempt >= condition.max_attempts:
        return True, "max_attempts"

    # Condición 3: Timeout
    if elapsed_time >= condition.timeout_seconds:
        return True, "timeout"

    return False, "continue"
```

## Actividad de Trabajo — Bloque 1 (25 minutos)

**Ejercicio 1: Completar `simple_loop_agent` (15 min)**

Implementa el cuerpo del loop agéntico básico usando las funciones
`code_executor` y `correctness_tester` definidas en la sección anterior.
El problema objetivo es `relu`: `y[i] = max(0, x[i])`.

```{code-cell} ipython3
:tags: [skip-execution]

def simple_loop_agent(
    spec: str,
    test_inputs: Dict[str, torch.Tensor],
    reference_output: torch.Tensor,
    llm_fn: Callable,
    max_attempts: int = 5
) -> Optional[str]:
    """
    Loop agéntico básico con retry simple (sin feedback de error).
    """
    for attempt in range(max_attempts):
        # TODO 1: Construir prompt con spec
        # TODO 2: Llamar llm_fn(prompt) para obtener código
        # TODO 3: Llamar code_executor(code) → ExecutionResult
        # TODO 4: Si execution_result.success es False → print error, continuar
        # TODO 5: Llamar correctness_tester(code, test_inputs, reference_output)
        # TODO 6: Si correct → return code
        pass

    return None  # Todos los intentos fallaron

# Prueba con ReLU
spec_relu = "Implement ReLU: y[i] = max(0, x[i]). Input: x:(N,) float32. Output: y:(N,) float32."
x = torch.randn(1024, device='cuda')
reference_relu = torch.relu(x)

result = simple_loop_agent(
    spec=spec_relu,
    test_inputs={'x': x},
    reference_output=reference_relu,
    llm_fn=your_llm_fn
)
print(f"Resultado: {'✓ éxito' if result else '✗ falló todos los intentos'}")
```

**Ejercicio 2: Comparar retry simple vs error-guided (10 min)**

Modifica `simple_loop_agent` para crear `error_guided_agent`:
- La diferencia: en `error_guided`, el prompt del intento N incluye
  `execution_result.error_message` del intento N-1.

Evalúa ambos sobre el mismo problema `relu` con `max_attempts=5`.
Rellena la tabla:

| Estrategia | Intente exitoso (1-5 o "fallido") | Tipo del primer error |
|---|---|---|
| `simple_loop_agent` | | |
| `error_guided_agent` | | |

¿Por qué `error_guided` debería necesitar menos intentos?

---

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

1. **Retry ciego vs retry informado**: En el loop simple, el agente regenera desde
   cero en cada intento. ¿Qué información del intento anterior podría incluir en
   el prompt para que el LLM no repita el mismo error? ¿Cuál es el riesgo de
   incluir demasiado contexto?
2. **Tipos de error y estrategia**: Un `SyntaxError` de Python y un
   `torch.allclose` fallido son dos tipos de fallo muy distintos. ¿Qué estrategia
   de retry es más adecuada para cada uno? ¿Tiene sentido usar la misma?
3. **Cuándo parar**: Si el agente lleva 4 intentos fallidos y el 5.º se ve
   "prometedor" pero lento, ¿debería parar por timeout o esperar? Diseña el
   criterio de parada ideal para un entorno de producción.
```

---

## Bloque 2: Implementar el Loop Completo

### Arquitectura Completa del Agente

Un agente completo integra todas las herramientas en una arquitectura cohesiva:

```{code-cell} ipython3
:tags: [skip-execution]

@dataclass
class Problem:
    """Definición de un problema de kernel."""
    spec: str  # Especificación del kernel
    test_inputs: Dict[str, torch.Tensor]  # Inputs de prueba
    reference_fn: Callable  # Función PyTorch de referencia
    name: str = "unnamed"

@dataclass
class AttemptResult:
    """Resultado de un intento de generación."""
    attempt_num: int
    code: str
    execution_result: ExecutionResult
    correctness_result: Optional[CorrectnessResult] = None
    profile_result: Optional[ProfileResult] = None

class KernelAgent:
    """
    Agente completo con loop de retroalimentación para generar kernels.

    Arquitectura:
    - generate(spec, context) -> code
    - execute(code) -> result | error
    - verify(result, reference) -> bool
    - classify_error(error) -> error_type
    - feedback(error_type, error_msg) -> context_update
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        system_prompt: str,
        max_attempts: int = 5,
        timeout: float = 300.0,
        strategy: str = 'error_guided'  # 'simple', 'error_guided', 'reflection'
    ):
        """
        Inicializa el agente.

        Args:
            llm_fn: Función LLM (spec, prompt) -> code. Backend flexible.
            system_prompt: System prompt base para el agente
            max_attempts: Máximo número de intentos
            timeout: Timeout total en segundos
            strategy: Estrategia de reintento
        """
        self.llm_fn = llm_fn
        self.system_prompt = system_prompt
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.strategy = strategy
        self.history: List[AttemptResult] = []

    def solve(self, problem: Problem) -> Optional[AttemptResult]:
        """
        Resuelve un problema de kernel usando el loop agéntico.

        Args:
            problem: Definición del problema

        Returns:
            AttemptResult exitoso o None si falló todos los intentos
        """
        print(f"\n{'='*60}")
        print(f"Solving: {problem.name}")
        print(f"Strategy: {self.strategy}, Max attempts: {self.max_attempts}")
        print(f"{'='*60}\n")

        start_time = time.time()
        context = {
            'spec': problem.spec,
            'history': [],
            'error_feedback': None
        }

        for attempt in range(self.max_attempts):
            elapsed = time.time() - start_time
            if elapsed >= self.timeout:
                print(f"Timeout after {elapsed:.1f}s")
                break

            print(f"\n--- Attempt {attempt + 1}/{self.max_attempts} ---")

            # 1. GENERATE
            code = self.generate(context, attempt)

            # 2. EXECUTE
            print("Executing kernel...")
            exec_result = code_executor(code, problem.test_inputs)

            if not exec_result.success:
                print(f"❌ Execution failed: {exec_result.error_type}")
                print(f"   {exec_result.error_message[:200]}...")

                # Update context con error feedback
                context = self.update_context_after_error(
                    context, code, exec_result
                )
                continue

            print(f"✓ Execution successful ({exec_result.execution_time*1000:.2f}ms)")

            # 3. VERIFY
            print("Verifying correctness...")
            reference = problem.reference_fn(**problem.test_inputs)
            correct_result = correctness_tester(
                exec_result.output, reference
            )

            if not correct_result.is_correct:
                print(f"❌ Incorrect output: {correct_result.error_message}")

                context = self.update_context_after_incorrect(
                    context, code, correct_result
                )
                continue

            print(f"✓ Output correct! (max_diff={correct_result.max_diff:.2e})")

            # 4. SUCCESS - Optional profiling
            result = AttemptResult(
                attempt_num=attempt + 1,
                code=code,
                execution_result=exec_result,
                correctness_result=correct_result
            )

            self.history.append(result)

            print(f"\n{'='*60}")
            print(f"✓ SUCCESS on attempt {attempt + 1}")
            print(f"   Total time: {time.time() - start_time:.2f}s")
            print(f"{'='*60}")

            return result

        # Failed all attempts
        print(f"\n{'='*60}")
        print(f"❌ FAILED after {self.max_attempts} attempts")
        print(f"{'='*60}")
        return None

    def generate(self, context: Dict[str, Any], attempt: int) -> str:
        """
        Genera código usando el LLM con el contexto actual.
        """
        if attempt == 0 or self.strategy == 'simple':
            # Primera intento o estrategia simple: prompt básico
            prompt = self._build_initial_prompt(context)
        elif self.strategy == 'error_guided':
            # Error-guided: incluir feedback del error
            prompt = self._build_error_guided_prompt(context)
        elif self.strategy == 'reflection':
            # Self-reflection: primero analizar, luego generar
            prompt = self._build_reflection_prompt(context)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Llamar al LLM
        response = self.llm_fn(prompt)

        # Extraer código (asumimos que viene en markdown code block)
        code = self._extract_code(response)

        return code

    def _build_initial_prompt(self, context: Dict[str, Any]) -> str:
        """Construye el prompt inicial."""
        return f"""{self.system_prompt}

Generate a Triton kernel for the following specification:

{context['spec']}

Requirements:
- Use @triton.jit decorator
- Include a run_kernel() wrapper function that prepares output tensors and launches the kernel
- The kernel function name must end with _kernel
- Use appropriate BLOCK_SIZE (typically 1024 for 1D kernels)
- Handle edge cases with masking

Return only the complete Python code, no explanations."""

    def _build_error_guided_prompt(self, context: Dict[str, Any]) -> str:
        """Construye prompt con feedback de error."""
        if context['error_feedback'] is None:
            return self._build_initial_prompt(context)

        feedback = context['error_feedback']

        return f"""{self.system_prompt}

The previous kernel attempt failed. Here's what happened:

**Error Type**: {feedback['error_type']}

**Error Message**:
{feedback['error_message']}

**Previous Code**:
```python
{feedback['previous_code']}
```

Please analyze the error and generate a corrected kernel for:

{context['spec']}

Common fixes:
- Compilation errors: check Triton syntax, type annotations, and language constraints
- Runtime errors: verify tensor shapes, indexing, and bounds checking
- Incorrect output: review the mathematical logic and reduction operations

Return only the corrected Python code."""

    def _build_reflection_prompt(self, context: Dict[str, Any]) -> str:
        """Construye prompt con self-reflection (simplificado)."""
        # En una implementación completa, haríamos dos llamadas al LLM:
        # 1. Reflexión sobre el error
        # 2. Generación basada en la reflexión
        # Por simplicidad, combinamos ambos pasos aquí

        if context['error_feedback'] is None:
            return self._build_initial_prompt(context)

        feedback = context['error_feedback']

        return f"""{self.system_prompt}

The previous kernel failed. First, analyze what went wrong:

**Error Type**: {feedback['error_type']}
**Error Message**: {feedback['error_message']}
**Previous Code**:
```python
{feedback['previous_code']}
```

Step 1: In 2-3 sentences, explain:
- What was the root cause of this error?
- What specific mistake was made?

Step 2: Generate the corrected kernel for:
{context['spec']}

Provide your analysis followed by the corrected code."""

    def _extract_code(self, response: str) -> str:
        """
        Extrae código Python de la respuesta del LLM.
        Busca bloques ```python ... ``` o retorna la respuesta completa.
        """
        import re

        # Buscar bloques de código Python
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Si no hay bloques de código, retornar todo
        return response.strip()

    def update_context_after_error(
        self,
        context: Dict[str, Any],
        code: str,
        exec_result: ExecutionResult
    ) -> Dict[str, Any]:
        """Actualiza el contexto después de un error de ejecución."""
        context['error_feedback'] = {
            'error_type': exec_result.error_type,
            'error_message': exec_result.error_message,
            'previous_code': code
        }

        context['history'].append({
            'code': code,
            'result': 'execution_error',
            'error_type': exec_result.error_type
        })

        return context

    def update_context_after_incorrect(
        self,
        context: Dict[str, Any],
        code: str,
        correct_result: CorrectnessResult
    ) -> Dict[str, Any]:
        """Actualiza el contexto después de output incorrecto."""
        context['error_feedback'] = {
            'error_type': 'incorrect_output',
            'error_message': correct_result.error_message,
            'previous_code': code
        }

        context['history'].append({
            'code': code,
            'result': 'incorrect_output',
            'max_diff': correct_result.max_diff
        })

        return context
```

### Ejemplo Completo: Agente Resolviendo Softmax

Veamos el agente en acción resolviendo un kernel de softmax:

```{code-cell} ipython3
:tags: [skip-execution]

# Definir el problema
softmax_problem = Problem(
    name="Softmax 1D",
    spec="""
Implement a 1D softmax kernel in Triton.

Input: x (1D tensor)
Output: softmax(x) where softmax(x)[i] = exp(x[i]) / sum(exp(x))

For numerical stability, use the max trick:
1. Find max_val = max(x)
2. Compute exp(x - max_val)
3. Normalize by the sum

The kernel should handle arbitrary input sizes efficiently.
""",
    test_inputs={'x': torch.randn(4096, device='cuda')},
    reference_fn=lambda x: torch.softmax(x, dim=0)
)

# Configurar el LLM (backend flexible)
def my_llm_function(prompt: str) -> str:
    """
    Wrapper flexible para cualquier LLM.
    Aquí puedes usar Anthropic, OpenAI, local models, etc.
    """
    # Ejemplo con Anthropic
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

# System prompt especializado
system_prompt = """You are an expert in writing high-performance GPU kernels using Triton.

You write efficient, correct, and idiomatic Triton code following best practices:
- Use appropriate block sizes (typically 1024 for 1D)
- Apply masking for edge cases
- Minimize memory accesses
- Use tl.load() and tl.store() correctly
- Follow Triton's language constraints (no dynamic shapes in kernel body)

Always provide complete, runnable code."""

# Crear el agente con estrategia error-guided
agent = KernelAgent(
    llm_fn=my_llm_function,
    system_prompt=system_prompt,
    max_attempts=5,
    strategy='error_guided'
)

# Resolver el problema
result = agent.solve(softmax_problem)

if result:
    print("\n" + "="*60)
    print("FINAL KERNEL:")
    print("="*60)
    print(result.code)
    print("\n" + "="*60)
    print(f"Correctness: max_diff = {result.correctness_result.max_diff:.2e}")
    print(f"Execution time: {result.execution_result.execution_time*1000:.2f}ms")
    print("="*60)
```

**Salida esperada:**

```
============================================================
Solving: Softmax 1D
Strategy: error_guided, Max attempts: 5
============================================================

--- Attempt 1/5 ---
Executing kernel...
❌ Execution failed: compilation
   TritonError: Cannot use tl.sum outside of reduction context...

--- Attempt 2/5 ---
Executing kernel...
✓ Execution successful (0.45ms)
Verifying correctness...
❌ Incorrect output: Output incorrect: max_diff=1.23e-01, mean_diff=3.45e-02...

--- Attempt 3/5 ---
Executing kernel...
✓ Execution successful (0.42ms)
Verifying correctness...
✓ Output correct! (max_diff=2.38e-07)

============================================================
✓ SUCCESS on attempt 3
   Total time: 8.34s
============================================================
```

### Comparar Pass Rates: Single-Shot vs Loop Agent

Evaluemos el agente en múltiples problemas:

```{code-cell} ipython3
:tags: [skip-execution]

# Definir benchmark suite
benchmark_problems = [
    Problem(
        name="Vector Add (L1)",
        spec="c[i] = a[i] + b[i]",
        test_inputs={
            'a': torch.randn(8192, device='cuda'),
            'b': torch.randn(8192, device='cuda')
        },
        reference_fn=lambda a, b: a + b
    ),
    Problem(
        name="Vector Scale (L1)",
        spec="y[i] = alpha * x[i]",
        test_inputs={
            'x': torch.randn(8192, device='cuda'),
            'alpha': torch.tensor(2.5, device='cuda')
        },
        reference_fn=lambda x, alpha: alpha * x
    ),
    Problem(
        name="ReLU (L1)",
        spec="y[i] = max(0, x[i])",
        test_inputs={'x': torch.randn(8192, device='cuda')},
        reference_fn=lambda x: torch.relu(x)
    ),
    Problem(
        name="Softmax (L2)",
        spec="softmax(x)[i] = exp(x[i]) / sum(exp(x))",
        test_inputs={'x': torch.randn(4096, device='cuda')},
        reference_fn=lambda x: torch.softmax(x, dim=0)
    ),
    Problem(
        name="Layer Norm (L2)",
        spec="y[i] = (x[i] - mean) / sqrt(var + eps)",
        test_inputs={
            'x': torch.randn(4096, device='cuda'),
            'eps': torch.tensor(1e-5, device='cuda')
        },
        reference_fn=lambda x, eps: torch.nn.functional.layer_norm(
            x, normalized_shape=(x.shape[0],), eps=eps
        )
    ),
]

def evaluate_agent(
    agent: KernelAgent,
    problems: List[Problem],
    n_runs: int = 3
) -> Dict[str, Any]:
    """
    Evalúa el agente en un conjunto de problemas.

    Returns:
        Estadísticas de pass rate y número de intentos
    """
    results = []

    for problem in problems:
        problem_results = []

        for run in range(n_runs):
            print(f"\n{'='*60}")
            print(f"Problem: {problem.name}, Run {run+1}/{n_runs}")
            print(f"{'='*60}")

            agent.history = []  # Reset history
            result = agent.solve(problem)

            problem_results.append({
                'success': result is not None,
                'attempts': result.attempt_num if result else agent.max_attempts,
                'problem': problem.name
            })

        results.extend(problem_results)

    # Calcular estadísticas
    total = len(results)
    successes = sum(1 for r in results if r['success'])
    pass_rate = successes / total * 100

    successful_attempts = [r['attempts'] for r in results if r['success']]
    avg_attempts = sum(successful_attempts) / len(successful_attempts) if successful_attempts else 0

    # Por nivel
    l1_results = [r for r in results if 'L1' in r['problem']]
    l2_results = [r for r in results if 'L2' in r['problem']]

    l1_pass_rate = sum(1 for r in l1_results if r['success']) / len(l1_results) * 100 if l1_results else 0
    l2_pass_rate = sum(1 for r in l2_results if r['success']) / len(l2_results) * 100 if l2_results else 0

    return {
        'overall_pass_rate': pass_rate,
        'l1_pass_rate': l1_pass_rate,
        'l2_pass_rate': l2_pass_rate,
        'avg_attempts': avg_attempts,
        'total_problems': total,
        'successes': successes
    }

# Evaluar con estrategia simple
print("\n" + "="*60)
print("EVALUATING: Simple Retry Strategy")
print("="*60)

agent_simple = KernelAgent(
    llm_fn=my_llm_function,
    system_prompt=system_prompt,
    max_attempts=5,
    strategy='simple'
)

simple_stats = evaluate_agent(agent_simple, benchmark_problems, n_runs=3)

# Evaluar con estrategia error-guided
print("\n" + "="*60)
print("EVALUATING: Error-Guided Strategy")
print("="*60)

agent_guided = KernelAgent(
    llm_fn=my_llm_function,
    system_prompt=system_prompt,
    max_attempts=5,
    strategy='error_guided'
)

guided_stats = evaluate_agent(agent_guided, benchmark_problems, n_runs=3)

# Comparar resultados
print("\n" + "="*60)
print("COMPARISON: Single-Shot vs Loop Agents")
print("="*60)

comparison_data = {
    'Strategy': ['Single-Shot (S6)', 'Simple Retry', 'Error-Guided'],
    'Overall Pass Rate': [45.0, simple_stats['overall_pass_rate'], guided_stats['overall_pass_rate']],
    'L1 Pass Rate': [60.0, simple_stats['l1_pass_rate'], guided_stats['l1_pass_rate']],
    'L2 Pass Rate': [20.0, simple_stats['l2_pass_rate'], guided_stats['l2_pass_rate']],
    'Avg Attempts': [1.0, simple_stats['avg_attempts'], guided_stats['avg_attempts']]
}

import pandas as pd
df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))

print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)
print(f"✓ Error-guided retry improves pass rate by ~{guided_stats['overall_pass_rate'] - 45:.0f}% vs single-shot")
print(f"✓ L1 kernels: ~{guided_stats['l1_pass_rate']:.0f}% success rate")
print(f"✓ L2 kernels: ~{guided_stats['l2_pass_rate']:.0f}% success rate")
print(f"✓ Average {guided_stats['avg_attempts']:.1f} attempts needed for success")
print("="*60)
```

**Resultados esperados:**

```
============================================================
COMPARISON: Single-Shot vs Loop Agents
============================================================
        Strategy  Overall Pass Rate  L1 Pass Rate  L2 Pass Rate  Avg Attempts
Single-Shot (S6)               45.0          60.0          20.0           1.0
    Simple Retry               62.0          78.0          33.0           2.8
   Error-Guided               81.0          93.0          56.0           2.3

============================================================
KEY FINDINGS:
============================================================
✓ Error-guided retry improves pass rate by ~36% vs single-shot
✓ L1 kernels: ~93% success rate
✓ L2 kernels: ~56% success rate
✓ Average 2.3 attempts needed for success
============================================================
```

### Verificación con torch.allclose()

La función `torch.allclose()` es crucial para verificar correctitud numérica:

```{code-cell} ipython3
:tags: [skip-execution]

def demo_torch_allclose():
    """
    Demuestra el uso de torch.allclose para verificación.
    """
    # Caso 1: Valores idénticos
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    print(f"Identical: {torch.allclose(a, b)}")  # True

    # Caso 2: Diferencias pequeñas (dentro de tolerancia)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0000001, 2.0000001, 3.0000001])
    print(f"Small diff: {torch.allclose(a, b, rtol=1e-5, atol=1e-8)}")  # True

    # Caso 3: Diferencias grandes (fuera de tolerancia)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.01, 2.01, 3.01])
    print(f"Large diff: {torch.allclose(a, b, rtol=1e-5, atol=1e-8)}")  # False

    # Caso 4: Tolerancias explicadas
    # rtol: relative tolerance -> |a - b| <= atol + rtol * |b|
    # atol: absolute tolerance

    print("\nTolerance formula: |a - b| <= atol + rtol * |b|")
    print("For a=1.0, b=1.00001, rtol=1e-5, atol=1e-8:")
    print(f"  |1.0 - 1.00001| = 1e-5")
    print(f"  atol + rtol*|b| = 1e-8 + 1e-5*1.0 = 1.00001e-5")
    print(f"  1e-5 <= 1.00001e-5 -> True")

demo_torch_allclose()
```

### Baseline con torch.compile

Comparar también con `torch.compile` como baseline adicional:

```{code-cell} ipython3
:tags: [skip-execution]

def compare_with_torch_compile(problem: Problem):
    """
    Compara el kernel generado con torch.compile baseline.
    """
    # 1. Triton kernel generado por agente
    agent = KernelAgent(
        llm_fn=my_llm_function,
        system_prompt=system_prompt,
        max_attempts=5,
        strategy='error_guided'
    )

    result = agent.solve(problem)
    if result is None:
        print("Agent failed to generate working kernel")
        return

    # 2. torch.compile baseline
    def pytorch_impl(**inputs):
        return problem.reference_fn(**inputs)

    compiled_fn = torch.compile(pytorch_impl, mode='max-autotune')

    # Warmup
    for _ in range(10):
        _ = compiled_fn(**problem.test_inputs)

    # Benchmark torch.compile
    torch.cuda.synchronize()
    start = time.time()
    n_iters = 100
    for _ in range(n_iters):
        _ = compiled_fn(**problem.test_inputs)
    torch.cuda.synchronize()
    torch_compile_time = (time.time() - start) / n_iters * 1000

    # Benchmark Triton kernel
    triton_time = result.execution_result.execution_time * 1000

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Problem: {problem.name}")
    print(f"Triton (agent):     {triton_time:.3f}ms")
    print(f"torch.compile:      {torch_compile_time:.3f}ms")
    print(f"Speedup:            {torch_compile_time/triton_time:.2f}x")
    print("="*60)

# Ejemplo
compare_with_torch_compile(softmax_problem)
```

## Actividad de Trabajo — Bloque 2 (25 minutos)

**Ejercicio 1: Evaluar `KernelAgent` en 4 problemas (20 min)**

Instancia `KernelAgent` con `strategy='error_guided'` y `max_attempts=5`.
Ejecútalo sobre los 4 problemas siguientes y completa la tabla:

```{code-cell} ipython3
:tags: [skip-execution]

agent = KernelAgent(
    llm_fn=your_llm_fn,
    system_prompt=TRITON_SYSTEM_PROMPT,   # de S6
    max_attempts=5,
    strategy='error_guided'
)

problems = [
    Problem(spec="c[i] = a[i] + b[i]",  reference_fn=lambda a,b: a+b,      name="vector_add_L1"),
    Problem(spec="y[i] = max(0, x[i])",  reference_fn=torch.relu,            name="relu_L1"),
    Problem(spec="y[i] = alpha * x[i]",  reference_fn=lambda x,a: x*a,      name="scale_L1"),
    Problem(spec="y[i]=exp(x[i])/sum(exp(x))", reference_fn=lambda x: torch.softmax(x, 0), name="softmax_L2"),
]

for p in problems:
    result = agent.solve(p)
    # TODO: Registrar attempts_needed, success, error_types
```

| Problem | Nivel | Éxito | Intentos hasta éxito | Error más frecuente |
|---------|-------|-------|----------------------|---------------------|
| vector_add | L1 | | | |
| relu | L1 | | | |
| scale | L1 | | | |
| softmax | L2 | | | |

¿El pass rate en L2 es menor? ¿Cuántos intentos extra necesita?

**Ejercicio 2: Comparar las 3 estrategias (5 min)**

Cambia la `strategy` de `KernelAgent` entre `'simple'`, `'error_guided'` y
`'self_reflection'` y ejecuta solo sobre `softmax_L2`.
Rellena:

| Estrategia | Éxito | Intentos | Observación |
|---|---|---|---|
| `simple` | | | |
| `error_guided` | | | |
| `self_reflection` | | | |

¿Para qué tipo de error (`SyntaxError`, mask, lógica) ayuda más
`self_reflection` sobre `error_guided`?

---

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: Un agente usa `strategy='error_guided'` y en el intento 3 recibe
`RuntimeError: CUDA error: device-side assert triggered`. ¿El agente lo clasifica
como `syntax_error`, `runtime_error`, o `correctness_error`? ¿Qué incluiría en
el prompt del intento 4 y por qué eso es mejor que regenerar desde cero?
<details><summary>Respuesta</summary>
Es un <strong>runtime_error</strong> (el kernel compiló pero falló en ejecución
GPU). El agente incluye el <code>error_message</code> completo en el prompt:
"Previous attempt failed with: CUDA error: device-side assert triggered — indica
acceso fuera de límites, probablemente máscara incorrecta o offset negativo".
Esto guía al LLM a revisar índices, en lugar de reescribir la lógica completa.
</details>

**Pregunta 2**: El agente resolvió `relu_L1` en el intento 1 y `softmax_L2` en el
intento 4. En el Resumen de Sesión, ¿cuál es la diferencia conceptual entre
ambos que explica la mayor dificultad del softmax?
<details><summary>Respuesta</summary>
<code>relu</code> es elementwise (cada elemento independiente, 1 pasada de
memoria). <code>softmax</code> necesita una <strong>reducción global</strong>
(calcular <code>sum(exp(x))</code> requiere conocer todos los elementos antes de
calcular cada salida) → necesita 2 pasadas o sincronización → el LLM tiende a
omitir <code>atomic_add</code> o a confundir el patrón de indexing.
</details>
```

### Para Pensar

> *En S8 usaremos el historial de errores del agente para depurar y mejorar el
> sistema. Si el agente siempre falla en el mismo tipo de error (ej. máscaras en
> reducciones), ¿cómo usarías ese patrón para actualizar automáticamente el
> `system_prompt`? ¿Qué riesgo introduciría esa actualización automática?*

---

## Resumen

En esta sesión construimos un **agente con loop de retroalimentación** que supera las limitaciones del enfoque de prompt único de S6:

**Conceptos clave:**

1. **Loop agéntico**: Generate → Execute → Verify → Classify Error → Regenerate
2. **Herramientas especializadas**: code_executor, correctness_tester, profiler
3. **Estrategias de reintento**:
   - Simple retry: regenerar desde cero (~10-15% mejora vs single-shot)
   - Error-guided: incluir feedback del error (~25-35% mejora)
   - Self-reflection: analizar el error antes de corregir (~35-40% mejora en L2)
4. **Verificación**: torch.allclose() con tolerancias rtol/atol apropiadas
5. **Criterios de parada**: max intentos, timeout, éxito

**Resultados:**

| Estrategia | Pass Rate L1 | Pass Rate L2 | Avg Attempts |
|------------|--------------|--------------|--------------|
| Single-shot | ~60% | ~20% | 1.0 |
| Simple retry | ~78% | ~33% | 2.8 |
| Error-guided | ~93% | ~56% | 2.3 |
| Self-reflection | ~95% | ~65% | 2.7 |

**Aplicaciones prácticas:**

- Generación automática de kernels para nuevas operaciones
- Testing y validación de implementaciones
- Exploración de optimizaciones alternativas
- Sistema de autocompletado inteligente para kernel development

**Próximos pasos (S8):**

En la próxima sesión exploraremos **búsqueda con múltiples candidatos y ranking**, donde el agente genera múltiples variantes del kernel en paralelo y selecciona la mejor basándose en métricas de performance.

## Tarea para Casa

### Ejercicio 1: `analyze_error_patterns` en 10 problemas L1 (★★☆)

Extiende el Ejercicio 1 del Bloque 2 a los **10 primeros problemas L1** de KernelBench:

1. Implementa `analyze_error_patterns(agent, problems)` que recorra todos los
   intentos fallidos en `agent.history` y devuelva un `Counter` por `error_type`.
2. Ejecuta con `strategy='error_guided'` y `max_attempts=5`.
3. Genera una tabla de distribución de errores y un gráfico de barras.
4. **Pregunta**: ¿La categoría de error dominante es la misma en L1 que en softmax_L2?
   ¿Qué conclusión sacas sobre la relación entre nivel de dificultad y tipo de fallo?

### Ejercicio 2: Implementar self-reflection como función separada (★★★)

Completa la función `solve_with_reflection` con dos llamadas LLM separadas:

```python
def solve_with_reflection(
    agent: KernelAgent,
    problem: Problem,
    previous_code: str,
    error_type: str,
    error_message: str
) -> str:
    """
    Paso 1: llama llm_fn con reflection_prompt para obtener análisis.
    Paso 2: llama llm_fn con correction_prompt que incluye el análisis.
    Retorna el código corregido.
    """
    # TODO: Construir reflection_prompt
    # TODO: analysis = agent.llm_fn(reflection_prompt)
    # TODO: Construir correction_prompt incluyendo analysis
    # TODO: return agent.llm_fn(correction_prompt)
    pass
```

Compara la función contra `strategy='error_guided'` sobre `softmax_L2`
con 5 corridas (usa `random.seed` fijo). ¿Self-reflection reduce intentos promedio?

## Referencias

### Papers

1. **ReAct: Synergizing Reasoning and Acting in Language Models**
   Yao et al., 2023
   https://arxiv.org/abs/2210.03629
   - Framework teórico para agentes con herramientas

2. **Reflexion: Language Agents with Verbal Reinforcement Learning**
   Shinn et al., 2023
   https://arxiv.org/abs/2303.11366
   - Self-reflection en agentes LLM

3. **Toolformer: Language Models Can Teach Themselves to Use Tools**
   Schick et al., 2023
   https://arxiv.org/abs/2302.04761
   - Aprendizaje de uso de herramientas

### Documentación

1. **Triton Documentation - Error Handling**
   https://triton-lang.org/main/programming-guide/chapter-2/error-handling.html

2. **PyTorch - torch.allclose**
   https://pytorch.org/docs/stable/generated/torch.allclose.html

3. **PyTorch - torch.compile**
   https://pytorch.org/docs/stable/generated/torch.compile.html

### Recursos Adicionales

1. **LangChain Agent Documentation**
   https://python.langchain.com/docs/modules/agents/
   - Framework para construir agentes con herramientas

2. **Anthropic - Tool Use Guide**
   https://docs.anthropic.com/claude/docs/tool-use
   - Guía oficial de tool use con Claude

3. **OpenAI - Function Calling Guide**
   https://platform.openai.com/docs/guides/function-calling
   - Implementación de herramientas en GPT

---

**Nota**: Esta sesión introduce patrones de sistemas agénticos al dominio GPU. Estos patrones se complementan con los que formalizas en P1 S10 (tool use, ReAct), que cursa en paralelo. La combinación de ambos te da una visión completa: fundamentos teóricos en P1, aplicación práctica en P2.
