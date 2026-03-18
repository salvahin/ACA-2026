---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Del Prompt al Agente Kernel

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/project_2/06_prompt_a_agente_kernel.ipynb)

```{code-cell} ipython3
:tags: [skip-execution]

# Configuración para Google Colab
import sys
import subprocess

if 'google.colab' in sys.modules:
    # Instalar dependencias necesarias
    subprocess.run(["pip", "install", "-q", "triton>=2.1.0"], check=True)
    subprocess.run(["pip", "install", "-q", "torch"], check=True)

    print("Entorno configurado para Colab")
else:
    print("Ejecutando en entorno local")
```

## Introducción

Project 1 y Project 2 se cursan **en paralelo**. Al llegar a esta sesión (P2-S6), ya habrás cubierto en P1:
- **P1 S3**: Ingeniería de prompts (few-shot, Chain-of-Thought, system prompts)
- **P1 S6-S7**: Gramáticas Triton L1-L4 (EBNF, XGrammar, JSON Schema)

> P1 S10 (sistemas agénticos) se verá más adelante en P1 — por ahora esta sesión introduce los fundamentos del pipeline de generación.

Esta sesión **NO** re-enseña esos conceptos. En cambio, los **APLICA** al dominio específico de generación de kernels GPU con Triton.

**Pregunta central**: ¿Cómo diseñar prompts y pipelines que permitan a un LLM generar kernels Triton correctos y eficientes?

**Formato de sesión**: 25' clase + 25' trabajo + 10' discusión + 25' clase + 25' trabajo + 10' cierre

```{admonition} Objetivos de la sesión
:class: tip

1. Adaptar técnicas de prompt engineering al dominio GPU
2. Diseñar system prompts y few-shot examples para generación de kernels
3. Construir un pipeline completo: prompt → generación → evaluación
4. Iterar sobre prompts para mejorar pass rate en KernelBench L1
```

---

## Bloque 1: Adaptación de P1 al Dominio GPU

### Sincronización con P1: Lo que ya cubriste en paralelo

Recordatorio breve (5 minutos) de las herramientas que ya conoces:

1. **Prompts estructurados** (P1 S3):
   - System prompt: define el rol y las reglas globales
   - Few-shot learning: ejemplos input-output para guiar al modelo
   - Chain-of-Thought: razonamiento paso a paso para problemas complejos

2. **Gramáticas como restricciones** (P1 S6-S7):
   - EBNF, XGrammar, JSON Schema
   - Forzar outputs válidos sintácticamente
   - Reducir el espacio de búsqueda del LLM

3. **Sistemas agénticos** (P1 S10):
   - Tool use: el LLM puede invocar funciones externas
   - ReAct: reasoning + acting en bucles iterativos
   - RAG: recuperar contexto relevante antes de generar

**Hoy no re-enseñamos esto**. Lo aplicamos al reto de generar kernels GPU.

### El Reto: Generar Kernels Triton Correctos con un LLM

Generar código Triton es **más difícil** que generar Python genérico porque:

1. **Semántica paralela**: el código se ejecuta en bloques paralelos (CUDA grids)
2. **Gestión explícita de memoria**: `tl.load`, `tl.store`, punteros, offsets
3. **Máscaras obligatorias**: para evitar accesos fuera de límites
4. **Restricciones de tipos**: `constexpr`, `tl.constexpr`
5. **Patrones específicos**: elementwise, reductions, fused kernels

Un LLM entrenado en código genérico puede generar sintaxis válida pero con errores sutiles (máscaras faltantes, offsets incorrectos, race conditions).

```{admonition} Ejemplo de error común
:class: warning

```python
# INCORRECTO: falta máscara en tl.load
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # ❌ Sin máscara si offsets >= n_elements
    y = tl.load(y_ptr + offsets)  # ❌ Sin máscara
    tl.store(out_ptr + offsets, x + y)  # ❌ Sin máscara
```

**Correcto**: siempre usar `mask=offsets < n_elements`.
```

### Contexto Necesario para Generar Kernels GPU

Para generar un kernel Triton correcto, el LLM necesita:

#### 1. Especificación de la operación
- **Qué hace**: "Suma dos tensores element-wise"
- **Fórmula matemática**: `out[i] = x[i] + y[i]`
- **Propiedades**: comutativa, associativa, broadcast, etc.

#### 2. Shapes y dtypes
- Dimensiones de entrada: `x: (N,)`, `y: (N,)`
- Dimensión de salida: `out: (N,)`
- Tipos de datos: `float32`, `float16`, `int32`, etc.

#### 3. Restricciones de memoria
- **BLOCK_SIZE**: tamaño del bloque de hilos (potencia de 2, típicamente 1024)
- **Máscaras**: `mask = offsets < n_elements` para evitar out-of-bounds
- **Alineación**: algunos kernels requieren que `n_elements % BLOCK_SIZE == 0`

#### 4. Patrón base
- **Elementwise**: operación independiente por elemento (`add`, `relu`, `sigmoid`)
- **Reduction**: agregación (`sum`, `max`, `softmax`)
- **Indexing**: gather, scatter, transpose
- **Fused**: combinación de operaciones (e.g., `gelu = x * 0.5 * (1 + tanh(...))`)

```{admonition} Ejemplo de especificación completa
:class: note

**Operación**: ReLU (Rectified Linear Unit)

**Fórmula**: `out[i] = max(0, x[i])`

**Shapes**:
- Input: `x: (N,)` de tipo `float32`
- Output: `out: (N,)` de tipo `float32`

**Restricciones**:
- `BLOCK_SIZE = 1024`
- Usar máscara para `offsets < n_elements`

**Patrón**: Elementwise (operación independiente por elemento)
```

### Diseño de Prompts Especializados para Kernels GPU

Ahora aplicamos las técnicas de P1 S3 al dominio Triton:

#### System Prompt con Reglas Triton

El system prompt define el rol del LLM y las reglas globales que debe seguir.

```{code-cell} ipython3
TRITON_SYSTEM_PROMPT = """Eres un experto en programación de kernels GPU con Triton.

REGLAS OBLIGATORIAS:
1. Siempre usa máscaras en tl.load y tl.store: mask = offsets < n_elements
2. BLOCK_SIZE debe ser tl.constexpr
3. Calcula offsets como: pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
4. Usa tl.program_id(axis) para obtener el índice del bloque
5. Para operaciones element-wise, usa un solo grid 1D
6. Para reductions, usa atomic operations o múltiples pasadas
7. Incluye docstring con la especificación de la operación
8. Usa nombres de variables claros: x_ptr, y_ptr, out_ptr, n_elements

ESTRUCTURA ESPERADA:
```python
import triton
import triton.language as tl

@triton.jit
def kernel_name(input_ptrs, output_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    \"\"\"Docstring con spec.\"\"\"
    # 1. Obtener program_id
    # 2. Calcular offsets
    # 3. Crear máscara
    # 4. Load con máscara
    # 5. Computación
    # 6. Store con máscara
```

Genera código Triton correcto, eficiente y seguro."""

print("System prompt definido")
```

#### Few-Shot: Ejemplos de Kernels Correctos vs Incorrectos

El few-shot proporciona ejemplos concretos para que el LLM aprenda el patrón.

```{code-cell} ipython3
FEW_SHOT_EXAMPLES = """
EJEMPLO 1: Suma element-wise (CORRECTO)

Especificación:
- Operación: out[i] = x[i] + y[i]
- Shapes: x:(N,), y:(N,), out:(N,)
- Dtype: float32

Código Triton:
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    \"\"\"Suma dos tensores element-wise: out = x + y\"\"\"
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

EJEMPLO 2: ReLU (CORRECTO)

Especificación:
- Operación: out[i] = max(0, x[i])
- Shapes: x:(N,), out:(N,)
- Dtype: float32

Código Triton:
```python
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    \"\"\"Aplica ReLU: out = max(0, x)\"\"\"
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(0.0, x)
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

EJEMPLO 3: Multiplicación escalar (CORRECTO)

Especificación:
- Operación: out[i] = x[i] * alpha
- Shapes: x:(N,), out:(N,)
- Dtype: float32
- Parámetro: alpha (escalar)

Código Triton:
```python
import triton
import triton.language as tl

@triton.jit
def mul_scalar_kernel(x_ptr, out_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    \"\"\"Multiplica tensor por escalar: out = x * alpha\"\"\"
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * alpha
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

CONTRAEJEMPLO: Suma SIN máscaras (INCORRECTO)

```python
# ❌ INCORRECTO: falta mask en tl.load y tl.store
@triton.jit
def bad_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # ❌ Sin máscara
    y = tl.load(y_ptr + offsets)  # ❌ Sin máscara
    tl.store(out_ptr + offsets, x + y)  # ❌ Sin máscara
```

**Error**: Si `n_elements` no es múltiplo de `BLOCK_SIZE`, los últimos hilos accederán fuera de límites.
"""

print("Few-shot examples definidos")
```

#### Chain-of-Thought para Kernels Complejos

Para operaciones complejas (fused kernels, reductions), podemos pedir al LLM que razone paso a paso.

```{code-cell} ipython3
def build_cot_prompt(spec: str) -> str:
    """Construye un prompt con Chain-of-Thought para kernels complejos."""
    return f"""Genera un kernel Triton para la siguiente operación.

Especificación:
{spec}

Piensa paso a paso:
1. ¿Qué tipo de patrón es? (elementwise, reduction, indexing, fused)
2. ¿Cuántos inputs y outputs hay?
3. ¿Qué shape tienen?
4. ¿Qué operaciones matemáticas se necesitan?
5. ¿Se necesita más de una pasada (e.g., reduction)?
6. ¿Cómo calcular offsets y máscaras?

Luego genera el código Triton completo."""

# Ejemplo de uso
spec_gelu = """
Operación: GELU (Gaussian Error Linear Unit)
Fórmula: out[i] = x[i] * 0.5 * (1 + tanh(sqrt(2/π) * (x[i] + 0.044715 * x[i]^3)))
Shapes: x:(N,), out:(N,)
Dtype: float32
"""

cot_prompt = build_cot_prompt(spec_gelu)
print("Chain-of-Thought prompt:")
print(cot_prompt)
```

#### Gramáticas como Restricción del Generador

En P1 S6-S7 (que estás cursando en paralelo) estás aprendiendo a usar gramáticas (EBNF, XGrammar, JSON Schema) para restringir la salida del LLM.

Para kernels Triton, podríamos definir una gramática que:
1. Fuerza la presencia de `@triton.jit`
2. Requiere parámetros con nombres estándar (`*_ptr`, `n_elements`, `BLOCK_SIZE`)
3. Valida la estructura: program_id → offsets → mask → load → compute → store

```{admonition} Gramáticas para Triton (Opcional)
:class: note

Si tu equipo ya implementó gramáticas en P1 S6-S7, puedes reutilizarlas aquí para forzar sintaxis válida. Esto reduce errores de parsing pero no garantiza corrección semántica (e.g., offsets incorrectos).

**Ejemplo de regla EBNF**:
```ebnf
triton_kernel ::= "@triton.jit" NEWLINE "def" IDENTIFIER "(" params ")" ":" NEWLINE body

body ::= pid_line offsets_line mask_line load_line+ compute_line+ store_line+

mask_line ::= "mask = offsets <" IDENTIFIER NEWLINE
```

Esto es **opcional** y más avanzado. Para esta sesión, nos enfocamos en prompts bien diseñados.
```

### LLM Backend Flexible

Cada equipo puede elegir su backend de LLM:
- **OpenAI**: GPT-4, GPT-3.5
- **Anthropic**: Claude 3 Opus/Sonnet
- **Google**: Gemini Pro
- **Local**: LLaMA, Mistral, CodeLlama con `transformers` u `ollama`
- **Otros**: Cohere, AI21, etc.

Usamos un patrón `llm_fn: Callable` para abstraer el proveedor:

```{code-cell} ipython3
from typing import Callable

def llm_fn_interface(prompt: str) -> str:
    """
    Interfaz estándar para llamar a un LLM.

    Args:
        prompt: El prompt completo (system + few-shot + user query)

    Returns:
        Código Triton generado (string)
    """
    raise NotImplementedError("Implementa tu propio llm_fn según tu proveedor")

# Ejemplo con OpenAI (requiere API key)
def llm_fn_openai(prompt: str, model: str = "gpt-4") -> str:
    """Backend OpenAI."""
    try:
        import openai
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": TRITON_SYSTEM_PROMPT},
                {"role": "user", "content": FEW_SHOT_EXAMPLES + "\n\n" + prompt}
            ],
            temperature=0.2,  # Baja temperatura para código
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"# Error: {e}"

# Ejemplo con Anthropic Claude (requiere API key)
def llm_fn_anthropic(prompt: str, model: str = "claude-3-sonnet-20240229") -> str:
    """Backend Anthropic."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=TRITON_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": FEW_SHOT_EXAMPLES + "\n\n" + prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"# Error: {e}"

# Ejemplo con modelo local (requiere transformers + GPU)
def llm_fn_local(prompt: str, model_name: str = "codellama/CodeLlama-7b-Instruct-hf") -> str:
    """Backend con modelo local."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        full_prompt = TRITON_SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES + "\n\n" + prompt
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True
        )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraer solo la parte generada (después del prompt)
        return generated[len(full_prompt):]
    except Exception as e:
        return f"# Error: {e}"

print("Backends de LLM definidos (requieren API keys o modelos locales)")
```

```{admonition} Configuración de API Keys
:class: warning

Para usar OpenAI, Anthropic u otros proveedores, necesitas:
1. Crear cuenta y obtener API key
2. Configurar variable de entorno: `export OPENAI_API_KEY="sk-..."`
3. O pasar la key directamente al cliente

Para modelos locales:
1. GPU con suficiente VRAM (8GB+ para modelos 7B)
2. Instalar `transformers`, `accelerate`, `bitsandbytes`
3. Descargar el modelo desde HuggingFace
```

## Actividad de Trabajo — Bloque 1 (25 minutos)

**Ejercicio 1: Construir system prompt + few-shot (15 min)**

A partir del `TRITON_SYSTEM_PROMPT` y los `FEW_SHOT_EXAMPLES` que acabas de leer:

1. En `MY_SYSTEM_PROMPT`, **añade al menos 2 reglas adicionales** que cubran los
   errores del contraejemplo (máscara faltante, offsets). Justifica por qué las
   redactaste de esa manera.

2. En `MY_FEW_SHOT_EXAMPLES`, **escribe 3 ejemplos correctos** que cubran:
   - 1 elementwise distinto a `add` y `relu` (ej. `sigmoid`, `abs`, `exp`)
   - 1 reduction simple (`sum` o `max` de vector 1D)
   - 1 fused simple (ej. `scale_shift: out = x * a + b`)

   Incluye también 1 contraejemplo con el error etiquetado con `# ❌`.

3. Elige tu backend de LLM. Sin API key, usa el `mock` de la plantilla.

**Ejercicio 2: Validación manual de salidas (10 min)**

Descomenta `test_manual_generation()` y genera un kernel `sigmoid`.
Rellena la tabla verificando la salida generada:

| Criterio | ¿Cumple? | Notas |
|---|---|---|
| `@triton.jit` presente | | |
| Máscara en `tl.load` | | |
| Máscara en `tl.store` | | |
| `BLOCK_SIZE: tl.constexpr` | | |
| Docstring con operación | | |
| Lógica matemática correcta | | |

¿Cuántos criterios falla el mock? ¿Cuántos falla tu LLM real (si tienes)?

```{code-cell} ipython3
:tags: [skip-execution]

# PLANTILLA PARA ALUMNOS

# 1. Tu system prompt
MY_SYSTEM_PROMPT = """
# TODO: Adapta TRITON_SYSTEM_PROMPT a tu estilo
# Copia y modifica el template anterior
"""

# 2. Tus few-shot examples
MY_FEW_SHOT_EXAMPLES = """
# TODO: Escribe al menos 3 ejemplos correctos + 1 contraejemplo
# Usa diferentes patrones: elementwise, fused, etc.
"""

# 3. Tu función LLM
def my_llm_fn(prompt: str) -> str:
    """
    TODO: Implementa tu backend.
    Opciones:
    - llm_fn_openai(prompt)
    - llm_fn_anthropic(prompt)
    - llm_fn_local(prompt)
    - Mock para testing sin API
    """
    # Mock simple para testing sin API
    return """
import triton
import triton.language as tl

@triton.jit
def generated_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.sigmoid(x)  # Ejemplo: sigmoid
    tl.store(out_ptr + offsets, out, mask=mask)
"""

# 4. Prueba manual
def test_manual_generation():
    spec = """
Operación: Sigmoid
Fórmula: out[i] = 1 / (1 + exp(-x[i]))
Shapes: x:(N,), out:(N,)
Dtype: float32
"""

    prompt = f"Genera un kernel Triton para:\n{spec}"
    code = my_llm_fn(prompt)

    print("=== CÓDIGO GENERADO ===")
    print(code)
    print("\n=== VALIDACIÓN MANUAL ===")
    print("✓ ¿Tiene @triton.jit?")
    print("✓ ¿Usa máscaras en load/store?")
    print("✓ ¿BLOCK_SIZE es tl.constexpr?")
    print("✓ ¿Docstring presente?")

# Descomentar para ejecutar (requiere backend configurado)
# test_manual_generation()
```

```{admonition} Discusión — Bloque 1 (10 min)
:class: tip

1. **Reducción vs elementwise**: Tu few-shot tiene `add`. ¿Por qué eso no es
   suficiente para que el LLM genere correctamente un kernel `sum` (reducción)?
   ¿Qué patrón cognitivo diferente necesita el modelo?
2. **Regla vs contraejemplo**: ¿Es más efectivo escribir en el system prompt
   "siempre usa máscara" o incluir un contraejemplo con `# ❌ Sin máscara`?
   Argumenten con lo que observaron en las generaciones del Ejercicio 2.
3. **Grammar vs prompt**: Una gramática XGrammar puede forzar `@triton.jit` y
   nombres de parámetros estándar. Describe un kernel Triton que pase esa
   gramática y aún así falle `step3_correctness` en `evaluate_kernel()`.
   ¿Qué no puede garantizar ninguna gramática?
```

---

## Bloque 2: Pipeline Prompt → Kernel → Evaluación

### Pipeline Completo: De Especificación a Pass Rate

Cada equipo diseña su pipeline para evaluar la calidad de generación:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Spec de     │      │ LLM genera  │      │ Pipeline S5:│      │ Resultado:  │
│ operación   │─────▶│ código      │─────▶│ compile +   │─────▶│ success/    │
│ (problema)  │      │ Triton      │      │ test + bench│      │ fail + error│
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

**Pasos**:

1. **Prompt al LLM**: Proporcionar spec de operación (fórmula, shapes, dtype)
2. **LLM genera código**: String con código Triton
3. **Pipeline de S5**:
   - Compilar el kernel con `triton.compile`
   - Testear vs implementación PyTorch de referencia
   - Medir performance con benchmark
4. **Medir pass rate**: `success_count / total_problems`

### Estructura del Pipeline

```{code-cell} ipython3
:tags: [skip-execution]

from dataclasses import dataclass
from typing import Optional, Callable
import torch
import triton

@dataclass
class KernelSpec:
    """Especificación de un kernel a generar."""
    name: str
    operation: str  # Descripción textual
    formula: str  # Fórmula matemática
    input_shapes: dict  # {"x": (N,), "y": (N,)}
    output_shape: tuple  # (N,)
    dtype: torch.dtype
    reference_fn: Callable  # Función PyTorch de referencia

@dataclass
class GenerationResult:
    """Resultado de generar y evaluar un kernel."""
    spec: KernelSpec
    generated_code: str
    success: bool
    error_msg: Optional[str] = None
    compile_time: Optional[float] = None
    correctness: Optional[bool] = None
    speedup: Optional[float] = None  # vs PyTorch

def build_prompt(spec: KernelSpec, system_prompt: str, few_shot: str) -> str:
    """Construye el prompt completo para el LLM."""
    user_query = f"""
Genera un kernel Triton para la siguiente operación:

Nombre: {spec.name}
Operación: {spec.operation}
Fórmula: {spec.formula}
Shapes de entrada: {spec.input_shapes}
Shape de salida: {spec.output_shape}
Dtype: {spec.dtype}

Genera código Triton completo y correcto.
"""
    return system_prompt + "\n\n" + few_shot + "\n\n" + user_query

def generate_kernel(spec: KernelSpec, llm_fn: Callable,
                   system_prompt: str, few_shot: str) -> str:
    """
    Genera código de kernel usando un LLM.

    Args:
        spec: Especificación del kernel
        llm_fn: Función que llama al LLM
        system_prompt: System prompt con reglas
        few_shot: Ejemplos few-shot

    Returns:
        Código Triton generado (string)
    """
    prompt = build_prompt(spec, system_prompt, few_shot)
    code = llm_fn(prompt)
    return code

print("Funciones de generación definidas")
```

### Evaluación: Compile + Test + Benchmark

Reutilizamos el pipeline de S5 para evaluar el kernel generado:

```{code-cell} ipython3
:tags: [skip-execution]

import time
import tempfile
import os

def evaluate_kernel(code: str, spec: KernelSpec) -> GenerationResult:
    """
    Evalúa un kernel generado: compila, testa corrección y mide performance.

    Args:
        code: Código Triton generado
        spec: Especificación del kernel

    Returns:
        GenerationResult con success, errores, métricas
    """
    result = GenerationResult(
        spec=spec,
        generated_code=code,
        success=False
    )

    try:
        # 1. COMPILACIÓN
        # Guardar código en archivo temporal para importar
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Intentar importar y compilar
        import importlib.util
        spec_module = importlib.util.spec_from_file_location("temp_kernel", temp_file)
        module = importlib.util.module_from_spec(spec_module)

        start = time.time()
        spec_module.loader.exec_module(module)
        compile_time = time.time() - start
        result.compile_time = compile_time

        # Buscar la función kernel (asumimos que hay una función con @triton.jit)
        kernel_fn = None
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, '__triton_jit__'):
                kernel_fn = obj
                break

        if kernel_fn is None:
            result.error_msg = "No se encontró función con @triton.jit"
            return result

        # 2. CORRECCIÓN
        # Crear tensores de prueba
        n = 1024
        torch.manual_seed(42)

        # Generar inputs según spec
        inputs = {}
        for name, shape in spec.input_shapes.items():
            # Reemplazar N con valor concreto
            concrete_shape = tuple(n if dim == "N" else dim for dim in shape)
            inputs[name] = torch.randn(concrete_shape, dtype=spec.dtype, device='cuda')

        # Output tensor
        output_shape = tuple(n if dim == "N" else dim for dim in spec.output_shape)
        output_triton = torch.empty(output_shape, dtype=spec.dtype, device='cuda')

        # Ejecutar kernel Triton
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        # Preparar argumentos (esto depende de la firma del kernel)
        # Para simplificar, asumimos orden: *inputs, output, n_elements, BLOCK_SIZE
        args = list(inputs.values()) + [output_triton, n, BLOCK_SIZE]
        kernel_fn[grid](*args)

        # Ejecutar referencia PyTorch
        output_torch = spec.reference_fn(**inputs)

        # Comparar
        if torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5):
            result.correctness = True
        else:
            result.correctness = False
            result.error_msg = "Salida incorrecta vs PyTorch"
            return result

        # 3. BENCHMARK (opcional, simplificado)
        # Medir tiempo Triton vs PyTorch
        num_runs = 100

        # Warm-up
        for _ in range(10):
            kernel_fn[grid](*args)
        torch.cuda.synchronize()

        # Benchmark Triton
        start = time.time()
        for _ in range(num_runs):
            kernel_fn[grid](*args)
        torch.cuda.synchronize()
        time_triton = (time.time() - start) / num_runs

        # Benchmark PyTorch
        start = time.time()
        for _ in range(num_runs):
            _ = spec.reference_fn(**inputs)
        torch.cuda.synchronize()
        time_torch = (time.time() - start) / num_runs

        result.speedup = time_torch / time_triton

        # Si llegamos aquí, todo OK
        result.success = True

    except Exception as e:
        result.error_msg = str(e)

    finally:
        # Limpiar archivo temporal
        if 'temp_file' in locals():
            os.unlink(temp_file)

    return result

print("Función de evaluación definida")
```

### Ejemplo Completo: Generar Kernel ReLU

Veamos un ejemplo completo del pipeline:

```{code-cell} ipython3
:tags: [skip-execution]

# Definir especificación de ReLU
relu_spec = KernelSpec(
    name="relu",
    operation="Rectified Linear Unit",
    formula="out[i] = max(0, x[i])",
    input_shapes={"x": ("N",)},
    output_shape=("N",),
    dtype=torch.float32,
    reference_fn=lambda x: torch.relu(x)
)

# Mock de LLM que retorna código correcto
def mock_llm_relu(prompt: str) -> str:
    return """
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    \"\"\"Aplica ReLU: out = max(0, x)\"\"\"
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(0.0, x)
    tl.store(out_ptr + offsets, out, mask=mask)
"""

# Generar y evaluar
print("=== GENERANDO KERNEL RELU ===")
code = generate_kernel(relu_spec, mock_llm_relu, TRITON_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES)
print(code)

print("\n=== EVALUANDO KERNEL ===")
result = evaluate_kernel(code, relu_spec)

print(f"Success: {result.success}")
print(f"Compile time: {result.compile_time:.4f}s")
print(f"Correctness: {result.correctness}")
print(f"Speedup vs PyTorch: {result.speedup:.2f}x")
if result.error_msg:
    print(f"Error: {result.error_msg}")
```

### Iterar sobre el Prompt: Mejorar Pass Rate

El desarrollo iterativo es clave:

1. **Medir baseline**: evaluar pass rate inicial en KernelBench L1
2. **Analizar fallos**: ¿qué tipo de errores ocurren? (sintaxis, offsets, máscaras)
3. **Mejorar prompt**:
   - Agregar reglas específicas para errores comunes
   - Añadir ejemplos que cubran casos edge
   - Refinar few-shot con contraejemplos
4. **Re-evaluar**: medir nuevo pass rate
5. **Repetir** hasta convergencia

```{code-cell} ipython3
:tags: [skip-execution]

def analyze_failures(results: list[GenerationResult]) -> dict:
    """
    Analiza los fallos para identificar patrones.

    Returns:
        Dict con categorías de errores y conteos
    """
    failures = [r for r in results if not r.success]

    error_categories = {
        "compile_error": 0,
        "missing_mask": 0,
        "incorrect_offsets": 0,
        "wrong_output": 0,
        "other": 0
    }

    for fail in failures:
        if fail.error_msg is None:
            continue

        msg = fail.error_msg.lower()
        if "compile" in msg or "syntax" in msg:
            error_categories["compile_error"] += 1
        elif "mask" in msg:
            error_categories["missing_mask"] += 1
        elif "offset" in msg or "index" in msg:
            error_categories["incorrect_offsets"] += 1
        elif "incorrect" in msg or "mismatch" in msg:
            error_categories["wrong_output"] += 1
        else:
            error_categories["other"] += 1

    return error_categories

def iterate_on_prompt(problems: list[KernelSpec], llm_fn: Callable,
                     system_prompt: str, few_shot: str, iterations: int = 3):
    """
    Itera sobre el prompt para mejorar pass rate.

    Args:
        problems: Lista de problemas de KernelBench
        llm_fn: Función LLM
        system_prompt: System prompt inicial
        few_shot: Few-shot inicial
        iterations: Número de iteraciones
    """
    current_system = system_prompt
    current_few_shot = few_shot

    for i in range(iterations):
        print(f"\n=== ITERACIÓN {i+1} ===")

        # Evaluar
        results = []
        for problem in problems:
            code = generate_kernel(problem, llm_fn, current_system, current_few_shot)
            result = evaluate_kernel(code, problem)
            results.append(result)

        # Calcular pass rate
        pass_rate = sum(r.success for r in results) / len(results)
        print(f"Pass rate: {pass_rate*100:.1f}%")

        # Analizar fallos
        error_cats = analyze_failures(results)
        print(f"Errores por categoría: {error_cats}")

        # Aquí el equipo decide cómo modificar el prompt
        # Por ejemplo, si hay muchos errores de máscaras:
        if error_cats["missing_mask"] > len(problems) * 0.2:
            print("⚠️ Detectados muchos errores de máscaras. Reforzar regla en system prompt.")
            # current_system += "\n\nRECUERDA: SIEMPRE usar mask= en tl.load y tl.store."

        # En práctica, esto requiere intervención manual o un agente que modifique el prompt
        # Para esta demo, solo reportamos

print("Función de iteración definida")
```

### Medir Pass Rate en KernelBench L1

```{code-cell} ipython3
:tags: [skip-execution]

def evaluate_pass_rate(problems: list[KernelSpec], llm_fn: Callable,
                       system_prompt: str, few_shot: str) -> tuple[float, list[GenerationResult]]:
    """
    Evalúa pass rate en un conjunto de problemas.

    Args:
        problems: Lista de KernelSpec
        llm_fn: Función LLM
        system_prompt: System prompt
        few_shot: Few-shot examples

    Returns:
        (pass_rate, lista de resultados)
    """
    results = []

    for problem in problems:
        print(f"Generando kernel para: {problem.name}")
        code = generate_kernel(problem, llm_fn, system_prompt, few_shot)
        result = evaluate_kernel(code, problem)
        results.append(result)

        status = "✓" if result.success else "✗"
        print(f"  {status} {problem.name}: {result.error_msg or 'OK'}")

    successes = sum(r.success for r in results)
    pass_rate = successes / len(results)

    print(f"\n=== RESUMEN ===")
    print(f"Pass rate: {pass_rate*100:.1f}% ({successes}/{len(results)})")

    return pass_rate, results

# Ejemplo con KernelBench L1 (simplificado)
kernelbench_l1 = [
    KernelSpec(
        name="add",
        operation="Suma element-wise",
        formula="out[i] = x[i] + y[i]",
        input_shapes={"x": ("N",), "y": ("N",)},
        output_shape=("N",),
        dtype=torch.float32,
        reference_fn=lambda x, y: x + y
    ),
    KernelSpec(
        name="relu",
        operation="ReLU",
        formula="out[i] = max(0, x[i])",
        input_shapes={"x": ("N",)},
        output_shape=("N",),
        dtype=torch.float32,
        reference_fn=lambda x: torch.relu(x)
    ),
    KernelSpec(
        name="sigmoid",
        operation="Sigmoid",
        formula="out[i] = 1 / (1 + exp(-x[i]))",
        input_shapes={"x": ("N",)},
        output_shape=("N",),
        dtype=torch.float32,
        reference_fn=lambda x: torch.sigmoid(x)
    ),
    KernelSpec(
        name="mul_scalar",
        operation="Multiplicación por escalar",
        formula="out[i] = x[i] * alpha",
        input_shapes={"x": ("N",)},
        output_shape=("N",),
        dtype=torch.float32,
        reference_fn=lambda x, alpha=2.0: x * alpha
    ),
    KernelSpec(
        name="square",
        operation="Cuadrado element-wise",
        formula="out[i] = x[i]^2",
        input_shapes={"x": ("N",)},
        output_shape=("N",),
        dtype=torch.float32,
        reference_fn=lambda x: x ** 2
    ),
]

# Descomentar para ejecutar (requiere GPU + backend LLM)
# pass_rate, results = evaluate_pass_rate(
#     kernelbench_l1,
#     my_llm_fn,  # Tu función LLM
#     MY_SYSTEM_PROMPT,
#     MY_FEW_SHOT_EXAMPLES
# )
```

## Actividad de Trabajo — Bloque 2 (25 minutos)

**Ejercicio 1: Medir baseline y diagnosticar fallos (15 min)**

Evalúa tu prompt del Bloque 1 sobre los primeros **5 problemas L1** del dataset.
Completa la tabla clasificando cada error por categoría:

| Problem ID | Compiled | Executed | Correct | Speedup | Categoría de error |
|------------|----------|----------|---------|---------|---------------------|
| ... | ✓/✗ | ✓/✗ | ✓/✗ | Nx | compile / mask / offset / logic |

Categorías:
- **compile**: error de sintaxis Python/Triton
- **mask**: falta `mask=` en `tl.load` o `tl.store`
- **offset**: `pid` mal calculado o `tl.arange` incorrecto
- **logic**: kernel corre pero resultado incorrecto (error semántico)

¿Cuál categoría es más frecuente?

**Ejercicio 2: Una iteración de mejora dirigida (10 min)**

Basándote en el error más frecuente del Ejercicio 1:

- **Si domina mask** → añade a `MY_SYSTEM_PROMPT` una regla explícita + un
  contraejemplo nuevo con `# ❌ tl.load sin máscara`.
- **Si domina offset** → añade un few-shot con el cálculo de offsets comentado
  paso a paso (`# paso 1: pid`, `# paso 2: arange(0, BLOCK_SIZE)`, etc.).
- **Si domina logic** → añade un CoT: "antes de generar código, escribe los
  pasos matemáticos de la operación numerados".

Re-evalúa sobre los mismos 5 problemas. ¿Mejoró el pass rate?

```{code-cell} ipython3
:tags: [skip-execution]

# PLANTILLA PARA ALUMNOS

# 1. Medir baseline
print("=== BASELINE ===")
# pass_rate_baseline, results_baseline = evaluate_pass_rate(
#     kernelbench_l1,
#     my_llm_fn,
#     MY_SYSTEM_PROMPT,
#     MY_FEW_SHOT_EXAMPLES
# )

# 2. Analizar fallos
# error_cats = analyze_failures(results_baseline)
# print(f"Errores: {error_cats}")

# 3. Mejorar prompt
# TODO: Basado en los errores, modifica MY_SYSTEM_PROMPT y MY_FEW_SHOT_EXAMPLES

# MY_SYSTEM_PROMPT_V2 = """
# ... mejoras basadas en análisis ...
# """

# MY_FEW_SHOT_EXAMPLES_V2 = """
# ... añadir ejemplos que cubran casos problemáticos ...
# """

# 4. Re-evaluar
# print("\n=== ITERACIÓN 2 ===")
# pass_rate_v2, results_v2 = evaluate_pass_rate(
#     kernelbench_l1,
#     my_llm_fn,
#     MY_SYSTEM_PROMPT_V2,
#     MY_FEW_SHOT_EXAMPLES_V2
# )

# 5. Continuar iterando...
```

```{admonition} Objetivo de Pass Rate
:class: tip

**Meta**: >50% pass rate en KernelBench L1 con iteración manual del prompt.

**Estrategias**:
- Si muchos errores de compilación: simplificar el few-shot, enfocarse en sintaxis básica
- Si muchos errores de máscaras: añadir regla explícita en system prompt + contraejemplo
- Si muchos errores de offsets: añadir ejemplo con cálculo de offsets comentado paso a paso
- Si temperatura alta: reducir a 0.1-0.2 para código más conservador
```

## Cierre de Sesión *(10 min)*

```{admonition} ✅ Verifica tu Comprensión
:class: note
**Pregunta 1**: Tu pipeline genera un kernel que compila y ejecuta, pero
`max_abs_diff = 0.002`. ¿Cuál es la causa más probable: (a) prompt mal diseñado,
(b) error de razonamiento del LLM, o (c) precisión numérica FP32? ¿Qué añadirías
al few-shot para diagnosticar cada causa?
<details><summary>Respuesta</summary>
Para elementwise simple, <code>max_abs_diff=0.002</code> apunta a (b): el LLM
implementó la fórmula ligeramente incorrecta (ej. orden de operaciones en GELU,
<code>tl.maximum</code> vs comparación manual). La causa (c) solo importa en
reducciones largas. Un CoT que pida "escribe los pasos matemáticos antes de
generar código" reduce errores de tipo (b).
</details>

**Pregunta 2**: Si eliminas la regla 3 del `TRITON_SYSTEM_PROMPT` ("Calcula
offsets como `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`"), ¿en qué paso del
pipeline `evaluate_kernel()` fallará el kernel generado con más probabilidad?
<details><summary>Respuesta</summary>
En <strong>step3_correctness</strong>, no en compilación. El LLM puede inventar
una fórmula de offset sintácticamente válida (compila + ejecuta) pero con
índices incorrectos → <code>allclose</code> falla. Ejemplo:
<code>offsets = tl.arange(pid, pid + BLOCK_SIZE)</code> compila pero es incorrecto.
</details>
```

### Para Pensar

> *En S7 construiremos un agente que itera automáticamente sobre el prompt
> basándose en los errores del pipeline. ¿Qué riesgo existe si el agente converge
> a un prompt que pasa el benchmark pero genera kernels lentos en producción?
> ¿Cómo diseñarías el reward signal para evitar ese overfitting?*

---

## Resumen

En esta sesión aplicaste técnicas de P1 al dominio GPU:

1. **Contexto para kernels GPU**:
   - Spec de operación (fórmula, shapes, dtype)
   - Restricciones de memoria (BLOCK_SIZE, máscaras)
   - Patrón base (elementwise, reduction, etc.)

2. **Diseño de prompts especializados**:
   - System prompt con reglas Triton
   - Few-shot con ejemplos correctos + contraejemplos
   - Chain-of-Thought para kernels complejos

3. **Pipeline completo**:
   - Prompt → LLM → código Triton
   - Compile → test → benchmark (reutilizando S5)
   - Medir pass rate

4. **Iteración**:
   - Analizar fallos (compilación, máscaras, offsets)
   - Mejorar prompt basado en errores
   - Re-evaluar hasta >50% pass rate en L1

**Limitación**: La iteración fue **manual**. En S7 veremos cómo **automatizar** este proceso con agentes.

---

## Tarea para Casa

### Ejercicio 1: Pass rate en KernelBench L1 completo (★★★)

Amplía el Ejercicio 1 de Bloque 2 a los **20 primeros problemas L1** y genera
un reporte completo:

1. Ejecuta `evaluate_pass_rate()` sobre los 20 problemas.
2. Clasifica cada fallo en la tabla por categoría (compile / mask / offset / logic).
3. Implementa `analyze_failures(results)` que devuelva counts por categoría y el
   `error_message` más frecuente de cada una.
4. Realiza **2 iteraciones de mejora** del prompt basándote en el análisis.
5. Grafica el progreso: `pass_rate` vs `iteration` (versión 0, 1, 2).

**Entregable**: tabla de resultados + gráfica de progreso.

### Ejercicio 2: Chain-of-Thought para kernel GELU fused (★★☆)

Diseña un prompt CoT específico para generar el kernel GELU fused:
`gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

1. El prompt debe pedir al LLM que razone: tipo → inputs/outputs → pasos de
   cómputo numerados → restricciones de memoria, **antes** de generar código.
2. Compara pass rate del prompt CoT vs tu mejor prompt estándar de la sesión.
3. ¿En qué categoría de error falla más el prompt estándar para GELU? ¿Y el CoT?

---

## Referencias

### Prompt Engineering para Código

1. **Chen et al. (2021)**: "Evaluating Large Language Models Trained on Code" (Codex paper)
   - https://arxiv.org/abs/2107.03374

2. **Wei et al. (2022)**: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
   - https://arxiv.org/abs/2201.11903

3. **Brown et al. (2020)**: "Language Models are Few-Shot Learners" (GPT-3)
   - https://arxiv.org/abs/2005.14165

### Generación de Código GPU

4. **Phothilimthana et al. (2023)**: "Automatic Kernel Synthesis with LLMs"
   - Técnicas específicas para generar kernels CUDA/Triton

5. **OpenAI Codex Documentation**:
   - https://platform.openai.com/docs/guides/code

6. **Anthropic Claude for Code**:
   - https://docs.anthropic.com/claude/docs/code-generation

### Triton y GPU Programming

7. **Triton Documentation**: https://triton-lang.org
   - Guías de best practices para escribir kernels

8. **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
   - Conceptos de parallelismo, memoria, sincronización

### Herramientas de Evaluación

9. **HumanEval**: Benchmark para evaluar generación de código
   - https://github.com/openai/human-eval

10. **MBPP** (Mostly Basic Python Problems): Dataset de problemas de programación
    - https://github.com/google-research/google-research/tree/master/mbpp

---

**Próxima sesión**: S7 - Agentes Iterativos para Kernel Synthesis

Automatizaremos la iteración del prompt con agentes que:
- Analizan errores automáticamente
- Modifican el prompt basándose en fallos
- Re-generan y evalúan en bucle (ReAct)
- Convergen a >80% pass rate en L1
