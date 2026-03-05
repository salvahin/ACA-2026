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

# Prompt Engineering para Generación de Código

> **Módulo:** Project 1 - LLM & Constrained Generation
> **Semana:** 3
> **Tiempo de lectura:** ~25 minutos

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Diseñar system prompts efectivos que definan el rol y expertise del modelo
- Aplicar técnicas de few-shot prompting con el número óptimo de ejemplos (1-3)
- Implementar Chain-of-Thought (CoT) para desglosar problemas complejos
- Construir prompts multi-componente (system + examples + CoT + constraints)
- Analizar y mejorar prompts para evitar errores comunes (ambigüedad, inconsistencia)
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Diseñar el sistema completo de prompts para KernelAgent, incluyendo system prompts para cada etapa (análisis, planificación, generación), templates de few-shot learning con ejemplos de kernels Triton, y cadenas de razonamiento (CoT) para decisiones de optimización.
```

---

## Introducción

Has aprendido sobre los modelos de lenguaje especializados en código como CodeLlama y StarCoder. Ahora viene la pregunta crucial: **¿cómo comunicarte efectivamente con estos modelos para obtener el código que necesitas?**

Prompt engineering es el arte y ciencia de diseñar instrucciones que guíen a los LLMs hacia las respuestas deseadas. Para generación de código, esto es especialmente crítico: un prompt mal diseñado puede producir código sintácticamente correcto pero funcionalmente inútil, mientras que un prompt bien diseñado puede generar soluciones elegantes y eficientes.

---

## Anatomía de un Prompt para Código

Un prompt efectivo para generación de código tiene varios componentes:

```{code-cell} ipython3
# Visualización de componentes del prompt
import plotly.graph_objects as go

fig = go.Figure()

componentes = [
    ('System Prompt', 0.9, '#FF6B6B', 'Define el rol y comportamiento'),
    ('Context', 0.7, '#4ECDC4', 'Información relevante'),
    ('Few-shot Examples', 0.5, '#45B7D1', 'Ejemplos de entrada/salida'),
    ('User Query', 0.3, '#FFE66D', 'La pregunta del usuario'),
    ('Output Format', 0.1, '#95E1A3', 'Cómo debe responder')
]

for nombre, y, color, desc in componentes:
    fig.add_shape(type="rect", x0=0.1, y0=y-0.08, x1=0.9, y1=y+0.08,
                  fillcolor=color, line=dict(color='black'))
    fig.add_annotation(x=0.5, y=y, text=f"<b>{nombre}</b><br>{desc}",
                       showarrow=False, font=dict(size=11))

fig.update_layout(title="Anatomía de un Prompt Efectivo",
                  xaxis=dict(visible=False), yaxis=dict(visible=False),
                  height=450, showlegend=False)
fig.show()
```

### 1. System Prompt

El system prompt establece el contexto y las reglas del juego. Para KernelAgent, podría verse así:

```{code-cell} ipython3
# Ejemplo de construcción de system prompts para generación de código
class PromptTemplate:
    """Template para construir prompts estructurados"""

    def __init__(self, role, rules, language="Python"):
        self.role = role
        self.rules = rules
        self.language = language

    def build_system_prompt(self):
        """Construye el system prompt"""
        prompt = f"You are {self.role}.\n"
        prompt += f"You write code in {self.language}.\n\n"
        prompt += "Follow these rules:\n"
        for i, rule in enumerate(self.rules, 1):
            prompt += f"{i}. {rule}\n"
        return prompt

# Ejemplo para kernels Triton
triton_template = PromptTemplate(
    role="an expert GPU kernel developer specializing in Triton",
    rules=[
        "Write efficient, correct kernels following best practices",
        "Always include proper bounds checking with masks",
        "Use descriptive variable names that indicate tensor dimensions",
        "Add comments explaining the parallelization strategy"
    ],
    language="Triton"
)

print(triton_template.build_system_prompt())
```

> 💡 **Concepto clave:** El system prompt define la "personalidad" y expertise del modelo. Para código, especifica el lenguaje, framework, y estándares de calidad esperados.

### 2. Contexto del Problema

Proporciona información relevante sobre el problema:

```
# Task: Implement a fused softmax kernel in Triton
# Input: tensor of shape (batch_size, seq_len, hidden_dim)
# Output: softmax applied along the last dimension
# Constraints: Must handle arbitrary hidden_dim sizes
```

:::{figure} diagrams/prompt_components.png
:name: fig-prompt-components
:alt: Diagrama mostrando los componentes de un prompt efectivo para generación de código
:align: center
:width: 90%

**Figura 1:** Componentes de un prompt para código - System, Context, Examples, y CoT.
:::

### 3. Ejemplos (Few-shot)

Incluir ejemplos ayuda al modelo a entender el formato esperado:

```python
# Example: Simple vector addition in Triton
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

---

## Few-Shot Prompting para Código

Few-shot prompting es particularmente poderoso para código porque:

1. **Establece patrones sintácticos**: El modelo aprende el estilo de código esperado
2. **Demuestra convenciones**: Nombres de variables, estructura de funciones
3. **Muestra el nivel de detalle**: Cuántos comentarios, qué tipo de error handling

### Selección de Ejemplos

No todos los ejemplos son igual de útiles. Considera:

```{code-cell} ipython3
# Simulación de Few-Shot Prompting para generación de código
import re

class FewShotPrompt:
    """Construye prompts con ejemplos few-shot"""

    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.examples = []

    def add_example(self, description, code):
        """Agrega un ejemplo al prompt"""
        self.examples.append({
            "description": description,
            "code": code
        })

    def build_prompt(self, task_description):
        """Construye el prompt completo con ejemplos"""
        prompt = self.system_prompt + "\n\n"
        prompt += "Here are some examples:\n\n"

        for i, example in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Task: {example['description']}\n"
            prompt += f"Code:\n{example['code']}\n\n"

        prompt += f"Now, complete this task:\n{task_description}\n"
        return prompt

# Ejemplo: Few-shot para funciones Python
few_shot = FewShotPrompt("You are a Python expert.")

# MAL: Ejemplo demasiado simple
few_shot.add_example(
    description="Add two numbers",
    code="""def add(a, b):
    return a + b"""
)

# BIEN: Ejemplo con patrones útiles
few_shot.add_example(
    description="Calculate moving average with error handling",
    code="""def moving_average(data, window_size):
    \"\"\"Calculate moving average with given window size.

    Args:
        data: List of numbers
        window_size: Size of the moving window

    Returns:
        List of moving averages
    \"\"\"
    if not data or window_size <= 0:
        raise ValueError("Invalid input")

    if window_size > len(data):
        window_size = len(data)

    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        avg = sum(window) / window_size
        result.append(avg)

    return result"""
)

task = "Write a function to calculate median of a list"
print(few_shot.build_prompt(task))
print("\n" + "="*60)
print(f"Total examples: {len(few_shot.examples)}")
print(f"Prompt length: {len(few_shot.build_prompt(task))} characters")
```

### Número Óptimo de Ejemplos

- **0 ejemplos (zero-shot)**: Útil para tareas simples o bien definidas
- **1-3 ejemplos**: Típicamente óptimo para generación de código
- **>5 ejemplos**: Puede confundir o llenar el contexto innecesariamente

**Criterios CoT vs few-shot:**
- Few-shot: ejemplos similares disponibles, patrón claro, formato específico
- CoT: problema complejo, múltiples pasos, trade-offs de diseño
- Combina ambos para kernels GPU: patrón estructural + razonamiento de optimización

```{code-cell} ipython3
# Experimento: Impacto del número de ejemplos en el tamaño del prompt
import matplotlib.pyplot as plt
import numpy as np

def estimate_prompt_tokens(num_examples, avg_example_tokens=150):
    """Estima tokens del prompt basado en número de ejemplos"""
    system_prompt_tokens = 50
    task_description_tokens = 30
    example_tokens = num_examples * avg_example_tokens
    return system_prompt_tokens + task_description_tokens + example_tokens

# Simular diferentes números de ejemplos
num_examples_range = range(0, 11)
prompt_sizes = [estimate_prompt_tokens(n) for n in num_examples_range]

# Simular efectividad (simplificado)
# Asumimos que la efectividad aumenta con ejemplos pero se satura
effectiveness = [min(50 + n * 15, 95) for n in num_examples_range]

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Tamaño del prompt
ax1.plot(num_examples_range, prompt_sizes, marker='o', color='blue')
ax1.axvline(x=3, color='red', linestyle='--', label='Óptimo (1-3 ejemplos)')
ax1.set_xlabel('Número de Ejemplos')
ax1.set_ylabel('Tokens en el Prompt')
ax1.set_title('Tamaño del Prompt vs Número de Ejemplos')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Efectividad vs tamaño
ax2.plot(num_examples_range, effectiveness, marker='s', color='green', label='Efectividad')
ax2.axvline(x=3, color='red', linestyle='--', label='Óptimo')
ax2.set_xlabel('Número de Ejemplos')
ax2.set_ylabel('Efectividad (%)')
ax2.set_title('Efectividad vs Número de Ejemplos')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Análisis del número óptimo de ejemplos:")
print("="*50)
for n in [0, 1, 3, 5, 10]:
    tokens = estimate_prompt_tokens(n)
    eff = min(50 + n * 15, 95)
    efficiency = eff / tokens if tokens > 0 else 0
    print(f"{n} ejemplos: {tokens} tokens, {eff}% efectividad, "
          f"{efficiency:.4f} efectividad/token")
```

---

## Chain-of-Thought para Código

Chain-of-Thought (CoT) pide al modelo que "piense paso a paso" antes de generar código. Esto es especialmente útil para:

- Algoritmos complejos
- Optimización de kernels GPU
- Debugging de código existente

### Ejemplo: CoT para Kernel de Softmax

```{code-cell} ipython3
# Simulación de Chain-of-Thought (CoT) para generación de código
class ChainOfThoughtPrompt:
    """Genera prompts con razonamiento paso a paso"""

    def __init__(self):
        self.reasoning_steps = []
        self.final_task = None

    def add_reasoning_step(self, step_number, title, details):
        """Agrega un paso de razonamiento"""
        self.reasoning_steps.append({
            "number": step_number,
            "title": title,
            "details": details
        })

    def set_task(self, task):
        """Define la tarea final"""
        self.final_task = task

    def build_prompt(self):
        """Construye el prompt CoT completo"""
        prompt = "Let's think step by step:\n\n"

        for step in self.reasoning_steps:
            prompt += f"{step['number']}. **{step['title']}**:\n"
            for detail in step['details']:
                prompt += f"   - {detail}\n"
            prompt += "\n"

        if self.final_task:
            prompt += f"Now, {self.final_task}\n"

        return prompt

# Ejemplo: CoT para implementar softmax
cot = ChainOfThoughtPrompt()

cot.add_reasoning_step(
    1, "Input Analysis",
    [
        "We have a tensor of shape (batch, seq, hidden)",
        "We need to apply softmax along the last dimension (hidden)",
        "Each (batch, seq) position is independent"
    ]
)

cot.add_reasoning_step(
    2, "Algorithm Design",
    [
        "For numerical stability: subtract max before exp",
        "Compute: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))",
        "Requires two passes: one for max, one for sum"
    ]
)

cot.add_reasoning_step(
    3, "Parallelization Strategy",
    [
        "Each program handles one (batch, seq) position",
        "Within a program, process hidden_dim elements in blocks",
        "Use reduction operations for max and sum"
    ]
)

cot.add_reasoning_step(
    4, "Memory Access Pattern",
    [
        "Load contiguous elements along hidden_dim",
        "Use masks for non-multiple-of-BLOCK_SIZE dimensions",
        "Coalesce memory accesses for performance"
    ]
)

cot.set_task("generate the Triton kernel implementation")

print(cot.build_prompt())
print("\n" + "="*60)
print("Ventajas del CoT:")
print("1. Razonamiento explícito y verificable")
print("2. Ayuda al modelo a considerar edge cases")
print("3. Facilita el debugging del código generado")
print("4. Mejora la calidad en problemas complejos")
```

> 💡 **Concepto clave:** CoT no solo mejora la calidad del código generado, sino que también hace el proceso más interpretable y debuggeable.

---

## Prompts en KernelAgent

KernelAgent utiliza un sistema de prompts estructurado en su pipeline. Analicemos sus componentes:

### Stage 1: Análisis del Problema

```python
# Prompt para entender el problema de KernelBench
"""
Analyze the following PyTorch reference implementation:
{pytorch_code}

Identify:
1. Input tensor shapes and dtypes
2. Mathematical operations performed
3. Output tensor characteristics
4. Potential optimization opportunities for GPU
"""
```

### Stage 2: Planificación

```python
# Prompt para diseñar la estrategia
"""
Design a Triton kernel strategy for:
{problem_description}

Consider:
- Block sizes and tiling strategy
- Memory access patterns
- Reduction operations needed
- Autotuning parameters
"""
```

### Stage 3: Generación

```python
# Prompt para generar código
"""
Generate a Triton kernel implementing:
{planned_strategy}

Requirements:
- Use @triton.jit decorator
- Include proper masks for bounds checking
- Follow the API: {api_signature}
"""
```

---

## Errores Comunes en Prompts para Código

### 1. Ambigüedad en Especificaciones

```{code-cell} ipython3
# Comparación de prompts: Ambiguos vs Específicos
class PromptQualityAnalyzer:
    """Analiza la calidad de prompts para generación de código"""

    @staticmethod
    def analyze_prompt(prompt):
        """Analiza un prompt y retorna métricas de calidad"""
        metrics = {
            "length": len(prompt),
            "has_types": bool(any(t in prompt.lower() for t in ["int", "float", "tensor", "array"])),
            "has_dimensions": bool(any(d in prompt for d in ["x", "M", "N", "K", "shape"])),
            "has_constraints": bool(any(c in prompt.lower() for c in ["constraint", "requirement", "must", "handle"])),
            "specificity_score": 0
        }

        # Calcular score de especificidad
        score = 0
        if metrics["has_types"]:
            score += 25
        if metrics["has_dimensions"]:
            score += 25
        if metrics["has_constraints"]:
            score += 25
        if metrics["length"] > 50:
            score += 25

        metrics["specificity_score"] = score
        return metrics

# Ejemplos de prompts
prompts = {
    "Ambiguo (MAL)": "Write a fast matrix multiplication",
    "Específico (BIEN)": """Write a Triton kernel for matrix multiplication of tensors
A (M x K) and B (K x N), producing C (M x N).
Use tiling with configurable BLOCK_M, BLOCK_N, BLOCK_K.
Handle non-multiples of block size with masking."""
}

print("Análisis de Calidad de Prompts")
print("="*60)

for label, prompt in prompts.items():
    print(f"\n{label}:")
    print(f"Prompt: {prompt[:50]}...")
    metrics = PromptQualityAnalyzer.analyze_prompt(prompt)
    print(f"Especificidad: {metrics['specificity_score']}/100")
    print(f"Tiene tipos: {metrics['has_types']}")
    print(f"Tiene dimensiones: {metrics['has_dimensions']}")
    print(f"Tiene restricciones: {metrics['has_constraints']}")
```

### 2. Falta de Contexto de Restricciones

```{code-cell} ipython3
# Ejemplo: Importancia de especificar restricciones
def generate_prompt_with_constraints(task, constraints=None):
    """Genera prompt con restricciones explícitas"""
    prompt = f"Task: {task}\n\n"

    if constraints:
        prompt += "Constraints:\n"
        for i, constraint in enumerate(constraints, 1):
            prompt += f"{i}. {constraint}\n"
    else:
        prompt += "No constraints specified (riesgo de solución incompleta)\n"

    return prompt

# Sin restricciones (MAL)
prompt_bad = generate_prompt_with_constraints("Implement softmax in Triton")

# Con restricciones (BIEN)
prompt_good = generate_prompt_with_constraints(
    "Implement softmax in Triton",
    constraints=[
        "Reduction dimension may not be a multiple of block size",
        "Handle edge cases with proper masking",
        "Ensure numerical stability by subtracting max",
        "Support tensors up to 4 dimensions"
    ]
)

print("SIN restricciones:")
print(prompt_bad)
print("\n" + "="*60)
print("\nCON restricciones:")
print(prompt_good)
```

### 3. Ejemplos Inconsistentes

Todos los ejemplos deben seguir el mismo estilo y convenciones. Mezclar estilos confunde al modelo.

```{code-cell} ipython3
# Detector de inconsistencias en ejemplos
import re

def check_example_consistency(examples):
    """Verifica consistencia entre ejemplos de código"""
    issues = []

    # Extraer características de cada ejemplo
    naming_styles = []
    has_docstrings = []
    indentation_types = []

    for i, example in enumerate(examples):
        # Detectar estilo de nombres (snake_case vs camelCase)
        if re.search(r'[a-z]+_[a-z]+', example):
            naming_styles.append(('snake_case', i))
        elif re.search(r'[a-z]+[A-Z][a-z]+', example):
            naming_styles.append(('camelCase', i))

        # Detectar docstrings
        has_docstrings.append(('"""' in example or "'''" in example, i))

        # Detectar tipo de indentación
        if '\t' in example:
            indentation_types.append(('tabs', i))
        elif '    ' in example:
            indentation_types.append(('spaces', i))

    # Verificar consistencia
    if len(set(style for style, _ in naming_styles)) > 1:
        issues.append("Estilos de nombres inconsistentes: " +
                     ", ".join(f"Ej{i}: {style}" for style, i in naming_styles))

    docstring_presence = [has for has, _ in has_docstrings]
    if docstring_presence and not all(docstring_presence):
        issues.append("Algunos ejemplos tienen docstrings, otros no")

    if len(set(indent for indent, _ in indentation_types)) > 1:
        issues.append("Indentación inconsistente: " +
                     ", ".join(f"Ej{i}: {indent}" for indent, i in indentation_types))

    return issues

# Ejemplos inconsistentes (MAL)
bad_examples = [
    """def calculate_mean(values):
    \"\"\"Calculate mean value\"\"\"
    return sum(values) / len(values)""",

    """def calculateMedian(vals):
\treturn sorted(vals)[len(vals)//2]"""  # camelCase, tabs, sin docstring
]

# Ejemplos consistentes (BIEN)
good_examples = [
    """def calculate_mean(values):
    \"\"\"Calculate mean value\"\"\"
    return sum(values) / len(values)""",

    """def calculate_median(values):
    \"\"\"Calculate median value\"\"\"
    sorted_vals = sorted(values)
    return sorted_vals[len(values) // 2]"""
]

print("Ejemplos INCONSISTENTES:")
issues = check_example_consistency(bad_examples)
for issue in issues:
    print(f"  - {issue}")

print("\nEjemplos CONSISTENTES:")
issues = check_example_consistency(good_examples)
if issues:
    for issue in issues:
        print(f"  - {issue}")
else:
    print("  ✓ Todos los ejemplos son consistentes")
```

---

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- System prompts definen el "rol" del modelo (expertise, lenguaje, reglas) y son fundamentales para generación consistente
- Few-shot learning con 1-3 ejemplos es óptimo: más ejemplos no mejoran proporcionalmente y consumen contexto
- Chain-of-Thought (CoT) desglos problemas complejos en pasos de razonamiento explícitos, mejorando calidad y debuggeability
- Errores comunes: ambigüedad en especificaciones, falta de restricciones, ejemplos inconsistentes
- Framework completo combina: system prompt + few-shot examples + CoT + constraints para máxima efectividad

**Siguiente paso:** En la próxima lectura exploraremos **XGrammar y Constrained Decoding**, donde aprenderás cómo forzar al modelo a generar código sintácticamente válido mediante gramáticas formales, combinando el arte de los prompts con la ciencia de las restricciones estructuradas.
```

```{admonition} Verifica tu comprensión
:class: note
1. ¿Por qué 1-3 ejemplos few-shot es óptimo? ¿Qué pasa si usas 10 ejemplos?
2. Diseña un system prompt para un modelo que debe generar código SQL optimizado
3. ¿Cuándo usarías CoT en lugar de few-shot prompting? Proporciona dos escenarios específicos
4. En KernelAgent, ¿qué información crítica debe incluirse en el prompt para generar un kernel de multiplicación de matrices?
```

```{code-cell} ipython3
# Framework completo de Prompt Engineering para generación de código
class PromptEngineeringFramework:
    """Framework completo para construir prompts efectivos"""

    def __init__(self):
        self.system_prompt = ""
        self.examples = []
        self.cot_steps = []
        self.constraints = []

    def set_system_prompt(self, role, language, rules):
        """Configura el system prompt"""
        prompt = f"You are {role}.\n"
        prompt += f"You write code in {language}.\n\n"
        prompt += "Follow these rules:\n"
        for i, rule in enumerate(rules, 1):
            prompt += f"{i}. {rule}\n"
        self.system_prompt = prompt
        return self

    def add_few_shot_example(self, description, code):
        """Agrega un ejemplo few-shot"""
        self.examples.append({"description": description, "code": code})
        return self

    def add_cot_step(self, step_num, title, details):
        """Agrega un paso de Chain-of-Thought"""
        self.cot_steps.append({
            "number": step_num,
            "title": title,
            "details": details
        })
        return self

    def add_constraint(self, constraint):
        """Agrega una restricción"""
        self.constraints.append(constraint)
        return self

    def build_complete_prompt(self, task):
        """Construye el prompt completo"""
        prompt = self.system_prompt + "\n\n"

        # Few-shot examples
        if self.examples:
            prompt += "EXAMPLES:\n\n"
            for i, ex in enumerate(self.examples, 1):
                prompt += f"Example {i}: {ex['description']}\n"
                prompt += f"```\n{ex['code']}\n```\n\n"

        # Chain-of-Thought
        if self.cot_steps:
            prompt += "REASONING STEPS:\n\n"
            for step in self.cot_steps:
                prompt += f"{step['number']}. {step['title']}:\n"
                for detail in step['details']:
                    prompt += f"   - {detail}\n"
                prompt += "\n"

        # Constraints
        if self.constraints:
            prompt += "CONSTRAINTS:\n"
            for i, constraint in enumerate(self.constraints, 1):
                prompt += f"{i}. {constraint}\n"
            prompt += "\n"

        # Task
        prompt += f"TASK:\n{task}\n"

        return prompt

    def analyze_prompt_quality(self, prompt):
        """Analiza la calidad del prompt generado"""
        metrics = {
            "length": len(prompt),
            "has_system": "You are" in prompt,
            "has_examples": len(self.examples) > 0,
            "has_cot": len(self.cot_steps) > 0,
            "has_constraints": len(self.constraints) > 0,
            "num_examples": len(self.examples),
            "quality_score": 0
        }

        # Calcular score
        score = 0
        if metrics["has_system"]:
            score += 25
        if metrics["has_examples"]:
            score += 25
        if metrics["has_cot"]:
            score += 25
        if metrics["has_constraints"]:
            score += 25

        metrics["quality_score"] = score
        return metrics

# Ejemplo de uso: Crear prompt para kernel de Triton
print("EJEMPLO: Construir prompt para kernel de softmax en Triton")
print("="*70)

framework = PromptEngineeringFramework()

# 1. System prompt
framework.set_system_prompt(
    role="an expert GPU kernel developer specializing in Triton",
    language="Triton",
    rules=[
        "Write efficient, correct kernels following best practices",
        "Always include proper bounds checking with masks",
        "Use descriptive variable names that indicate tensor dimensions",
        "Add comments explaining the parallelization strategy"
    ]
)

# 2. Few-shot example
framework.add_few_shot_example(
    description="Simple vector addition with masking",
    code="""@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)"""
)

# 3. Chain-of-Thought
framework.add_cot_step(
    1, "Input Analysis",
    ["Tensor shape: (batch, seq, hidden)",
     "Apply softmax along last dimension (hidden)",
     "Each (batch, seq) position is independent"]
)

framework.add_cot_step(
    2, "Algorithm Design",
    ["For numerical stability: subtract max before exp",
     "Compute: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))",
     "Requires two passes: one for max, one for sum"]
)

framework.add_cot_step(
    3, "Parallelization Strategy",
    ["Each program handles one (batch, seq) position",
     "Within a program, process hidden_dim elements in blocks",
     "Use reduction operations for max and sum"]
)

# 4. Constraints
framework.add_constraint("Reduction dimension may not be a multiple of block size")
framework.add_constraint("Handle edge cases with proper masking")
framework.add_constraint("Ensure numerical stability")

# Build prompt
task = "Implement a fused softmax kernel in Triton"
complete_prompt = framework.build_complete_prompt(task)

print("\nCOMPLETE PROMPT:")
print("="*70)
print(complete_prompt)

# Analyze quality
print("\n" + "="*70)
print("PROMPT QUALITY ANALYSIS:")
print("="*70)
metrics = framework.analyze_prompt_quality(complete_prompt)

for key, value in metrics.items():
    if key != "quality_score":
        print(f"  {key}: {value}")

print(f"\n  Overall Quality Score: {metrics['quality_score']}/100")

if metrics['quality_score'] >= 75:
    print("\n  ✓ EXCELLENT: Prompt tiene todos los componentes clave")
elif metrics['quality_score'] >= 50:
    print("\n  ⚠ GOOD: Prompt es funcional pero podría mejorarse")
else:
    print("\n  ✗ POOR: Prompt necesita más contexto y estructura")

print("\n" + "="*70)
print("MEJORES PRÁCTICAS:")
print("="*70)
print("""
1. System Prompt:
   - Define rol y expertise del modelo
   - Especifica lenguaje y framework
   - Lista reglas y estándares de calidad

2. Few-Shot Examples:
   - 1-3 ejemplos es óptimo
   - Ejemplos deben mostrar patrones relevantes
   - Mantener consistencia de estilo

3. Chain-of-Thought:
   - Útil para problemas complejos
   - Desglosar el razonamiento paso a paso
   - Mejora calidad y debuggeability

4. Constraints:
   - Especificar edge cases
   - Definir requisitos de performance
   - Aclarar limitaciones técnicas

5. Task Description:
   - Clara y sin ambigüedad
   - Incluir tipos y dimensiones
   - Especificar formato de salida
""")
```

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Por qué es importante incluir ejemplos de código con masks en prompts para Triton?
2. ¿Cuándo usarías CoT vs few-shot prompting para generación de código?
3. ¿Qué información del problema de KernelBench es esencial incluir en el prompt?

### Ejercicio práctico

Toma uno de los prompts de KernelAgent (disponibles en `.fuse/prompts/`) y:
1. Identifica sus componentes (system, context, examples)
2. Propón una mejora al prompt
3. Compara la salida del modelo con ambas versiones

### Para pensar

> *¿Cómo cambiaría tu estrategia de prompting si tuvieras que generar código para un DSL completamente nuevo que el modelo nunca ha visto en su entrenamiento?*

---

## Próximos pasos

En la siguiente semana, exploraremos **XGrammar y Constrained Decoding** - cómo forzar al modelo a generar código que sea sintácticamente válido, combinando el arte de los prompts con la ciencia de las gramáticas formales.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Wei, J. et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). NeurIPS 2022.
- Brown, T. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS 2020.
- Zhou, D. et al. (2023). [Least-to-Most Prompting Enables Complex Reasoning](https://arxiv.org/abs/2205.10625). ICLR 2023.
