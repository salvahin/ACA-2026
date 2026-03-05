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

# Lectura 6: LLMs para Código


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/02_llms_para_codigo.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Distinguir entre LLMs generales y modelos especializados para código
- Diseñar prompts efectivos para generación de código con type hints y contexto
- Aplicar técnicas de few-shot learning y chain-of-thought para mejorar calidad
- Implementar validación automática de código generado
- Evaluar trade-offs entre generación libre y constrained decoding
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Diseñar prompts efectivos para que tu agente KernelAgent genere kernels Triton con la estructura y estilo correctos. Comprenderás por qué los modelos especializados son superiores para generación de código y cómo validar automáticamente el código generado.
```

## Introducción

Los LLMs entrenados en lenguaje natural (GPT, Claude) son competentes en código, pero existen modelos especializados que dominan esta tarea: CodeLlama, StarCoder, DeepSeek-Coder. ¿Qué los hace especiales? ¿Cómo los usamos efectivamente?

En esta lectura exploraremos cómo entrenar y usar LLMs para código, la importancia del **prompt engineering** y la **chain-of-thought**, y los trade-offs entre calidad y restricciones.

---

## Parte 1: Modelos Especializados para Código

### ¿Por qué Modelos Especializados?

Los LLMs generales (GPT-4, Claude) se entrenan con ~80% texto natural y ~20% código. Pero para programación:

```
Tarea: Escribe una función en Python que ordena un array
Complejidad: Media

LLM General (GPT-3.5):     ✓ Funciona, ~70% de las veces produce código que ejecuta
LLM Especializado (StarCoder):  ✓ Funciona mejor, ~85% de ejecución correcta
```

¿Por qué la diferencia?

```
1. DISTRIBUCIÓN DE ENTRENAMIENTO
   GPT: [lenguaje natural 80%, código 20%]
   StarCoder: [código 50%, lenguaje natural 40%, documentación 10%]

2. DIVERSIDAD DE LENGUAJES
   GPT: Principalmente Python y JavaScript
   StarCoder: 80 lenguajes de programación

3. CONTEXTO Y DOCUMENTACIÓN
   GPT: "def sort():" sin contexto
   StarCoder: "def sort(arr: List[int]) -> List[int]:" con tipos

4. DENSIDAD DE INFORMACIÓN
   En código, cada caractér importa
   StarCoder está optimizado para notar detalles
```

### Principales Modelos de Código

```
CodeLlama (Meta)
  - Base: Llama 2 (7B, 13B, 34B)
  - Variante Python: optimizado para Python
  - Variante Instruct: siguiendo instrucciones
  - Rendimiento: muy bueno en Python, OK en otros lenguajes

StarCoder (Big Code)
  - 15B parámetros
  - Entrenado en 80 lenguajes
  - Especialización: balanceada entre lenguajes
  - Rendimiento: excelente en JavaScript, Python, Java

DeepSeek-Coder (DeepSeek)
  - 6B a 33B parámetros
  - Enfoque en contexto largo (4K a 16K)
  - Rendimiento: excelente en código chinoz contexto

Codex (OpenAI) - Cerrado
  - Base: GPT-3
  - Motor detrás de GitHub Copilot
  - Rendimiento: estado del arte
```

---

## Parte 2: Características que Importan

### Type Hints

Los LLMs para código responden mejor cuando ves type hints:

```python
# SIN TYPE HINTS (ambiguo)
def process(data):
    return data * 2  # ¿Multiplica? ¿Repite lista?

# CON TYPE HINTS (claro)
def process(data: List[int]) -> List[int]:
    return [x * 2 for x in data]
```

El modelo entiende:
- `data` es una lista de enteros
- Retorna una lista de enteros
- La operación probablemente es elemento-sabio

### Docstrings

```python
# SIN DOCSTRING
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# CON DOCSTRING (malo - ineficiente)
def fibonacci(n: int) -> int:
    """Retorna el n-ésimo número de Fibonacci.

    Nota: Recursión ingenuua es ineficiente.
    """
    ...

# El modelo ENTIENDE que debe evitar recursión ingenua
```

### Contexto de Archivo

LLMs para código funcionan mejor cuando ven:
1. Importes (qué librerías se usan)
2. Clases definidas (estructura del código)
3. Funciones similares (patrones del codebase)

```{code-cell} ipython3
# Contexto (primeras líneas del archivo)
import numpy as np
from typing import List

class DataProcessor:
    def __init__(self, data: np.ndarray):
        self.data = data

    def normalize(self) -> np.ndarray:
        return self.data / np.max(self.data)

    def _helper(self) -> float:
        return np.mean(self.data)

# Ahora pide al modelo
# "Escribe una función `outliers()` que retorna índices de outliers"
# El modelo VE que usamos numpy, arrays, que tenemos self.data, etc.
# Más probable generar código coherente con el estilo del archivo
```

---

## Parte 3: Prompt Engineering para Código

### El Problema Base

```python
Prompt: "Escribe una función que ordena un array"
Salida:
def sort_array(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

Problemas:
- Bubble sort (O(n²))
- Sin type hints
- Sin docstring
- Sin manejo de errores
```

### Prompt Engineering: Especificidad

```python
# MEJOR
Prompt:
"""
Escribe una función Python que ordena un array de enteros.

Requisitos:
- Use quicksort o mergesort (eficiencia O(n log n))
- Incluye type hints
- Incluye un docstring con ejemplo
- Maneja arrays vacíos
- Escribir comentarios para pasos complejos

Ejemplo esperado:
    >>> sort_array([3, 1, 4, 1, 5])
    [1, 1, 3, 4, 5]
"""

Salida (probable):
def sort_array(arr: List[int]) -> List[int]:
    """Ordena un array de enteros usando mergesort.

    Args:
        arr: Lista de enteros

    Returns:
        Lista ordenada

    Examples:
        >>> sort_array([3, 1, 4])
        [1, 3, 4]
    """
    if not arr or len(arr) == 1:
        return arr

    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        return merge(left, right)

    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    return merge_sort(arr)
```

**Mejoras:**
- Especificó algoritmo (mergesort)
- Pidió type hints
- Pidió docstring con ejemplo
- Mencionó casos especiales
- El modelo produce código mejor

### Técnica: Few-Shot Learning

```python
Prompt:
"""
Aquí están ejemplos de funciones bien escritas:

Ejemplo 1:
def add(a: int, b: int) -> int:
    '''Suma dos números.'''
    return a + b

Ejemplo 2:
def multiply(a: float, b: float) -> float:
    '''Multiplica dos números.'''
    return a * b

Ahora escribe una función que divide dos números:
"""

El modelo IMITA el estilo de los ejemplos:
def divide(a: float, b: float) -> float:
    '''Divide dos números.'''
    if b == 0:
        raise ValueError("No puedes dividir por cero")
    return a / b

✓ Type hints (como ejemplos)
✓ Docstring (como ejemplos)
✓ Manejo de errores (razonable)
```

---

## Parte 4: Chain-of-Thought para Código

La idea: antes de generar código, el modelo explica su pensamiento.

### Problema Complejo

```
Tarea: "Dados dos arrays, encuentra elementos comunes"

SIN Chain-of-Thought:
def common_elements(a, b):
    return [x for x in a if x in b]

Problema: O(n*m) por la búsqueda `x in b`
No óptimo, el modelo no pensó en eficiencia
```

### CON Chain-of-Thought

```python
Prompt:
"""
Dados dos arrays, encuentra elementos comunes.

Piensa paso a paso:
1. ¿Cuál es la complejidad ingenuua?
2. ¿Cómo podemos optimizar?
3. ¿Hay estructuras de datos útiles?
4. Escribe el código con explicación
"""

Respuesta del modelo:

**Pensamiento:**
1. Complejidad ingenuua: O(n*m) - para cada elemento de a, buscar en b
2. Optimizar: convertir uno de los arrays en set (lookup O(1))
3. Estructura de datos: set o hash table
4. Código:

def common_elements(a: List[int], b: List[int]) -> List[int]:
    '''Encuentra elementos comunes entre dos arrays.

    Usa set para O(n + m) complejidad.
    '''
    set_b = set(b)  # O(m)
    return [x for x in a if x in set_b]  # O(n)
```

**Ventaja:** El modelo explicitó pensamiento, generó código mejor.

---

## Parte 5: Testing y Validación

### El Problema

Código generado por LLMs **parece correcto** pero puede tener bugs:

```python
# El modelo genera:
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Funciona para n pequeño:
fibonacci(5)  # = 5 ✓

# Pero explota para n grande:
fibonacci(35)  # tarda 10 segundos
fibonacci(40)  # tarda minutos
# Recursión ingenua, sin memoización
```

### Validación Automática

```python
def validate_solution(code: str, test_cases: List[Tuple]):
    """Ejecuta código contra test cases"""
    try:
        exec(code)  # Ejecuta código
        for input_val, expected_output in test_cases:
            result = eval(f"solution({input_val})")
            if result != expected_output:
                return False, f"Test falló: {input_val} → {result}, esperado {expected_output}"
        return True, "Todos los tests pasaron"
    except Exception as e:
        return False, f"Error en ejecución: {e}"

# Uso:
code = model.generate(prompt)
valid, msg = validate_solution(code, test_cases)
if not valid:
    # Pide al modelo que lo corrija
    code = model.generate(f"{prompt}\nError anterior: {msg}")
```

### Test Cases

```python
# Mal prompt
"Escribe fibonacci"

# Mejor prompt
"""
Escribe una función fibonacci eficiente.

Test cases:
    fibonacci(0) == 0
    fibonacci(1) == 1
    fibonacci(5) == 5
    fibonacci(10) == 55

Debe ser rápida incluso para fibonacci(50).
"""
```

---

## Parte 6: Trade-off Calidad vs Restricciones

### Generación Libre

```python
modelo.generate(
    "Escribe una función de ordenamiento",
    temperature=1.0,
    top_p=0.95
)

✓ Potencialmente más creativa
✓ Puede usar algoritmos modernos
✗ Puede generar código ineficiente
✗ Puede tener bugs
```

### Con Constrainted Decoding

```python
# Restricción: Solo palabras clave válidas de Python
grammar = python_grammar

modelo.generate(
    "Escribe una función",
    grammar=grammar
)

✓ Código garantizado sintácticamente válido
✓ No hay errores de sintaxis
✗ Menos creatividad (restringido a construcciones válidas)
✗ Puede reducir calidad (el modelo quiere usar algo no válido en la gramática)
```

### Recomendación

Para código:
1. **Usa constrained decoding:** Sintaxis incorrecta es inaceptable
2. **Agrega testing automático:** Valida lógica
3. **Combina:** Constrained decoding (sintaxis) + reintento en fallos (lógica)

---

## Parte 7: Ejemplo Completo: Sistema de Generación de Código

```{code-cell} ipython3
:tags: [skip-execution]

from transformers import AutoModelForCausalLM, AutoTokenizer

class CodeGenerator:
    def __init__(self, model_name: str = "bigcode/starcoder"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt: str, test_cases: List = None) -> str:
        """Genera código con validación"""

        # Genera código
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        code = self.tokenizer.decode(outputs[0])

        # Valida si tenemos test cases
        if test_cases:
            valid, msg = self.validate(code, test_cases)
            if not valid:
                # Reintenta con feedback
                prompt_fixed = f"{prompt}\n\nAnterior intento falló: {msg}\nIntenta nuevamente:"
                return self.generate(prompt_fixed, test_cases)

        return code

    def validate(self, code: str, test_cases: List) -> Tuple[bool, str]:
        """Valida código contra test cases"""
        try:
            exec(code)
            for inp, expected in test_cases:
                result = eval(f"solution({inp})")
                assert result == expected, f"Test {inp} falló"
            return True, "OK"
        except Exception as e:
            return False, str(e)

# Uso
gen = CodeGenerator("bigcode/starcoder")

prompt = """
def solution(arr: List[int]) -> List[int]:
    '''Ordena array de forma eficiente'''
"""

test_cases = [
    ([3, 1, 2], [1, 2, 3]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
]

code = gen.generate(prompt, test_cases)
print(code)
```

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **Especialización:** ¿Por qué un modelo especializado en código (StarCoder) es mejor que GPT-4 en código, si GPT-4 es más grande?

2. **Type Hints:** ¿Son realmente necesarios para LLMs? ¿Qué información adicional proporcionan?

3. **Testing:** ¿Cómo balancearías test cases automáticos vs inspección manual en producción?

### Ejercicios Prácticos:

1. **Mejora un prompt:**
   ```
   Actual: "Escribe una función que calcula factorial"
   Reescribe el prompt siendo más específico:
   - Requisitos de eficiencia
   - Type hints esperados
   - Manejo de errores
   - Ejemplos
   ```

2. **Chain-of-Thought:**
   ```
   Tarea: Dado un array, encuentra el elemento que aparece más de n/2 veces

   Escribe un chain-of-thought pensamiento paso a paso
   (sin código, solo pensamiento)
   ```

3. **Análisis: Modelos de Código**
   ```
   Compara CodeLlama, StarCoder, y DeepSeek-Coder en:
   - Parámetros
   - Lenguajes soportados
   - Rendimiento en Python
   - Ventajas/desventajas
   ```

4. **Reflexión escrita (350 palabras):** "Los LLMs para código pueden generar código que 'se ve bien' pero que no ejecuta. ¿Cómo estructurarías un sistema en producción para confiar en código generado por LLMs sin ejecutar automáticamente?"

---

## Puntos Clave

- **Modelos especializados:** CodeLlama, StarCoder, DeepSeek-Coder entrenados con más código
- **Type hints y docstrings:** Ayudan enormemente al modelo a generar código mejor
- **Prompt engineering:** Sé específico, incluye requisitos, ejemplos, y casos especiales
- **Chain-of-Thought:** Pide al modelo que pense antes de generar código
- **Testing automático:** Valida código contra test cases, reintenta si falla
- **Constrained decoding:** Usa para sintaxis (garantiza código válido), no para lógica
- **Trade-off:** Libertad creativa vs garantías de validez

---

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- Los modelos especializados (CodeLlama, StarCoder) superan a los LLMs generales en generación de código porque fueron entrenados con mayor proporción de código (50% vs 20%)
- Type hints, docstrings y contexto de archivo son fundamentales para guiar al modelo
- Prompt engineering efectivo incluye: especificidad, ejemplos (few-shot), y chain-of-thought para problemas complejos
- Validación automática con test cases + reintento en fallos garantiza código funcional
- Constrained decoding garantiza sintaxis válida pero puede reducir creatividad

**Siguiente paso:** En la próxima lectura profundizaremos en **Prompt Engineering avanzado**, explorando cómo diseñar system prompts, técnicas de few-shot learning óptimas, y cómo estructurar prompts multi-etapa para generación de código complejo.
```

```{admonition} Verifica tu comprensión
:class: note
1. ¿Por qué StarCoder (15B parámetros) puede superar a GPT-3.5 (175B) en tareas de código?
2. Si tienes que generar un kernel Triton complejo, ¿usarías generación libre o constrained decoding? ¿Por qué?
3. Diseña un chain-of-thought para la tarea: "Genera un kernel que multiplica matrices 2D con tiling"
4. ¿Cuál es la diferencia entre validar sintaxis (constrained decoding) y validar lógica (test cases)?
```

---

## Referencias

- Rozière, B. et al. (2023). [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950). arXiv.
- Li, R. et al. (2023). [StarCoder: May the Source Be With You!](https://arxiv.org/abs/2305.06161). arXiv.
- Guo, D. et al. (2024). [DeepSeek-Coder: When the Large Language Model Meets Programming](https://arxiv.org/abs/2401.14196). arXiv.

