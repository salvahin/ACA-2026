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

# 07. Gramática L3-L4: Expresiones y Control Flow

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
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/07_gramatica_triton_l3l4.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Modelar expresiones aritméticas, lógicas y de comparación en EBNF y JSON Schema
- Definir llamadas a funciones de Triton (tl.load, tl.store, tl.arange, etc.)
- Especificar estructuras de control flow (if/else, for, while) con cuerpos recursivos
- Generar código Python desde specs JSON de expresiones y statements
- Identificar los 5 patrones fundamentales que cubren 80% de kernels Triton
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Completar la gramática completa de KernelAgent con L3-L4, permitiendo que el LLM genere la lógica computacional de kernels Triton. Ahora tu agente puede generar kernels completos y funcionales desde estructura hasta lógica.
```


## Introducción

Ahora llegamos al corazón del kernel: el cuerpo del código. **L3** cubre expresiones (aritmética, comparaciones, indexación), y **L4** cubre control flow (if, for, while). Esto es donde ocurre la computación real.

Esta lectura es la más técnica, porque los patrones son complejos. Pero la idea es la misma: especificar qué sintaxis es válida en código Triton.

## L3: Expresiones en Triton

### Categorías de Expresiones

En código Triton típicamente ves:

```python
# Operaciones aritméticas
result = x + y * 2
result = tl.sum(x)

# Indexación
value = array[0]
value = array[idx]

# Comparaciones
mask = x > 0
mask = x == y

# Llamadas a funciones API
data = tl.load(ptr + offset, mask=mask)
tl.store(ptr + offset, data, mask=mask)

# Conversiones de tipo
x = x.to(tl.float32)
```

### EBNF para L3

```ebnf
expression = assignment
           | ternary_expr

assignment = identifier "=" expression

ternary_expr = logical_or_expr ("if" logical_or_expr "else" ternary_expr)?

logical_or_expr = logical_and_expr ("or" logical_and_expr)*

logical_and_expr = bitwise_or_expr ("and" bitwise_or_expr)*

bitwise_or_expr = bitwise_xor_expr ("|" bitwise_xor_expr)*

bitwise_xor_expr = bitwise_and_expr ("^" bitwise_and_expr)*

bitwise_and_expr = comparison_expr ("&" comparison_expr)*

comparison_expr = arithmetic_expr (comp_op arithmetic_expr)*

comp_op = "==" | "!=" | "<" | "<=" | ">" | ">="

arithmetic_expr = mul_expr (add_op mul_expr)*

add_op = "+" | "-"

mul_expr = unary_expr (mul_op unary_expr)*

mul_op = "*" | "/" | "%" | "//"

unary_expr = unary_op? postfix_expr

unary_op = "-" | "+" | "~" | "not"

postfix_expr = primary_expr postfix_part*

postfix_part = subscript | method_call | member_access

subscript = "[" expression "]"

method_call = "." identifier "(" args? ")"

member_access = "." identifier

primary_expr = literal
             | identifier
             | "(" expression ")"
             | function_call

function_call = identifier "(" args? ")"
              | "tl" "." identifier "(" args? ")"

args = expression ("," expression)*

literal = NUMBER | STRING | "True" | "False" | "None"
```

### Expresiones Comunes en Triton

```python
# Cargar datos
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# Operaciones vectoriales
y = x + 1
z = x * y

# Reducciones
sum_x = tl.sum(x)
max_x = tl.max(x)

# Conversiones
x_float = x.to(tl.float32)

# Creación de tensores
offsets = tl.arange(0, BLOCK_SIZE)
zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

# Operaciones condicionales
result = tl.where(mask, x, y)

# Operaciones booleanas
mask = (x > 0) & (y < 10)
```

## L4: Control Flow

### Patrones de Control Flow

Triton soporta `if`, `for`, `while` con algunas limitaciones (deben ser compilables):

```python
# if simple
if condition:
    x = y + 1

# if-else
if x > 0:
    result = x
else:
    result = -x

# for loop
for i in range(n):
    x[i] = y[i] + 1

# while loop
while condition:
    x = x + 1
```

### EBNF para L4

```ebnf
statement = expression_stmt
          | if_stmt
          | for_stmt
          | while_stmt
          | return_stmt
          | assignment_stmt

if_stmt = "if" condition ":" suite
          ("elif" condition ":" suite)*
          ("else" ":" suite)?

for_stmt = "for" identifier "in" iterable ":" suite

while_stmt = "while" condition ":" suite

return_stmt = "return" expression?

assignment_stmt = identifier "=" expression
                | identifier ("[" expression "]") "=" expression

suite = simple_stmt
      | NEWLINE INDENT statement+ DEDENT

condition = expression

iterable = "range" "(" arguments ")"
         | identifier

arguments = expression ("," expression)*
```

## Implementación L3-L4 en JSON Schema

```{code-cell} ipython3
# schemas/triton_l3_l4_schema.py

import json

TRITON_L3_L4_EXPRESSION_SCHEMA = {
    "type": "object",
    "oneOf": [
        # Expresiones simples
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["literal"]},
                "value": {"oneOf": [
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "boolean"},
                    {"enum": [None]}
                ]}
            },
            "required": ["type", "value"]
        },
        # Variable reference
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["identifier"]},
                "name": {"type": "string"}
            },
            "required": ["type", "name"]
        },
        # Operación binaria
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["binary_op"]},
                "operator": {
                    "enum": ["+", "-", "*", "/", "%", "//",
                             "&", "|", "^", "<<", ">>",
                             "==", "!=", "<", "<=", ">", ">=",
                             "and", "or"]
                },
                "left": {"$ref": "#"},
                "right": {"$ref": "#"}
            },
            "required": ["type", "operator", "left", "right"]
        },
        # Operación unaria
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["unary_op"]},
                "operator": {"enum": ["-", "+", "~", "not"]},
                "operand": {"$ref": "#"}
            },
            "required": ["type", "operator", "operand"]
        },
        # Indexación
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["subscript"]},
                "object": {"$ref": "#"},
                "index": {"$ref": "#"}
            },
            "required": ["type", "object", "index"]
        },
        # Llamada a función
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["call"]},
                "func": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"enum": ["identifier"]},
                                "name": {"type": "string"}
                            }
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {"enum": ["member"]},
                                "object": {"type": "string"},  # "tl"
                                "member": {"type": "string"}     # "load", "store", etc.
                            }
                        }
                    ]
                },
                "args": {
                    "type": "array",
                    "items": {"$ref": "#"}
                },
                "kwargs": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#"}
                }
            },
            "required": ["type", "func"]
        }
    ]
}

TRITON_L3_L4_STATEMENT_SCHEMA = {
    "type": "object",
    "oneOf": [
        # Assignment
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["assign"]},
                "target": {"type": "string"},
                "value": TRITON_L3_L4_EXPRESSION_SCHEMA
            },
            "required": ["type", "target", "value"]
        },
        # If statement
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["if"]},
                "condition": TRITON_L3_L4_EXPRESSION_SCHEMA,
                "then_body": {
                    "type": "array",
                    "items": {"$ref": "#"}
                },
                "else_body": {
                    "type": "array",
                    "items": {"$ref": "#"}
                }
            },
            "required": ["type", "condition", "then_body"]
        },
        # For loop
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["for"]},
                "target": {"type": "string"},
                "iter": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"enum": ["range_call"]},
                                "start": TRITON_L3_L4_EXPRESSION_SCHEMA,
                                "end": TRITON_L3_L4_EXPRESSION_SCHEMA,
                                "step": TRITON_L3_L4_EXPRESSION_SCHEMA
                            }
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {"enum": ["identifier"]},
                                "name": {"type": "string"}
                            }
                        }
                    ]
                },
                "body": {
                    "type": "array",
                    "items": {"$ref": "#"}
                }
            },
            "required": ["type", "target", "iter", "body"]
        },
        # While loop
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["while"]},
                "condition": TRITON_L3_L4_EXPRESSION_SCHEMA,
                "body": {
                    "type": "array",
                    "items": {"$ref": "#"}
                }
            },
            "required": ["type", "condition", "body"]
        },
        # Return
        {
            "type": "object",
            "properties": {
                "type": {"enum": ["return"]},
                "value": TRITON_L3_L4_EXPRESSION_SCHEMA
            }
        }
    ]
}

TRITON_KERNEL_BODY_SCHEMA = {
    "type": "array",
    "items": TRITON_L3_L4_STATEMENT_SCHEMA,
    "description": "Cuerpo del kernel (L3-L4)"
}
```

## Generador de Código L3-L4

```{code-cell} ipython3
# generators/l3_l4_generator.py

class ExpressionGenerator:
    """Genera expresiones Python/Triton desde JSON"""

    def generate_expression(self, expr_spec):
        """Generar una expresión"""

        if expr_spec["type"] == "literal":
            return self._format_literal(expr_spec["value"])

        elif expr_spec["type"] == "identifier":
            return expr_spec["name"]

        elif expr_spec["type"] == "binary_op":
            left = self.generate_expression(expr_spec["left"])
            right = self.generate_expression(expr_spec["right"])
            op = expr_spec["operator"]
            return f"({left} {op} {right})"

        elif expr_spec["type"] == "unary_op":
            operand = self.generate_expression(expr_spec["operand"])
            op = expr_spec["operator"]
            return f"{op}{operand}"

        elif expr_spec["type"] == "subscript":
            obj = self.generate_expression(expr_spec["object"])
            idx = self.generate_expression(expr_spec["index"])
            return f"{obj}[{idx}]"

        elif expr_spec["type"] == "call":
            func_spec = expr_spec["func"]
            if func_spec["type"] == "identifier":
                func_name = func_spec["name"]
            else:  # member
                func_name = f"{func_spec['object']}.{func_spec['member']}"

            args = [self.generate_expression(arg) for arg in expr_spec.get("args", [])]
            kwargs = {
                k: self.generate_expression(v)
                for k, v in expr_spec.get("kwargs", {}).items()
            }

            args_str = ", ".join(args)
            if kwargs:
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str

            return f"{func_name}({args_str})"

        else:
            raise ValueError(f"Tipo desconocido: {expr_spec['type']}")

    @staticmethod
    def _format_literal(value):
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif value is None:
            return "None"
        else:
            return str(value)


class StatementGenerator(ExpressionGenerator):
    """Genera statements (if, for, etc.)"""

    def generate_statement(self, stmt_spec, indent=0):
        """Generar un statement con indentación"""

        indent_str = "    " * indent

        if stmt_spec["type"] == "assign":
            target = stmt_spec["target"]
            value = self.generate_expression(stmt_spec["value"])
            return f"{indent_str}{target} = {value}"

        elif stmt_spec["type"] == "if":
            lines = []
            condition = self.generate_expression(stmt_spec["condition"])
            lines.append(f"{indent_str}if {condition}:")

            for stmt in stmt_spec["then_body"]:
                lines.append(self.generate_statement(stmt, indent + 1))

            if "else_body" in stmt_spec:
                lines.append(f"{indent_str}else:")
                for stmt in stmt_spec["else_body"]:
                    lines.append(self.generate_statement(stmt, indent + 1))

            return "\n".join(lines)

        elif stmt_spec["type"] == "for":
            lines = []
            target = stmt_spec["target"]
            iter_spec = stmt_spec["iter"]

            if iter_spec["type"] == "range_call":
                start = self.generate_expression(iter_spec["start"])
                end = self.generate_expression(iter_spec["end"])
                step = self.generate_expression(iter_spec.get("step", {"type": "literal", "value": 1}))
                iter_expr = f"range({start}, {end}, {step})"
            else:
                iter_expr = iter_spec["name"]

            lines.append(f"{indent_str}for {target} in {iter_expr}:")

            for stmt in stmt_spec["body"]:
                lines.append(self.generate_statement(stmt, indent + 1))

            return "\n".join(lines)

        elif stmt_spec["type"] == "while":
            lines = []
            condition = self.generate_expression(stmt_spec["condition"])
            lines.append(f"{indent_str}while {condition}:")

            for stmt in stmt_spec["body"]:
                lines.append(self.generate_statement(stmt, indent + 1))

            return "\n".join(lines)

        elif stmt_spec["type"] == "return":
            if "value" in stmt_spec:
                value = self.generate_expression(stmt_spec["value"])
                return f"{indent_str}return {value}"
            else:
                return f"{indent_str}return"

        else:
            raise ValueError(f"Statement desconocido: {stmt_spec['type']}")

# Uso
if __name__ == "__main__":
    # Expresión: x + y * 2
    expr_spec = {
        "type": "binary_op",
        "operator": "+",
        "left": {
            "type": "identifier",
            "name": "x"
        },
        "right": {
            "type": "binary_op",
            "operator": "*",
            "left": {"type": "identifier", "name": "y"},
            "right": {"type": "literal", "value": 2}
        }
    }

    gen = ExpressionGenerator()
    print(gen.generate_expression(expr_spec))
    # Output: (x (y * 2))

    # Statement: if x > 0: y = x
    stmt_spec = {
        "type": "if",
        "condition": {
            "type": "binary_op",
            "operator": ">",
            "left": {"type": "identifier", "name": "x"},
            "right": {"type": "literal", "value": 0}
        },
        "then_body": [
            {
                "type": "assign",
                "target": "y",
                "value": {"type": "identifier", "name": "x"}
            }
        ]
    }

    stmt_gen = StatementGenerator()
    print(stmt_gen.generate_statement(stmt_spec))
    # Output:
    # if (x > 0):
    #     y = x
```

## Tests L3-L4

```{code-cell} ipython3
# tests/test_l3_l4.py

import pytest
from generators.l3_l4_generator import ExpressionGenerator, StatementGenerator

class TestExpressions:
    """Tests para expresiones L3"""

    def test_literal(self):
        gen = ExpressionGenerator()
        expr = {"type": "literal", "value": 42}
        assert gen.generate_expression(expr) == "42"

    def test_identifier(self):
        gen = ExpressionGenerator()
        expr = {"type": "identifier", "name": "x"}
        assert gen.generate_expression(expr) == "x"

    def test_binary_op(self):
        gen = ExpressionGenerator()
        expr = {
            "type": "binary_op",
            "operator": "+",
            "left": {"type": "identifier", "name": "x"},
            "right": {"type": "literal", "value": 1}
        }
        result = gen.generate_expression(expr)
        assert "x" in result and "1" in result and "+" in result

    def test_tl_load_call(self):
        gen = ExpressionGenerator()
        expr = {
            "type": "call",
            "func": {
                "type": "member",
                "object": "tl",
                "member": "load"
            },
            "args": [{"type": "identifier", "name": "ptr"}],
            "kwargs": {
                "mask": {"type": "identifier", "name": "mask"},
                "other": {"type": "literal", "value": 0}
            }
        }
        result = gen.generate_expression(expr)
        assert "tl.load" in result
        assert "ptr" in result
        assert "mask=" in result

class TestStatements:
    """Tests para control flow L4"""

    def test_assignment(self):
        gen = StatementGenerator()
        stmt = {
            "type": "assign",
            "target": "x",
            "value": {"type": "literal", "value": 42}
        }
        result = gen.generate_statement(stmt)
        assert result == "x = 42"

    def test_if_statement(self):
        gen = StatementGenerator()
        stmt = {
            "type": "if",
            "condition": {
                "type": "binary_op",
                "operator": ">",
                "left": {"type": "identifier", "name": "x"},
                "right": {"type": "literal", "value": 0}
            },
            "then_body": [
                {
                    "type": "assign",
                    "target": "y",
                    "value": {"type": "identifier", "name": "x"}
                }
            ]
        }
        result = gen.generate_statement(stmt)
        assert "if" in result
        assert "y = x" in result

    def test_for_loop(self):
        gen = StatementGenerator()
        stmt = {
            "type": "for",
            "target": "i",
            "iter": {
                "type": "range_call",
                "start": {"type": "literal", "value": 0},
                "end": {"type": "literal", "value": 10},
                "step": {"type": "literal", "value": 1}
            },
            "body": [
                {
                    "type": "assign",
                    "target": "x",
                    "value": {"type": "identifier", "name": "i"}
                }
            ]
        }
        result = gen.generate_statement(stmt)
        assert "for i in range" in result
```

## Ejercicios

1. **Extiende las expresiones**:
   - Agrega soporte para operadores de asignación compuesta (+=, -=, etc.)
   - Agrega soporte para expresiones ternarias (`x if cond else y`)
   - Actualiza el generador y tests

2. **Analiza patrones del corpus**:
   - Extrae las expresiones más comunes de tus kernels del corpus
   - Crea un histograma de operadores usados
   - ¿Cuál es la profundidad máxima de expresiones anidadas?

3. **Validación semántica**:
   - Crea un analizador que detecte variables no definidas
   - Valida que las funciones tl.* se usan correctamente
   - (Pista: necesitarás rastrear el "scope" de variables)

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- L3 (expresiones): modelar operaciones aritméticas, lógicas, llamadas a tl.* (load/store/arange), indexación
- L4 (control flow): if/else, for, while con cuerpos recursivos (statements que contienen statements)
- 5 patrones fundamentales cubren 80% de kernels: offsets (tl.arange), mask (condición), load, compute, store
- Composición de expresiones es infinita pero gramática debe cubrir casos comunes sin ser infinita
- Trade-off crítico: gramática restrictiva (segura) vs permisiva (flexible) - balance óptimo cubre 95% de casos

**Siguiente paso:** En la próxima lectura aprenderemos sobre **Serving y Optimización de LLMs**, explorando cómo desplegar tu modelo de forma eficiente con KV-Cache, continuous batching, cuantización y frameworks como vLLM.
```

```{admonition} Verifica tu comprensión
:class: note
1. ¿Por qué L3-L4 es más complejo que L1-L2? ¿Qué hace que las expresiones sean "infinitas"?
2. Genera el JSON spec para la expresión: `result = tl.sum(tl.load(ptr + offsets, mask=mask))`
3. Los 5 patrones fundamentales cubren 80% de kernels. ¿Cuáles son y en qué orden típicamente aparecen?
4. ¿Cómo validarías semánticamente que una máscara se usa correctamente en tl.load?
```

## Conexión con el Proyecto

L3-L4 es donde ocurre **la magia real**: la lógica de computación del kernel. Mientras L1-L2 define la estructura, L3-L4 define qué hace el kernel.

**Importancia crítica en KernelAgent**:
1. **Expresiones (L3)**: Todas las operaciones aritméticas, loads/stores, transformaciones
2. **Control Flow (L4)**: Loops para procesar bloques, condicionales para masking
3. **Correctitud**: Un error en L3-L4 puede causar resultados incorrectos (peor que un error de sintaxis)

**Desafío principal**:
- L1-L2 es finito y simple (pocos patrones)
- L3-L4 es **infinito y complejo** (infinitas combinaciones de expresiones)
- XGrammar debe restringir sin ser demasiado limitante

**Flujo en el proyecto**:
```
Usuario: "kernel que multiplica matrices"
→ LLM genera L1-L2 (estructura)
→ LLM genera L3-L4:
   - Cargar bloques de A y B (tl.load)
   - Multiplicar (operaciones aritméticas)
   - Reducir (tl.sum)
   - Guardar resultado (tl.store)
→ XGrammar valida cada expresión
→ Código Triton compilable
```

**Trade-offs de diseño**:
- **Gramática muy restrictiva**: Solo permite patrones conocidos → Limitado pero seguro
- **Gramática muy permisiva**: Permite casi cualquier expresión → Flexible pero arriesgado
- **Balance óptimo**: Cubre 95% de casos comunes, rechaza construcciones raras

En KernelAgent, necesitas el balance óptimo para productividad.

## Ejercicio Práctico: Generar Expresión Triton Específica

```{code-cell} ipython3
import json

# Vamos a generar una expresión completa para vector addition kernel
# El patrón típico es: cargar, computar, guardar

# Schema ya definido en la lectura (usaremos fragmento)
class TritonExpressionGenerator:
    """Genera expresiones Triton desde especificaciones JSON"""

    def generate_expression(self, expr_spec):
        """Generar una expresión Triton"""
        if expr_spec["type"] == "literal":
            value = expr_spec["value"]
            if isinstance(value, str):
                return f'"{value}"'
            elif isinstance(value, bool):
                return "True" if value else "False"
            else:
                return str(value)

        elif expr_spec["type"] == "identifier":
            return expr_spec["name"]

        elif expr_spec["type"] == "binary_op":
            left = self.generate_expression(expr_spec["left"])
            right = self.generate_expression(expr_spec["right"])
            op = expr_spec["operator"]
            return f"({left} {op} {right})"

        elif expr_spec["type"] == "call":
            func_spec = expr_spec["func"]
            if func_spec["type"] == "member":
                func_name = f"{func_spec['object']}.{func_spec['member']}"
            else:
                func_name = func_spec["name"]

            args = [self.generate_expression(arg) for arg in expr_spec.get("args", [])]
            kwargs = {
                k: self.generate_expression(v)
                for k, v in expr_spec.get("kwargs", {}).items()
            }

            args_str = ", ".join(args)
            if kwargs:
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str

            return f"{func_name}({args_str})"

        else:
            return f"<unknown: {expr_spec['type']}>"

# Caso práctico: Kernel vector add
# Patrón: offsets = tl.arange(0, BLOCK_SIZE)

offsets_expr_spec = {
    "type": "call",
    "func": {
        "type": "member",
        "object": "tl",
        "member": "arange"
    },
    "args": [
        {"type": "literal", "value": 0},
        {"type": "identifier", "name": "BLOCK_SIZE"}
    ]
}

# Patrón: mask = offsets < n
mask_expr_spec = {
    "type": "binary_op",
    "operator": "<",
    "left": {"type": "identifier", "name": "offsets"},
    "right": {"type": "identifier", "name": "n"}
}

# Patrón: x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
load_x_spec = {
    "type": "call",
    "func": {
        "type": "member",
        "object": "tl",
        "member": "load"
    },
    "args": [
        {
            "type": "binary_op",
            "operator": "+",
            "left": {"type": "identifier", "name": "x_ptr"},
            "right": {"type": "identifier", "name": "offsets"}
        }
    ],
    "kwargs": {
        "mask": {"type": "identifier", "name": "mask"},
        "other": {"type": "literal", "value": 0.0}
    }
}

# Patrón: output = x + y
add_expr_spec = {
    "type": "binary_op",
    "operator": "+",
    "left": {"type": "identifier", "name": "x"},
    "right": {"type": "identifier", "name": "y"}
}

# Patrón: tl.store(output_ptr + offsets, output, mask=mask)
store_spec = {
    "type": "call",
    "func": {
        "type": "member",
        "object": "tl",
        "member": "store"
    },
    "args": [
        {
            "type": "binary_op",
            "operator": "+",
            "left": {"type": "identifier", "name": "output_ptr"},
            "right": {"type": "identifier", "name": "offsets"}
        },
        {"type": "identifier", "name": "output"}
    ],
    "kwargs": {
        "mask": {"type": "identifier", "name": "mask"}
    }
}

# Generar código
generator = TritonExpressionGenerator()

print("="*60)
print("KERNEL VECTOR ADD - EXPRESIONES L3")
print("="*60)
print()
print("# Calcular offsets de bloque")
print(f"offsets = {generator.generate_expression(offsets_expr_spec)}")
print()
print("# Crear máscara para elementos válidos")
print(f"mask = {generator.generate_expression(mask_expr_spec)}")
print()
print("# Cargar datos desde memoria global")
print(f"x = {generator.generate_expression(load_x_spec)}")
print(f"y = {generator.generate_expression(load_x_spec).replace('x_ptr', 'y_ptr')}")
print()
print("# Computar suma")
print(f"output = {generator.generate_expression(add_expr_spec)}")
print()
print("# Guardar resultado")
print(generator.generate_expression(store_spec))

print("\n" + "="*60)
print("KERNEL COMPLETO GENERADO")
print("="*60)

complete_kernel = """import triton
import triton.language as tl

@triton.jit
def kernel_add(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calcular offsets de bloque
    offsets = tl.arange(0, BLOCK_SIZE)

    # Crear máscara para elementos válidos
    mask = (offsets < n)

    # Cargar datos desde memoria global
    x = tl.load((x_ptr + offsets), mask=mask, other=0.0)
    y = tl.load((y_ptr + offsets), mask=mask, other=0.0)

    # Computar suma
    output = (x + y)

    # Guardar resultado
    tl.store((output_ptr + offsets), output, mask=mask)
"""

print(complete_kernel)

print("="*60)
print("ANÁLISIS DE PATRONES L3-L4")
print("="*60)
print()
print("Patrones comunes identificados:")
print("1. OFFSETS: tl.arange(0, BLOCK_SIZE) - Siempre necesario")
print("2. MASK: (offsets < n) - Previene accesos fuera de límites")
print("3. LOAD: tl.load(ptr + offsets, mask=mask) - Lectura segura")
print("4. COMPUTE: Expresiones aritméticas simples (+, -, *, /)")
print("5. STORE: tl.store(ptr + offsets, data, mask=mask) - Escritura segura")
print()
print("Estos 5 patrones cubren ~80% de kernels Triton simples")
print()
print("Para KernelAgent:")
print("- El LLM debe aprender estos patrones")
print("- XGrammar garantiza que la sintaxis sea correcta")
print("- Tu código valida la lógica (ej: mask usada correctamente)")

# Ejercicio adicional: Generar kernel más complejo
print("\n" + "="*60)
print("EJERCICIO: Kernel con reducción (sum)")
print("="*60)

# Patrón: result = tl.sum(x)
sum_expr_spec = {
    "type": "call",
    "func": {
        "type": "member",
        "object": "tl",
        "member": "sum"
    },
    "args": [
        {"type": "identifier", "name": "x"}
    ]
}

print(f"result = {generator.generate_expression(sum_expr_spec)}")
print()
print("Este patrón es común en kernels de reducción (mean, max, etc.)")
print("XGrammar permitiría generar esta expresión automáticamente")
```

**Observaciones importantes**:

1. **Composición de expresiones**: Las expresiones se anidan (ej: `tl.load(x_ptr + offsets, ...)`)
2. **Patrones repetitivos**: Load-compute-store aparece en casi todos los kernels
3. **Validación en capas**:
   - XGrammar: Sintaxis correcta
   - Tu código: Semántica correcta (ej: mask aplicada correctamente)

**Tarea adicional**:
1. Modifica el generador para crear un kernel de multiplicación elemento-wise
2. Agrega una reducción al final (suma todos los elementos)
3. Genera el código completo y verifica que sea válido Triton

**Conexión con proyecto real**:
En tu implementación de KernelAgent:
- El LLM recibe: "suma dos vectores"
- Genera spec JSON con estos patrones L3-L4
- Tu generador produce código Triton
- XGrammar garantiza que no haya errores de sintaxis
- El resultado se compila y ejecuta en GPU

---

## Referencias

- Triton. [Triton Language and Compiler](https://github.com/triton-lang/triton). GitHub.
- Tillet, P., Kung, H.T., & Cox, D. (2019). [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). MAPL.
