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

# 06. Gramática L1-L2: Estructura Base de Kernels Triton


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/06_gramatica_triton_l1l2.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Modelar en EBNF y JSON Schema los imports requeridos (L1) de kernels Triton
- Definir decoradores (@triton.jit) con argumentos opcionales en la gramática
- Especificar firmas de función (L2) con parámetros tipados (tl.constexpr)
- Generar código Python válido desde specs JSON usando generadores
- Validar gramáticas L1-L2 con test cases exhaustivos
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Implementar el esqueleto estructural completo de KernelAgent - la gramática L1-L2 que garantiza que todo kernel generado tenga imports, decoradores y firmas correctas. Esta base sólida permitirá enfocarte en la lógica del kernel (L3-L4) sin preocuparte por errores estructurales.
```


## Introducción

Ahora que tenemos un corpus de kernels Triton y entendemos la estructura interna de XGrammar, vamos a crear la gramática **L1-L2**. Esta cubre la estructura "esqueletal" de los kernels Triton:

- **L1**: Imports y decoradores (qué necesita todo kernel)
- **L2**: Firma de la función (parámetros, tipos)

Es lo más básico. Una vez que domines esto, L3 (expresiones) y L4 (control flow) serán naturales.

## Anatomía de un Kernel Triton Mínimo

```python
import triton
import triton.language as tl

@triton.jit
def kernel_name(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # Aquí va el cuerpo (L3-L4)
    pass
```

:::{figure} diagrams/triton_grammar_levels.png
:name: fig-triton-levels
:alt: Niveles de la gramática Triton - L1 imports, L2 firma, L3-L4 lógica
:align: center
:width: 90%

**Figura 5:** Arquitectura de niveles de gramática para kernels Triton.
:::

Desglosemos:

1. **L1-Imports**: `import triton`, `import triton.language as tl`
2. **L1-Decoradores**: `@triton.jit` (obligatorio)
3. **L2-Firma**: Nombre, parámetros, tipos

## L1: Imports Requeridos

Todo kernel Triton necesita estos imports. Vamos a modelarlos en EBNF:

```ebnf
imports = import_triton import_triton_language
        | import_triton_language import_triton

import_triton = "import" "triton"
import_triton_language = "import" "triton" "." "language" ("as" "tl")?
```

En práctica, casi siempre ves:
```python
import triton
import triton.language as tl
```

Pero técnicamente podrías hacer:
```python
import triton.language
# Y luego usar triton.language.load() en vez de tl.load()
```

## L1: Decorador @triton.jit

El decorador `@triton.jit` es obligatorio. Opcionalmente puede tener argumentos:

```python
@triton.jit
def kernel(...): pass

# O con argumentos
@triton.jit(version=1)
def kernel(...): pass

@triton.jit(interpret=True)
def kernel(...): pass
```

EBNF:
```ebnf
decorator = "@" "triton" "." "jit" decorator_args?

decorator_args = "(" kwarg ("," kwarg)* ")"
kwarg = identifier "=" (string | number | boolean)
```

## L2: Firma de la Función

La firma especifica parámetros y sus tipos:

```python
def kernel_add(x_ptr, y_ptr, output_ptr, n: tl.constexpr):
    pass
```

Características:
- Parámetros sin tipo: `x_ptr, y_ptr, output_ptr` (estos son punteros)
- Parámetros con tipo `tl.constexpr`: `n: tl.constexpr` (valores constantes en tiempo de compilación)
- Opcionalmente, tipos con valores por defecto: `BLOCK_SIZE: tl.constexpr = 1024`

EBNF:
```ebnf
function_def = "def" identifier "(" parameters? ")"

parameters = parameter ("," parameter)*

parameter = identifier
          | identifier ":" type_annotation
          | identifier ":" type_annotation "=" literal

type_annotation = "tl" "." "constexpr"
                | identifier

literal = number | string | "True" | "False"
```

## Ejemplo Práctico: Kernel Vector Add

```python
import triton
import triton.language as tl

@triton.jit
def kernel_add(
    x_ptr,
    y_ptr,
    output_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    """
    Estructura L1-L2 completada.
    Falta L3 (expresiones) y L4 (control flow).
    """
    pass
```

## Implementación EBNF Completa

Aquí está la especificación formal de L1-L2 en EBNF:

```ebnf
# Nivel L1-L2: Estructura e Importes

source_code = imports decorators function_def

# ===== IMPORTS (L1) =====

imports = import_statement+

import_statement = "import" "triton"
                 | "import" "triton" "." "language" ("as" "tl")?

# ===== DECORADORES (L1) =====

decorators = decorator+

decorator = "@" "triton" "." "jit" decorator_args?

decorator_args = "(" kwarg_list? ")"

kwarg_list = kwarg ("," kwarg)*

kwarg = IDENTIFIER "=" literal

# ===== FUNCIÓN (L2) =====

function_def = "def" IDENTIFIER "(" params? ")" ":"

params = parameter ("," parameter)*

parameter = IDENTIFIER type_hint? default_value?

type_hint = ":" type_expr

type_expr = "tl" "." "constexpr"
          | IDENTIFIER

default_value = "=" literal

# ===== LITERALES =====

literal = NUMBER
        | STRING
        | "True"
        | "False"
        | "None"

# ===== IDENTIFIERS (terminales) =====

IDENTIFIER = [a-zA-Z_][a-zA-Z0-9_]*
NUMBER = [0-9]+
STRING = '"' (~["])* '"'
       | "'" (~['])* "'"
```

## Implementación en XGrammar

Ahora convertiremos esto a JSON Schema que XGrammar pueda compilar:

```{code-cell} ipython3
# schemas/triton_l1_l2_schema.py

import json

TRITON_L1_L2_SCHEMA = {
    "type": "object",
    "properties": {
        "imports": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "enum": ["import_triton", "import_triton_language"]
                    },
                    "as_name": {
                        "type": "string",
                        "default": "tl"
                    }
                },
                "required": ["type"]
            },
            "minItems": 2,
            "maxItems": 2,
            "description": "Imports requeridos: triton y triton.language"
        },
        "decorator": {
            "type": "object",
            "properties": {
                "name": {
                    "enum": ["triton.jit"]
                },
                "kwargs": {
                    "type": "object",
                    "additionalProperties": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"}
                        ]
                    }
                }
            },
            "required": ["name"]
        },
        "function": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^kernel_[a-z_][a-z0-9_]*$",
                    "description": "Nombres de kernel típicamente empiezan con 'kernel_'"
                },
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type_hint": {
                                "oneOf": [
                                    {"enum": ["tl.constexpr"]},
                                    {"type": "string"}
                                ]
                            },
                            "default": {
                                "oneOf": [
                                    {"type": "number"},
                                    {"type": "string"},
                                    {"type": "boolean"},
                                    {"enum": [None]}
                                ]
                            }
                        },
                        "required": ["name"]
                    },
                    "minItems": 1,
                    "maxItems": 16,  # Límite razonable de parámetros
                    "description": "Parámetros del kernel"
                }
            },
            "required": ["name", "parameters"]
        }
    },
    "required": ["imports", "decorator", "function"],
    "additionalProperties": False
}
```

## Generador de Código: De JSON a Python

Ahora creamos un generador que convierte el JSON estructurado de vuelta a código Python:

```{code-cell} ipython3
# generators/triton_generator.py

import json
from typing import Dict, List, Any

class TritonCodeGenerator:
    """Genera código Triton a partir de JSON estructurado"""

    def __init__(self):
        pass

    def generate_imports(self, imports_list: List[Dict]) -> str:
        """Generar sección de imports"""

        lines = []
        for imp in imports_list:
            if imp["type"] == "import_triton":
                lines.append("import triton")
            elif imp["type"] == "import_triton_language":
                as_name = imp.get("as_name", "tl")
                lines.append(f"import triton.language as {as_name}")

        return "\n".join(lines)

    def generate_decorator(self, decorator_spec: Dict) -> str:
        """Generar decorador"""

        code = "@triton.jit"

        if "kwargs" in decorator_spec and decorator_spec["kwargs"]:
            kwargs_str = ", ".join(
                f"{k}={self._format_value(v)}"
                for k, v in decorator_spec["kwargs"].items()
            )
            code += f"({kwargs_str})"

        return code

    def generate_function_signature(self, function_spec: Dict) -> str:
        """Generar firma de función"""

        name = function_spec["name"]
        params = function_spec.get("parameters", [])

        # Generar lista de parámetros
        param_strs = []
        for param in params:
            param_str = param["name"]

            # Agregar type hint si existe
            if "type_hint" in param:
                param_str += f": {param['type_hint']}"

            # Agregar default si existe
            if "default" in param and param["default"] is not None:
                param_str += f"={self._format_value(param['default'])}"

            param_strs.append(param_str)

        params_str = ", ".join(param_strs)
        return f"def {name}({params_str}):"

    def generate(self, spec: Dict) -> str:
        """Generar código Triton completo (L1-L2)"""

        lines = [
            self.generate_imports(spec["imports"]),
            "",  # Línea en blanco
            self.generate_decorator(spec["decorator"]),
            self.generate_function_signature(spec["function"]),
            "    pass  # Cuerpo del kernel (L3-L4 por venir)",
        ]

        return "\n".join(lines)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Formatear un valor para código Python"""

        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif value is None:
            return "None"
        else:
            return str(value)

# Uso
if __name__ == "__main__":
    spec = {
        "imports": [
            {"type": "import_triton"},
            {"type": "import_triton_language", "as_name": "tl"}
        ],
        "decorator": {
            "name": "triton.jit",
            "kwargs": {}
        },
        "function": {
            "name": "kernel_add",
            "parameters": [
                {"name": "x_ptr"},
                {"name": "y_ptr"},
                {"name": "output_ptr"},
                {"name": "n"},
                {"name": "BLOCK_SIZE", "type_hint": "tl.constexpr", "default": 1024}
            ]
        }
    }

    generator = TritonCodeGenerator()
    code = generator.generate(spec)
    print(code)

    # Output:
    # import triton
    # import triton.language as tl
    #
    # @triton.jit
    # def kernel_add(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr=1024):
    #     pass  # Cuerpo del kernel (L3-L4 por venir)
```

## Validación L1-L2

Tests para verificar que nuestra gramática funciona:

```{code-cell} ipython3
# tests/test_l1_l2.py

import pytest
import json
from generators.triton_generator import TritonCodeGenerator
from schemas.triton_l1_l2_schema import TRITON_L1_L2_SCHEMA
from jsonschema import validate, ValidationError

class TestTritonL1L2:
    """Tests para importes, decoradores y firma"""

    def test_valid_basic_kernel(self):
        """Kernel válido más simple"""
        spec = {
            "imports": [
                {"type": "import_triton"},
                {"type": "import_triton_language"}
            ],
            "decorator": {"name": "triton.jit"},
            "function": {
                "name": "kernel_basic",
                "parameters": [
                    {"name": "x_ptr"},
                    {"name": "y_ptr"}
                ]
            }
        }

        validate(instance=spec, schema=TRITON_L1_L2_SCHEMA)

        # Generar código
        generator = TritonCodeGenerator()
        code = generator.generate(spec)
        assert "import triton" in code
        assert "@triton.jit" in code
        assert "def kernel_basic" in code

    def test_kernel_with_constexpr(self):
        """Kernel con parámetros constexpr"""
        spec = {
            "imports": [
                {"type": "import_triton"},
                {"type": "import_triton_language"}
            ],
            "decorator": {"name": "triton.jit"},
            "function": {
                "name": "kernel_with_const",
                "parameters": [
                    {"name": "x_ptr"},
                    {"name": "BLOCK_SIZE", "type_hint": "tl.constexpr"}
                ]
            }
        }

        validate(instance=spec, schema=TRITON_L1_L2_SCHEMA)

        generator = TritonCodeGenerator()
        code = generator.generate(spec)
        assert "tl.constexpr" in code

    def test_kernel_with_defaults(self):
        """Kernel con valores por defecto"""
        spec = {
            "imports": [
                {"type": "import_triton"},
                {"type": "import_triton_language"}
            ],
            "decorator": {"name": "triton.jit"},
            "function": {
                "name": "kernel_defaults",
                "parameters": [
                    {"name": "x_ptr"},
                    {"name": "BLOCK_SIZE", "type_hint": "tl.constexpr", "default": 1024}
                ]
            }
        }

        validate(instance=spec, schema=TRITON_L1_L2_SCHEMA)

        generator = TritonCodeGenerator()
        code = generator.generate(spec)
        assert "BLOCK_SIZE: tl.constexpr=1024" in code

    def test_invalid_missing_imports(self):
        """Debe fallar: faltan imports"""
        spec = {
            "imports": [{"type": "import_triton"}],  # Solo uno
            "decorator": {"name": "triton.jit"},
            "function": {
                "name": "kernel_test",
                "parameters": [{"name": "x"}]
            }
        }

        with pytest.raises(ValidationError):
            validate(instance=spec, schema=TRITON_L1_L2_SCHEMA)

    def test_invalid_missing_decorator(self):
        """Debe fallar: falta decorator"""
        spec = {
            "imports": [
                {"type": "import_triton"},
                {"type": "import_triton_language"}
            ],
            # Falta decorator
            "function": {
                "name": "kernel_test",
                "parameters": [{"name": "x"}]
            }
        }

        with pytest.raises(ValidationError):
            validate(instance=spec, schema=TRITON_L1_L2_SCHEMA)

    def test_invalid_kernel_name(self):
        """Nombre de kernel no válido"""
        spec = {
            "imports": [
                {"type": "import_triton"},
                {"type": "import_triton_language"}
            ],
            "decorator": {"name": "triton.jit"},
            "function": {
                "name": "not_kernel_123",  # No empieza con kernel_
                "parameters": [{"name": "x"}]
            }
        }

        with pytest.raises(ValidationError):
            validate(instance=spec, schema=TRITON_L1_L2_SCHEMA)

    def test_too_many_parameters(self):
        """Demasiados parámetros"""
        spec = {
            "imports": [
                {"type": "import_triton"},
                {"type": "import_triton_language"}
            ],
            "decorator": {"name": "triton.jit"},
            "function": {
                "name": "kernel_test",
                "parameters": [
                    {"name": f"p{i}"}
                    for i in range(20)  # Más que maxItems: 16
                ]
            }
        }

        with pytest.raises(ValidationError):
            validate(instance=spec, schema=TRITON_L1_L2_SCHEMA)
```

## Ejercicios

1. **Extiende el schema**:
   - Agrega soporte para otros decoradores (ej: `@triton.autotune`)
   - Permite múltiples decoradores
   - Actualiza el generador

2. **Crea un parser inverso**:
   - Lee código Triton Python
   - Extrae la sección L1-L2
   - Genera el JSON estructurado

3. **Test contra corpus**:
   - Toma kernels del corpus de la lectura anterior
   - Extrae su L1-L2
   - ¿Tu schema los acepta?

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- L1-L2 define el esqueleto estructural: imports obligatorios (triton, triton.language) y decoradores (@triton.jit)
- Firmas de función especifican parámetros con/sin tipos (punteros sin tipo, constexprs con tl.constexpr)
- JSON Schema permite validar automáticamente que el LLM genera estructura correcta
- Generadores transforman JSON → código Python, separando lógica de presentación
- Separar L1-L2 de L3-L4 facilita debugging: sabes si el error es estructural o lógico

**Siguiente paso:** En la próxima lectura completaremos las gramáticas con **L3-L4: Expresiones y Control Flow**, donde definiremos operaciones aritméticas, llamadas a tl.load/store, y estructuras if/for/while para la lógica del kernel.
```

```{admonition} Verifica tu comprensión
:class: note
1. ¿Por qué separar L1-L2 (estructura) de L3-L4 (lógica)? ¿Qué ventajas tiene para debugging?
2. Diseña un schema JSON para un kernel que usa @triton.autotune con configs
3. ¿Qué pasa si omites el decorador @triton.jit? ¿Lo previene el schema?
4. Extiende el generador para soportar múltiples decoradores (autotune + jit)
```

## Conexión con el Proyecto

La gramática L1-L2 es el **esqueleto estructural** de todo kernel Triton. Sin imports correctos y decoradores, el código no compila. Sin una firma bien definida, el kernel no puede ejecutarse.

**Importancia en KernelAgent**:
1. **Validación de estructura**: XGrammar garantiza que el LLM siempre genere imports y decoradores correctos
2. **Firmas consistentes**: Los parámetros siguen convenciones (punteros sin tipo, constexprs con tipo)
3. **Base para L3-L4**: Una vez que L1-L2 está bien, puedes enfocarte en la lógica del kernel

**Flujo completo**:
```
Prompt usuario → LLM genera JSON (L1-L2) → XGrammar valida →
Generador crea código Python → Compilador Triton → Kernel ejecutable
```

**Casos de uso práctico**:
- Usuario pide "kernel que suma dos vectores"
- LLM genera spec L1-L2: imports, @triton.jit, función con params
- Tu código verifica que los params tienen sentido
- L3-L4 agrega la lógica (próximas lecturas)
- Resultado: kernel funcional

**Por qué separar L1-L2 de L3-L4**:
- Separación de concerns: estructura vs lógica
- Más fácil de debuggear: si falla, sabes dónde buscar
- Permite generación incremental: primero estructura, luego lógica

## Ejercicio Práctico: Modificar Gramática L1-L2

```{code-cell} ipython3
import json
from jsonschema import validate, ValidationError

# Vamos a extender el schema L1-L2 para soportar múltiples decoradores
# Caso real: @triton.autotune seguido de @triton.jit

EXTENDED_L1_L2_SCHEMA = {
    "type": "object",
    "properties": {
        "imports": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["import_triton", "import_triton_language"]},
                    "as_name": {"type": "string"}
                },
                "required": ["type"]
            },
            "minItems": 2,
            "maxItems": 2
        },
        "decorators": {  # Ahora plural - múltiples decoradores
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"enum": ["triton.jit", "triton.autotune"]},
                    "kwargs": {
                        "type": "object",
                        "additionalProperties": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "array"}  # Para configs de autotune
                            ]
                        }
                    }
                },
                "required": ["name"]
            },
            "minItems": 1,
            "maxItems": 3,
            "description": "Uno o más decoradores"
        },
        "function": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^kernel_[a-z_][a-z0-9_]*$"
                },
                "parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type_hint": {"oneOf": [
                                {"enum": ["tl.constexpr"]},
                                {"type": "string"}
                            ]},
                            "default": {"oneOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"}
                            ]}
                        },
                        "required": ["name"]
                    },
                    "minItems": 1,
                    "maxItems": 16
                }
            },
            "required": ["name", "parameters"]
        }
    },
    "required": ["imports", "decorators", "function"],
    "additionalProperties": False
}

# Caso 1: Kernel simple con solo @triton.jit
simple_kernel_spec = {
    "imports": [
        {"type": "import_triton"},
        {"type": "import_triton_language", "as_name": "tl"}
    ],
    "decorators": [
        {"name": "triton.jit"}
    ],
    "function": {
        "name": "kernel_add",
        "parameters": [
            {"name": "x_ptr"},
            {"name": "y_ptr"},
            {"name": "output_ptr"},
            {"name": "n", "type_hint": "tl.constexpr"}
        ]
    }
}

# Caso 2: Kernel con autotune
autotuned_kernel_spec = {
    "imports": [
        {"type": "import_triton"},
        {"type": "import_triton_language", "as_name": "tl"}
    ],
    "decorators": [
        {
            "name": "triton.autotune",
            "kwargs": {
                "configs": [
                    {"BLOCK_SIZE": 128},
                    {"BLOCK_SIZE": 256}
                ],
                "key": ["n"]
            }
        },
        {
            "name": "triton.jit"
        }
    ],
    "function": {
        "name": "kernel_matmul",
        "parameters": [
            {"name": "a_ptr"},
            {"name": "b_ptr"},
            {"name": "c_ptr"},
            {"name": "M", "type_hint": "tl.constexpr"},
            {"name": "N", "type_hint": "tl.constexpr"},
            {"name": "K", "type_hint": "tl.constexpr"},
            {"name": "BLOCK_SIZE", "type_hint": "tl.constexpr", "default": 128}
        ]
    }
}

# Validar ambos casos
print("Validando kernel simple...")
try:
    validate(instance=simple_kernel_spec, schema=EXTENDED_L1_L2_SCHEMA)
    print("OK: Kernel simple válido")
except ValidationError as e:
    print(f"ERROR: {e.message}")

print("\nValidando kernel con autotune...")
try:
    validate(instance=autotuned_kernel_spec, schema=EXTENDED_L1_L2_SCHEMA)
    print("OK: Kernel con autotune válido")
except ValidationError as e:
    print(f"ERROR: {e.message}")

# Generador actualizado para múltiples decoradores
class ExtendedTritonCodeGenerator:
    """Genera código Triton con soporte para múltiples decoradores"""

    def generate_decorators(self, decorators_list):
        """Generar múltiples decoradores"""
        lines = []
        for dec in decorators_list:
            code = f"@{dec['name'].replace('.', '.')}"  # triton.jit

            if "kwargs" in dec and dec["kwargs"]:
                # Formatear kwargs
                kwargs_parts = []
                for k, v in dec["kwargs"].items():
                    if isinstance(v, list):
                        # Para configs de autotune
                        v_str = str(v).replace("'", "")
                    elif isinstance(v, str):
                        v_str = f'"{v}"'
                    else:
                        v_str = str(v)
                    kwargs_parts.append(f"{k}={v_str}")

                kwargs_str = ", ".join(kwargs_parts)
                code += f"({kwargs_str})"

            lines.append(code)

        return "\n".join(lines)

    def generate_function_signature(self, function_spec):
        """Generar firma de función"""
        name = function_spec["name"]
        params = function_spec.get("parameters", [])

        param_strs = []
        for param in params:
            param_str = param["name"]

            if "type_hint" in param:
                param_str += f": {param['type_hint']}"

            if "default" in param and param["default"] is not None:
                default_val = param["default"]
                if isinstance(default_val, str):
                    default_val = f'"{default_val}"'
                param_str += f"={default_val}"

            param_strs.append(param_str)

        params_str = ", ".join(param_strs)
        return f"def {name}({params_str}):"

# Generar código para el kernel con autotune
generator = ExtendedTritonCodeGenerator()
print("\n" + "="*60)
print("CÓDIGO GENERADO PARA KERNEL CON AUTOTUNE:")
print("="*60)
print("import triton")
print("import triton.language as tl")
print()
print(generator.generate_decorators(autotuned_kernel_spec["decorators"]))
print(generator.generate_function_signature(autotuned_kernel_spec["function"]))
print("    pass  # Cuerpo del kernel (L3-L4)")

print("\n" + "="*60)
print("ANÁLISIS DEL EJERCICIO")
print("="*60)
print("1. Extendimos el schema para soportar múltiples decoradores")
print("2. El generador ahora maneja decoradores complejos (autotune)")
print("3. XGrammar compilaría este schema y forzaría al LLM a seguirlo")
print("4. Resultado: kernels más sofisticados con autotuning automático")
```

**Observaciones importantes**:
1. Múltiples decoradores se aplican en orden (autotune ANTES de jit)
2. El schema ahora es más flexible pero sigue siendo restrictivo
3. XGrammar garantiza que el LLM no puede generar decoradores inválidos

**Tarea adicional**:
1. Agrega soporte para el decorador `@triton.heuristics`
2. Crea un test case con los tres decoradores: autotune, heuristics, jit
3. Genera el código Python resultante
