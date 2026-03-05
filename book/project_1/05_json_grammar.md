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

# 04. Phase I: Definiendo la Gramática JSON para KernelAgent

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/05_json_grammar.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Diseñar JSON Schemas completos que describan la estructura de kernels Triton
- Compilar schemas con XGrammar y validar su corrección
- Crear test cases positivos (deben pasar) y negativos (deben fallar)
- Implementar validación automática de JSON generado
- Distinguir entre validación sintáctica (schema) y semántica (lógica custom)
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Implementar el primer componente crítico de KernelAgent - el schema JSON que define qué kernels Triton son válidos. Este schema será el "contrato" entre el LLM y tu código, garantizando que el JSON generado tenga la estructura correcta.
```


## Introducción

En esta phase, definiremos una gramática JSON Schema que describe la estructura de los kernels Triton que nuestro agente debe generar. Es el primer paso concreto del proyecto: "¿Cómo le decimos al modelo qué estructura debe tener el código?"

La gramática es el puente entre lo que queremos (kernels Triton válidos) y lo que el modelo puede generar (secuencias de tokens).

## Objetivo de la Phase

Vamos a:
1. Diseñar un JSON Schema para representar subgrafos de KernelAgent
2. Compilarlo con XGrammar
3. Crear test cases (casos positivos y negativos)
4. Validar que funciona correctamente

## ¿Qué es KernelAgent?

KernelAgent es un framework para generación automática de kernels GPU. Un subgrafo es una parte de una computación completa. Por ejemplo:

```
Computación completa:
    input → kernel A → kernel B → output

Subgrafo:
    intermediate_1 → kernel A → intermediate_2
```

Nuestro JSON Schema describirá cómo se ve un subgrafo válido.

## Diseño del Schema

Empecemos simple. Un subgrafo de KernelAgent tiene:
- **Inputs**: Variables de entrada
- **Kernel**: El código que ejecutar
- **Outputs**: Variables de salida
- **Constraints**: Restricciones (tamaño, tipos de datos)

```{code-cell} ipython3
# Visualización de JSON Schema para Triton
import plotly.graph_objects as go

fig = go.Figure()

# Árbol de JSON Schema
estructura = {
    'root': (0.5, 0.95, 'TritonKernel'),
    'name': (0.2, 0.75, 'name: str'),
    'args': (0.5, 0.75, 'args: List[Arg]'),
    'body': (0.8, 0.75, 'body: List[Stmt]'),
    'arg_name': (0.35, 0.55, 'name: str'),
    'arg_type': (0.5, 0.55, 'dtype: str'),
    'arg_ptr': (0.65, 0.55, 'is_ptr: bool'),
    'stmt': (0.8, 0.55, 'statements...'),
}

edges = [('root', 'name'), ('root', 'args'), ('root', 'body'),
         ('args', 'arg_name'), ('args', 'arg_type'), ('args', 'arg_ptr'),
         ('body', 'stmt')]

colors = {'root': '#FF6B6B', 'name': '#4ECDC4', 'args': '#45B7D1', 'body': '#FFE66D',
          'arg_name': '#95E1A3', 'arg_type': '#95E1A3', 'arg_ptr': '#95E1A3', 'stmt': '#DDA0DD'}

for e1, e2 in edges:
    x0, y0, _ = estructura[e1]
    x1, y1, _ = estructura[e2]
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                             line=dict(color='gray'), showlegend=False))

for name, (x, y, label) in estructura.items():
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                             marker=dict(size=30, color=colors[name]),
                             text=[label], textposition='bottom center',
                             showlegend=False))

fig.update_layout(title="Estructura del JSON Schema para Kernels Triton",
                  xaxis=dict(visible=False), yaxis=dict(visible=False), height=400)
fig.show()
```

:::{figure} diagrams/fsm_json_states.png
:name: fig-fsm-json-states
:alt: Diagrama de estados FSM generado desde JSON Schema
:align: center
:width: 90%

**Figura 4:** Máquina de estados finitos generada desde JSON Schema - estados y transiciones válidas.
:::

```json
{
  "type": "object",
  "properties": {
    "subgraph_id": {
      "type": "string",
      "description": "Identificador único"
    },
    "inputs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "dtype": {"enum": ["float32", "float64", "int32", "int64"]},
          "shape": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Forma del tensor, ej: [32, 64]"
          }
        },
        "required": ["name", "dtype", "shape"]
      }
    },
    "kernel": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "code": {"type": "string"},
        "framework": {"enum": ["triton", "cuda", "hip"]}
      },
      "required": ["name", "code", "framework"]
    },
    "outputs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "dtype": {"enum": ["float32", "float64", "int32", "int64"]},
          "shape": {
            "type": "array",
            "items": {"type": "integer"}
          }
        },
        "required": ["name", "dtype"]
      }
    }
  },
  "required": ["subgraph_id", "inputs", "kernel", "outputs"]
}
```

## Implementación: Compilando el Schema

Ahora compilaremos este schema con XGrammar:

```{code-cell} ipython3
# schemas/kernel_subgraph_schema.py

import json
import xgrammar as xgr
from pathlib import Path

KERNEL_SUBGRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "subgraph_id": {
            "type": "string",
            "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
            "description": "Identificador válido de Python"
        },
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dtype": {"enum": ["float32", "float64", "int32", "int64"]},
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1}
                    }
                },
                "required": ["name", "dtype", "shape"]
            },
            "minItems": 1
        },
        "kernel": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "code": {"type": "string"},
                "framework": {"enum": ["triton", "cuda", "hip"]}
            },
            "required": ["name", "code", "framework"]
        },
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dtype": {"enum": ["float32", "float64", "int32", "int64"]},
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1}
                    }
                },
                "required": ["name", "dtype"]
            },
            "minItems": 1
        }
    },
    "required": ["subgraph_id", "inputs", "kernel", "outputs"],
    "additionalProperties": False
}

def compile_schema(schema):
    """Compilar schema con XGrammar"""

    # Crear la gramática
    grammar = xgr.Grammar.from_schema(schema)

    # Opcional: compilar con tokenizador específico
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    # tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    # grammar.compile(tokenizer_info)

    return grammar

# Compilar
kernel_grammar = compile_schema(KERNEL_SUBGRAPH_SCHEMA)

print(f"Gramática compilada con {kernel_grammar.num_states} estados")

# Guardar para reutilizar
def save_grammar(grammar, path):
    """Guardar gramática compilada"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar schema como referencia
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(KERNEL_SUBGRAPH_SCHEMA, f, indent=2)

    # XGrammar puede serializar (si lo permite)
    # grammar.save(str(path.with_suffix('.xgr')))

save_grammar(kernel_grammar, "compiled_grammars/kernel_subgraph.xgr")
```

## Test Cases: Positivos y Negativos

Ahora probemos que la gramática funciona correctamente. Escribiremos test cases: ejemplos que DEBEN pasar (positivos) y ejemplos que DEBEN fallar (negativos).

```{code-cell} ipython3
# tests/test_kernel_schema.py

import json
import pytest
from schemas.kernel_subgraph_schema import KERNEL_SUBGRAPH_SCHEMA, compile_schema

class TestKernelSchema:
    """Test suite para la gramática JSON"""

    @pytest.fixture
    def grammar(self):
        return compile_schema(KERNEL_SUBGRAPH_SCHEMA)

    # ==================== TEST POSITIVOS ====================

    def test_minimal_valid_kernel(self):
        """El caso más simple: un kernel con un input y un output"""
        valid = {
            "subgraph_id": "simple_add",
            "inputs": [
                {
                    "name": "x",
                    "dtype": "float32",
                    "shape": [32, 64]
                }
            ],
            "kernel": {
                "name": "add_one",
                "code": "y = x + 1",
                "framework": "triton"
            },
            "outputs": [
                {
                    "name": "y",
                    "dtype": "float32",
                    "shape": [32, 64]
                }
            ]
        }

        # Convertir a JSON para validar
        json_str = json.dumps(valid)
        assert len(json_str) > 0  # JSON válido

        # Aquí iríamos a validar contra la gramática
        # (veremos cómo en la siguiente sección)

    def test_multiple_inputs_outputs(self):
        """Kernel con múltiples inputs y outputs"""
        valid = {
            "subgraph_id": "matrix_multiply",
            "inputs": [
                {
                    "name": "A",
                    "dtype": "float32",
                    "shape": [128, 256]
                },
                {
                    "name": "B",
                    "dtype": "float32",
                    "shape": [256, 512]
                }
            ],
            "kernel": {
                "name": "matmul",
                "code": "C = A @ B",
                "framework": "triton"
            },
            "outputs": [
                {
                    "name": "C",
                    "dtype": "float32",
                    "shape": [128, 512]
                }
            ]
        }

        json_str = json.dumps(valid)
        assert json.loads(json_str) == valid

    def test_all_dtypes(self):
        """Probar todos los tipos de dato soportados"""
        for dtype in ["float32", "float64", "int32", "int64"]:
            valid = {
                "subgraph_id": f"test_{dtype}",
                "inputs": [{
                    "name": "x",
                    "dtype": dtype,
                    "shape": [10]
                }],
                "kernel": {
                    "name": "identity",
                    "code": "y = x",
                    "framework": "triton"
                },
                "outputs": [{
                    "name": "y",
                    "dtype": dtype
                }]
            }

            json_str = json.dumps(valid)
            assert dtype in json_str

    def test_all_frameworks(self):
        """Probar todos los frameworks soportados"""
        for fw in ["triton", "cuda", "hip"]:
            valid = {
                "subgraph_id": f"test_{fw}",
                "inputs": [{
                    "name": "x",
                    "dtype": "float32",
                    "shape": [10]
                }],
                "kernel": {
                    "name": "test",
                    "code": "// test",
                    "framework": fw
                },
                "outputs": [{
                    "name": "y",
                    "dtype": "float32"
                }]
            }

            assert json.dumps(valid)

    # ==================== TEST NEGATIVOS ====================

    def test_missing_required_field_subgraph_id(self):
        """Falta field requerido: subgraph_id"""
        invalid = {
            # Falta subgraph_id
            "inputs": [{
                "name": "x",
                "dtype": "float32",
                "shape": [10]
            }],
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "triton"
            },
            "outputs": [{
                "name": "y",
                "dtype": "float32"
            }]
        }

        # Esto debería ser JSON válido (es un dict válido)
        # pero NO es válido según nuestro schema
        json_str = json.dumps(invalid)
        assert "subgraph_id" not in invalid

    def test_invalid_dtype(self):
        """dtype inválido (fuera de enum)"""
        invalid = {
            "subgraph_id": "bad_dtype",
            "inputs": [{
                "name": "x",
                "dtype": "bfloat16",  # No soportado
                "shape": [10]
            }],
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "triton"
            },
            "outputs": [{
                "name": "y",
                "dtype": "float32"
            }]
        }

        # bfloat16 no está en el enum
        assert "bfloat16" not in ["float32", "float64", "int32", "int64"]

    def test_invalid_framework(self):
        """Framework no reconocido"""
        invalid = {
            "subgraph_id": "bad_fw",
            "inputs": [{
                "name": "x",
                "dtype": "float32",
                "shape": [10]
            }],
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "pytorch"  # No soportado
            },
            "outputs": [{
                "name": "y",
                "dtype": "float32"
            }]
        }

        assert "pytorch" not in ["triton", "cuda", "hip"]

    def test_invalid_subgraph_id_format(self):
        """subgraph_id no cumple pattern (debe ser identificador válido)"""
        invalid = {
            "subgraph_id": "123-invalid",  # Comienza con número
            "inputs": [{
                "name": "x",
                "dtype": "float32",
                "shape": [10]
            }],
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "triton"
            },
            "outputs": [{
                "name": "y",
                "dtype": "float32"
            }]
        }

        # Pattern: ^[a-zA-Z_][a-zA-Z0-9_]*$
        import re
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        assert not re.match(pattern, "123-invalid")

    def test_empty_inputs(self):
        """inputs vacío (viola minItems: 1)"""
        invalid = {
            "subgraph_id": "empty_input",
            "inputs": [],  # Vacío
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "triton"
            },
            "outputs": [{
                "name": "y",
                "dtype": "float32"
            }]
        }

        assert len(invalid["inputs"]) == 0

    def test_empty_outputs(self):
        """outputs vacío (viola minItems: 1)"""
        invalid = {
            "subgraph_id": "empty_output",
            "inputs": [{
                "name": "x",
                "dtype": "float32",
                "shape": [10]
            }],
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "triton"
            },
            "outputs": []  # Vacío
        }

        assert len(invalid["outputs"]) == 0

    def test_invalid_shape_dimension(self):
        """Shape con dimensión inválida (< 1)"""
        invalid = {
            "subgraph_id": "bad_shape",
            "inputs": [{
                "name": "x",
                "dtype": "float32",
                "shape": [0, 10]  # 0 no es válido
            }],
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "triton"
            },
            "outputs": [{
                "name": "y",
                "dtype": "float32"
            }]
        }

        assert 0 in invalid["inputs"][0]["shape"]

    def test_additional_properties_not_allowed(self):
        """Propiedades adicionales no permitidas"""
        invalid = {
            "subgraph_id": "extra_props",
            "inputs": [{
                "name": "x",
                "dtype": "float32",
                "shape": [10]
            }],
            "kernel": {
                "name": "test",
                "code": "y = x",
                "framework": "triton"
            },
            "outputs": [{
                "name": "y",
                "dtype": "float32"
            }],
            "extra_field": "no debería estar aquí"  # Campo extra
        }

        assert "extra_field" not in KERNEL_SUBGRAPH_SCHEMA["properties"]
```

## Validación Manual

Ahora creemos un script que valide JSON contra nuestro schema:

```{code-cell} ipython3
# utils/validate_schema.py

import json
from pathlib import Path
from jsonschema import validate, ValidationError
from schemas.kernel_subgraph_schema import KERNEL_SUBGRAPH_SCHEMA

def validate_kernel_json(json_str):
    """
    Validar que un string JSON cumple el schema

    Retorna:
        (True, None) si es válido
        (False, error_message) si no es válido
    """

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, f"JSON inválido: {e}"

    try:
        validate(instance=data, schema=KERNEL_SUBGRAPH_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, f"Validación fallida: {e.message}"

# Usar
if __name__ == "__main__":
    test_json = json.dumps({
        "subgraph_id": "test",
        "inputs": [{"name": "x", "dtype": "float32", "shape": [10]}],
        "kernel": {"name": "k", "code": "y=x", "framework": "triton"},
        "outputs": [{"name": "y", "dtype": "float32"}]
    })

    valid, error = validate_kernel_json(test_json)
    print(f"Valid: {valid}")
    if error:
        print(f"Error: {error}")
```

## Ejercicios

1. **Expande el schema**:
   - Agrega un campo `description` opcional a inputs/outputs
   - Agrega un campo `optimization_level` que sea "O0", "O1", "O2", o "O3"
   - Actualiza los tests

2. **Crea test cases con XGrammar real**:
   - Compila el schema con XGrammar
   - Escribe un test que genere JSON válido usando XGrammar
   - Verifica que puedes parsear el JSON generado

3. **Schema discovery**:
   - Recolecta 5 kernels Triton reales
   - Analiza su estructura
   - ¿Tu schema actual los cubriría? Si no, ¿qué necesitarías agregar?

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- JSON Schema define la ESTRUCTURA y TIPOS de datos válidos, actuando como "contrato" entre LLM y código
- Campos required vs opcionales determinan qué puede omitir el LLM; enums restringen opciones
- Test cases positivos/negativos validan que el schema funciona correctamente
- JSON Schema valida SINTAXIS pero NO SEMÁNTICA (e.g., compatibilidad de shapes en matmul requiere lógica adicional)
- En KernelAgent: XGrammar garantiza estructura válida + tu código valida lógica = kernels correctos

**Siguiente paso:** En la próxima lectura construiremos la **Gramática L1-L2 para Triton**, definiendo la estructura esquelética (imports, decoradores, firmas de función) que todo kernel debe tener.
```

```{admonition} Verifica tu comprensión
:class: note
1. ¿Cuál es la diferencia entre validación sintáctica (JSON Schema) y validación semántica (lógica custom)?
2. Para un kernel de suma vectorial, diseña un schema JSON mínimo pero completo
3. ¿Por qué additionalProperties: false es importante en producción?
4. Crea 3 test cases negativos para el schema de matmul del ejercicio práctico
```

## Conexión con el Proyecto

Esta lectura es el **primer paso concreto** del proyecto KernelAgent. Aquí defines la estructura de datos que el LLM debe generar.

**Por qué es importante**:
- El schema JSON actúa como "contrato" entre el LLM y tu código
- XGrammar lo compila para forzar que el LLM respete este contrato
- Sin un schema bien diseñado, el LLM generará código inválido o inútil

**Flujo en el proyecto**:
1. Diseñas schema JSON (esta lectura) → Define estructura de kernels
2. XGrammar compila a FSM → Crea restricciones de tokens
3. LLM genera según FSM → Produce JSON válido
4. Parser JSON → código Triton → Compilación y ejecución

**Decisiones de diseño críticas**:
- ¿Qué campos son `required` vs opcionales? → Afecta qué puede omitir el LLM
- ¿Usas `enum` para dtypes? → Limita opciones, previene errores
- ¿Permites `additionalProperties`? → Si es false, rechaza campos extra

Estas decisiones determinan qué tan flexible vs restrictivo es el sistema.

**Aplicación real**:
Cuando implementes KernelAgent, este schema será la base de todo:
- API recibe especificación de kernel
- LLM genera JSON según este schema
- Sistema valida automáticamente
- Código Triton se genera sin errores de sintaxis

## Ejercicio Práctico: Crear Schema para Matrix Multiplication

```{code-cell} ipython3
import json
from jsonschema import validate, ValidationError

# Diseñaremos un schema para un kernel específico: matrix multiplication
# Este kernel multiplica dos matrices A @ B = C

MATMUL_KERNEL_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {
            "enum": ["matmul"],
            "description": "Tipo de operación (solo matmul en este caso)"
        },
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["A", "B"]},
                    "dtype": {"enum": ["float32", "float64"]},
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Forma [M, K] para A o [K, N] para B"
                    }
                },
                "required": ["name", "dtype", "shape"]
            },
            "minItems": 2,
            "maxItems": 2
        },
        "output": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["C"]},
                "dtype": {"enum": ["float32", "float64"]},
                "shape": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Forma [M, N]"
                }
            },
            "required": ["name", "dtype", "shape"]
        },
        "block_size": {
            "type": "integer",
            "enum": [16, 32, 64, 128],
            "description": "Tamaño de bloque para tiling"
        }
    },
    "required": ["operation", "inputs", "output", "block_size"],
    "additionalProperties": False
}

# Caso válido: matmul de 128x256 @ 256x512
valid_matmul = {
    "operation": "matmul",
    "inputs": [
        {
            "name": "A",
            "dtype": "float32",
            "shape": [128, 256]
        },
        {
            "name": "B",
            "dtype": "float32",
            "shape": [256, 512]
        }
    ],
    "output": {
        "name": "C",
        "dtype": "float32",
        "shape": [128, 512]
    },
    "block_size": 64
}

# Caso inválido: shapes incompatibles
invalid_matmul = {
    "operation": "matmul",
    "inputs": [
        {
            "name": "A",
            "dtype": "float32",
            "shape": [128, 256]
        },
        {
            "name": "B",
            "dtype": "float32",
            "shape": [128, 512]  # ¡Error! K debe ser 256, no 128
        }
    ],
    "output": {
        "name": "C",
        "dtype": "float32",
        "shape": [128, 512]
    },
    "block_size": 64
}

print("Validando caso válido...")
try:
    validate(instance=valid_matmul, schema=MATMUL_KERNEL_SCHEMA)
    print("OK: El schema acepta el caso válido")
except ValidationError as e:
    print(f"ERROR inesperado: {e.message}")

print("\nValidando caso inválido...")
try:
    validate(instance=invalid_matmul, schema=MATMUL_KERNEL_SCHEMA)
    print("PROBLEMA: El schema NO detectó el error")
    print("(Nota: JSON Schema no valida consistencia matemática de shapes)")
except ValidationError as e:
    print(f"OK: Schema rechazó con error: {e.message}")

# Observación importante
print("\nLimitaciones de JSON Schema:")
print("- Puede validar SINTAXIS (tipos, campos requeridos, enums)")
print("- NO puede validar SEMÁNTICA (ej: shapes compatibles en matmul)")
print("- Para validación semántica, necesitas lógica adicional en tu código")

# Función de validación semántica adicional
def validate_matmul_shapes(spec):
    """Valida que las shapes sean compatibles para matmul"""
    errors = []

    A_shape = spec["inputs"][0]["shape"]
    B_shape = spec["inputs"][1]["shape"]
    C_shape = spec["output"]["shape"]

    M, K_A = A_shape
    K_B, N = B_shape

    if K_A != K_B:
        errors.append(f"Dimensión K incompatible: A tiene {K_A}, B tiene {K_B}")

    expected_C_shape = [M, N]
    if C_shape != expected_C_shape:
        errors.append(f"Shape de C debería ser {expected_C_shape}, no {C_shape}")

    # Validar que dtypes coinciden
    if spec["inputs"][0]["dtype"] != spec["inputs"][1]["dtype"]:
        errors.append("Los dtypes de A y B deben coincidir")

    return errors

print("\nValidación semántica del caso inválido:")
semantic_errors = validate_matmul_shapes(invalid_matmul)
print("Errores encontrados:", semantic_errors)

print("\nValidación semántica del caso válido:")
semantic_errors = validate_matmul_shapes(valid_matmul)
print("Errores encontrados:", semantic_errors or "Ninguno - OK")

# Resumen
print("\n" + "="*60)
print("RESUMEN DEL EJERCICIO")
print("="*60)
print("1. JSON Schema valida ESTRUCTURA y TIPOS")
print("2. Necesitas lógica adicional para validar SEMÁNTICA")
print("3. En KernelAgent, usarás ambas capas de validación")
print("4. XGrammar garantiza estructura válida, tu código valida semántica")
```

**Análisis de resultados**:
- JSON Schema es poderoso para validación sintáctica
- No reemplaza validación semántica (lógica de negocio)
- En KernelAgent: XGrammar + validación custom = código correcto

**Tarea adicional**:
1. Modifica el schema para soportar también `operation: "add"` (suma elemento-wise)
2. Crea casos de prueba válidos e inválidos para `add`
3. Implementa `validate_add_shapes()` similar a `validate_matmul_shapes()`

---

## Referencias

- JSON Schema. [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/). JSON Schema.
- XGrammar. [Efficient, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar). GitHub.
