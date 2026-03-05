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

# 03. Internos de XGrammar: El Pipeline de Compilación


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/04_xgrammar_constrained.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Explicar el pipeline completo de XGrammar: Parser → AST → FSM → LogitsProcessor
- Comprender cómo las gramáticas JSON Schema se transforman en máquinas de estados finitos
- Integrar TokenizerInfo para adaptar gramáticas a tokenizadores específicos
- Depurar gramáticas inspeccionando estados y transiciones del FSM
- Evaluar trade-offs entre gramáticas simples (FSM pequeño) vs complejas (FSM grande)
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Compilar gramáticas JSON Schema con XGrammar para tu proyecto KernelAgent, integrándolas con el tokenizador del modelo que uses. Sabrás cómo depurar problemas cuando las gramáticas no generan lo esperado, inspeccionando el FSM resultante.
```


## Introducción

XGrammar es una librería que compila gramáticas en máquinas de estados finitos (FSMs) para guiar la generación de texto en modelos de lenguaje. Suena complejo, pero la arquitectura es elegante. En esta lectura entenderemos cómo funciona internamente, qué es cada componente, y cómo explorar el código.

## La Visión General: Del JSON al FSM

El pipeline de XGrammar transforma gramáticas en máquinas que pueden guiar la generación de tokens. Aquí está el flujo:

```
Gramática JSON (entrada)
        ↓
    Parser (analiza)
        ↓
      AST (árbol)
        ↓
   Compiler (procesa)
        ↓
      FSM (máquina de estados)
        ↓
LogitsProcessor (restricciones en generación)
```

:::{figure} diagrams/xgrammar_compilation_pipeline.png
:name: fig-xgrammar-pipeline
:alt: Pipeline completo de compilación de XGrammar de JSON Schema a LogitsProcessor
:align: center
:width: 90%

**Figura 2:** Pipeline de compilación de XGrammar: de JSON Schema a máquina de estados finitos a procesador de logits.
:::

Veamos cada paso.

## Paso 1: Gramática JSON (Entrada)

Una gramática describe qué texto es válido. XGrammar usa JSON Schema como base:

```json
{
  "type": "object",
  "properties": {
    "nombre": {"type": "string"},
    "edad": {"type": "integer"},
    "ciudad": {"type": "string"}
  },
  "required": ["nombre", "edad"]
}
```

Esta gramática especifica: "generarás un JSON objeto con propiedades nombre, edad, ciudad".

### Características soportadas

- **Tipos primitivos**: `string`, `integer`, `number`, `boolean`, `null`
- **Estructuras**: `object`, `array`
- **Restricciones**: `required`, `properties`, `items`, `enum`, `pattern`
- **Composición**: `oneOf`, `anyOf`, `allOf`

```python
import xgrammar as xgr

# Ejemplo: gramática para números
numero_schema = {
    "type": "number",
    "minimum": 0,
    "maximum": 100
}

# Ejemplo: gramática para arrays
lista_schema = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 5
}
```

## Paso 2: Parser (Análisis)

El parser lee la gramática y la analiza estructuralmente. Verifica que sea válida y extrae información.

```python
# Internamente, XGrammar hace algo así:
def parse_schema(schema):
    """Analizar JSON Schema"""

    if "type" not in schema:
        raise ValueError("type es obligatorio")

    schema_type = schema["type"]

    if schema_type == "string":
        return parse_string_schema(schema)
    elif schema_type == "object":
        return parse_object_schema(schema)
    elif schema_type == "array":
        return parse_array_schema(schema)
    # ...
    else:
        raise ValueError(f"type desconocido: {schema_type}")
```

### Validación durante el parsing

```{code-cell} ipython3
import json

def validar_schema(schema):
    """Validar que un schema sea JSON Schema válido"""

    # 1. Debe ser un diccionario
    if not isinstance(schema, dict):
        raise TypeError("Schema debe ser un dict")

    # 2. Si tiene type, debe ser string o array de strings
    if "type" in schema:
        tipo = schema["type"]
        if isinstance(tipo, str):
            if tipo not in ["string", "integer", "object", "array", "boolean", "null"]:
                raise ValueError(f"type inválido: {tipo}")

    # 3. Si es object, properties debe ser dict
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        if not isinstance(props, dict):
            raise TypeError("properties debe ser dict")

    return True
```

## Paso 3: AST (Abstract Syntax Tree)

El AST es una representación interna. Imagina que es un árbol donde cada nodo representa una parte de la gramática:

```
Object
├── Property "nombre"
│   └── String
├── Property "edad"
│   └── Integer
└── Property "ciudad"
    └── String
```

```{code-cell} ipython3
# Representación simplificada de un AST
class ASTNode:
    pass

class ObjectNode(ASTNode):
    def __init__(self, properties, required):
        self.properties = properties  # dict de nombre -> tipo
        self.required = required       # set de nombres requeridos

class StringNode(ASTNode):
    def __init__(self, pattern=None, enum=None):
        self.pattern = pattern
        self.enum = enum

class IntegerNode(ASTNode):
    def __init__(self, minimum=None, maximum=None):
        self.minimum = minimum
        self.maximum = maximum

# Construir el AST para nuestro schema de JSON
ast = ObjectNode(
    properties={
        "nombre": StringNode(),
        "edad": IntegerNode(minimum=0),
        "ciudad": StringNode()
    },
    required={"nombre", "edad"}
)
```

## Paso 4: Compiler (Compilación)

El compilador transforma el AST en una máquina de estados finitos. Esta es la parte más compleja internamente.

### ¿Qué es una máquina de estados finitos?

Una FSM es un modelo matemático con:
- **Estados**: posiciones en la generación (ej: "esperando '{'", "leyendo nombre", etc.)
- **Transiciones**: cambios entre estados (ej: ver token `{` → pasar a siguiente estado)
- **Aceptación**: estados finales válidos

Visualiza así:

```
START → { → nombre : → "..." → , → edad : → número → } → END
```

### Compilación simplificada

```python
class FSMCompiler:
    def __init__(self):
        self.state_counter = 0
        self.states = {}
        self.transitions = {}

    def new_state(self):
        state_id = self.state_counter
        self.state_counter += 1
        return state_id

    def compile_object(self, obj_node):
        """Compilar ObjectNode a FSM"""

        start = self.new_state()
        current = start

        # Estado después de {
        self.transitions[(current, '{')] = self.new_state()
        current = self.transitions[(current, '{')]

        # Para cada propiedad
        for prop_name, prop_type in obj_node.properties.items():
            # Transición a "nombre" : valor
            self.transitions[(current, prop_name)] = self.new_state()
            current = self.transitions[(current, prop_name)]

            self.transitions[(current, ':')] = self.new_state()
            current = self.transitions[(current, ':')]

            # Compilar el tipo
            current = self.compile_type(prop_type, current)

            # Coma (si no es el último)
            next_state = self.new_state()
            self.transitions[(current, ',')] = next_state
            self.transitions[(current, '}')] = end_state
            current = next_state

        # Estado final
        end_state = self.new_state()
        self.transitions[(current, '}')] = end_state

        return start, end_state

    def compile_type(self, node, current):
        # Compilar tipos recursivamente
        if isinstance(node, StringNode):
            # ... manejo especial para strings
            pass
        # ... otros tipos
        return current

compiler = FSMCompiler()
start_state, end_state = compiler.compile_object(ast)
```

## Paso 5: TokenizerInfo

Para que XGrammar funcione con modelos de lenguaje, necesita saber cómo el modelo tokeniza el texto. Diferentes modelos usan diferentes tokenizadores:

- **GPT-2**: BPE (Byte-Pair Encoding)
- **LLAMA**: Sentencepiece
- **T5**: Sentencepiece

```{code-cell} ipython3
:tags: [skip-execution]

import xgrammar as xgr
from transformers import AutoTokenizer

# Obtener el tokenizador del modelo
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Crear TokenizerInfo
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

# Ahora XGrammar sabe:
# - Cuántos tokens hay
# - Qué string representa cada token
# - Cómo funciona el tokenizador
```

XGrammar usa TokenizerInfo para crear máquinas que trabajan a nivel de **tokens**, no caracteres.

```python
# TokenizerInfo internamente almacena:
class TokenizerInfo:
    def __init__(self):
        self.vocab_size = 32000  # Número de tokens
        self.id_to_token = {}     # token_id -> string
        self.token_to_id = {}     # string -> token_id
        self.eos_token_id = 2     # ID del token de final

    def encode(self, text):
        """Convertir texto a token IDs"""
        return [self.token_to_id[t] for t in text.split()]

    def decode(self, token_ids):
        """Convertir token IDs a texto"""
        return "".join(self.id_to_token[id] for id in token_ids)
```

## Paso 6: LogitsProcessor (Restricción en Generación)

Una vez compilada la gramática a FSM, XGrammar crea un "procesador de logits". Durante la generación de texto, este procesador:

1. Rastrea en qué estado del FSM estamos
2. Permite solo tokens que conducen a estados válidos
3. Actualiza el estado cuando se genera un token

```{code-cell} ipython3
# Visualización de Token Masking
import numpy as np
import plotly.graph_objects as go

# Vocabulario simplificado
vocab = ['def', 'return', 'if', '(', ')', ':', '+', 'x', 'y', '1', '2', '<EOS>']
n_tokens = len(vocab)

# Máscara según estado de la gramática
estados = {
    'Inicio': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Solo 'def'
    'Después def': [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Identificadores
    'Después (': [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],  # Params o )
    'Body': [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # Return, if, vars, nums
}

fig = go.Figure()

for i, (estado, mask) in enumerate(estados.items()):
    colors = ['green' if m else 'red' for m in mask]
    fig.add_trace(go.Bar(name=estado, x=vocab, y=[0.8]*n_tokens,
                         marker_color=colors, opacity=0.7,
                         visible=(i==0)))

# Dropdown para cambiar estado
buttons = []
for i, estado in enumerate(estados.keys()):
    visibility = [j == i for j in range(len(estados))]
    buttons.append(dict(label=estado, method='update',
                        args=[{'visible': visibility}]))

fig.update_layout(
    title="Token Masking por Estado de la Gramática",
    updatemenus=[dict(active=0, buttons=buttons, x=0.1, y=1.15)],
    yaxis=dict(visible=False), height=350, showlegend=False
)
fig.show()
```

:::{figure} diagrams/constrained_decoding_flow.png
:name: fig-constrained-decoding
:alt: Flujo del proceso de generación restringida usando LogitsProcessor
:align: center
:width: 90%

**Figura 3:** Flujo de generación restringida - LogitsProcessor filtra tokens inválidos en cada paso.
:::

```python
# Pseudocódigo de LogitsProcessor
class GrammarLogitsProcessor:
    def __init__(self, fsm, tokenizer_info):
        self.fsm = fsm
        self.tokenizer = tokenizer_info
        self.current_state = fsm.start_state

    def __call__(self, input_ids, scores):
        """
        input_ids: tokens generados hasta ahora
        scores: logits (antes de softmax) para cada token del vocab

        Retorna: scores modificados para desalentar tokens inválidos
        """

        # Encontrar qué tokens son válidos desde current_state
        valid_tokens = self.fsm.get_valid_transitions(self.current_state)

        # Penalizar tokens inválidos
        for token_id in range(len(scores)):
            if token_id not in valid_tokens:
                scores[token_id] = -float('inf')  # Imposible de generar

        return scores

    def update_state(self, token_id):
        """Actualizar estado después de generar un token"""
        self.current_state = self.fsm.get_next_state(
            self.current_state,
            token_id
        )
```

## Explorando el Código de XGrammar

### Estructura del repositorio

```
xgrammar/
├── src/                    # Código C++
│   ├── compiler/           # Compilador (AST a FSM)
│   ├── grammar/            # Definiciones de gramática
│   └── runtime/            # Runtime (LogitsProcessor)
├── python/                 # Bindings Python
│   ├── xgrammar/
│   │   ├── __init__.py
│   │   ├── grammar.py      # API de gramáticas
│   │   ├── tokenizer.py    # TokenizerInfo
│   │   └── processor.py    # LogitsProcessor
├── tests/
├── CMakeLists.txt
└── README.md
```

### Inspeccionando objetos de XGrammar

```{code-cell} ipython3
import xgrammar as xgr
import json

# Crear una gramática compilada
schema = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "edad": {"type": "integer"}
    }
}

grammar = xgr.Grammar.from_schema(schema)

# Inspeccionarla
print(type(grammar))
# <class 'xgrammar.grammar.Grammar'>

print(dir(grammar))
# ['as_json', 'start_state', 'num_states', 'transitions', ...]

print(grammar.num_states)
# 12 (el compilador creó 12 estados)

# Obtener el estado inicial
print(grammar.start_state)
# 0
```

### Depurando gramáticas

Para entender qué hace una gramática, puedes visualizar sus transiciones:

```python
def print_fsm(grammar):
    """Imprimir la FSM en formato legible"""
    visited = set()

    def dfs(state, indent=0):
        if state in visited:
            return
        visited.add(state)

        print("  " * indent + f"State {state}:")

        # Obtener transiciones desde este estado
        for next_state, token_id in grammar.get_transitions(state):
            token_str = tokenizer.decode([token_id])
            print("  " * (indent + 1) + f"--{token_str}→ {next_state}")
            dfs(next_state, indent + 2)

    dfs(grammar.start_state)

grammar = xgr.Grammar.from_schema({"type": "string"})
print_fsm(grammar)
```

## Flujo Completo: De Schema a Generación

Aquí está todo junto:

```{code-cell} ipython3
:tags: [skip-execution]

import xgrammar as xgr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Preparar tokenizador
model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

# 2. Compilar gramática
schema = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "edad": {"type": "integer"}
    },
    "required": ["nombre"]
}

grammar = xgr.Grammar.from_schema(schema)
grammar.compile(tokenizer_info)

# 3. Cargar modelo
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. Crear processor
processor = xgr.LogitsProcessor(grammar)

# 5. Generar con restricciones
input_ids = tokenizer.encode("Genera JSON: ")

with torch.no_grad():
    for _ in range(100):  # Máximo 100 tokens
        outputs = model(torch.tensor([input_ids]))
        logits = outputs.logits[0, -1, :]

        # Aplicar restricciones
        logits = processor(input_ids, logits)

        # Muestreo
        token_id = torch.argmax(logits).item()
        input_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

# Decodificar resultado
resultado = tokenizer.decode(input_ids)
print(resultado)
# Output garantizado que es JSON válido
```

## Ejercicios

1. **Explora un schema simple**:
   - Compila un schema `{"type": "string", "enum": ["a", "b", "c"]}`
   - Imprime las transiciones de su FSM
   - ¿Cuántos estados tiene?

2. **Crea un visualizador**:
   - Escribe un script que lea una gramática
   - Exporte sus transiciones a formato DOT (para Graphviz)
   - Visualiza el FSM

3. **Comparar compilaciones**:
   - Compila dos esquemas similares
   - Compara el número de estados
   - ¿Cuál es más "grande"? ¿Por qué?

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- XGrammar pipeline: JSON Schema → Parser (valida) → AST (árbol) → Compiler (FSM) → TokenizerInfo (adaptación) → LogitsProcessor (restricción en generación)
- FSM (Finite State Machine) rastrea estados válidos y permite solo tokens que conducen a estados aceptables
- TokenizerInfo es crucial para adaptar gramáticas a diferentes tokenizadores (GPT-2 BPE vs Llama Sentencepiece)
- Gramáticas más complejas → FSMs más grandes → más memoria pero mayor validación
- Trade-off crítico: especificidad (muchas reglas) vs flexibilidad (pocas reglas)

**Siguiente paso:** En la próxima lectura aplicaremos estos conceptos para **definir la gramática JSON completa para kernels Triton en KernelAgent**, diseñando el schema que garantizará que el LLM genere código estructuralmente válido.
```

```{admonition} Verifica tu comprensión
:class: note
1. ¿Por qué XGrammar necesita TokenizerInfo? ¿Qué pasaría si usas el tokenizador incorrecto?
2. Explica cómo el LogitsProcessor restringe la generación: ¿qué hace con tokens inválidos?
3. Una gramática simple tiene 50 restricciones, otra compleja tiene 300. ¿Cuál prefieres para KernelAgent y por qué?
4. ¿Cómo depurarías una gramática que genera código inesperado? Describe los pasos usando el FSM.
```

## Conexión con el Proyecto

En el proyecto KernelAgent, entender el pipeline interno de XGrammar es crucial para:

1. **Depurar gramáticas**: Cuando tu gramática no genera lo que esperas, necesitas entender cómo se compila a FSM
2. **Optimizar performance**: Gramáticas más simples → FSMs más pequeñas → generación más rápida
3. **Diseñar gramáticas efectivas**: Saber cómo funciona el compilador te ayuda a diseñar mejor

El flujo completo en nuestro proyecto:
```
Schema JSON (kernel Triton)
    → Parser valida sintaxis
    → Compiler genera AST
    → AST se convierte a FSM
    → TokenizerInfo adapta FSM a tokens del modelo
    → LogitsProcessor filtra tokens durante generación
    → Output: código Triton válido
```

Cuando diseñes gramáticas para kernels Triton (lecturas 5-7), estarás creando estos schemas JSON. XGrammar los procesará exactamente como vimos en esta lectura.

**Casos de uso práctico**:
- Diseñar schema para kernel_add → XGrammar compila → LLM genera solo código sintácticamente válido
- Gramática compleja (muchas reglas) → FSM grande → generación más lenta
- Gramática simple → FSM pequeña → generación rápida pero menos expresiva

El trade-off es: ¿qué tan restrictivo quieres ser vs. qué tan flexible?

## Ejercicio Práctico: Comparar Schemas Simples

```{code-cell} ipython3
import json

# Vamos a comparar dos schemas: uno simple y uno complejo
# Simularemos el "tamaño de FSM" contando las restricciones

def count_constraints(schema, depth=0):
    """Cuenta el número de restricciones en un schema (proxy de complejidad FSM)"""
    count = 0

    if isinstance(schema, dict):
        # Cada propiedad es una restricción
        if "properties" in schema:
            count += len(schema["properties"])
            for prop_schema in schema["properties"].values():
                count += count_constraints(prop_schema, depth + 1)

        # Otras restricciones
        constraints = ["required", "enum", "pattern", "minimum", "maximum",
                      "minItems", "maxItems", "minLength", "maxLength"]
        for constraint in constraints:
            if constraint in schema:
                count += 1

        # Recursión para items, oneOf, anyOf
        if "items" in schema:
            count += count_constraints(schema["items"], depth + 1)
        if "oneOf" in schema:
            for sub_schema in schema["oneOf"]:
                count += count_constraints(sub_schema, depth + 1)

    return count

# Schema simple: solo nombre y tipo
simple_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "number"}
    },
    "required": ["name"]
}

# Schema complejo: con validaciones
complex_schema = {
    "type": "object",
    "properties": {
        "kernel_name": {
            "type": "string",
            "pattern": "^kernel_[a-z_]+$",
            "minLength": 8,
            "maxLength": 64
        },
        "block_size": {
            "type": "integer",
            "minimum": 32,
            "maximum": 1024,
            "enum": [32, 64, 128, 256, 512, 1024]
        },
        "dtype": {
            "enum": ["float16", "float32", "float64", "int32"]
        }
    },
    "required": ["kernel_name", "block_size", "dtype"],
    "additionalProperties": False
}

simple_constraints = count_constraints(simple_schema)
complex_constraints = count_constraints(complex_schema)

print(f"Schema simple: {simple_constraints} restricciones")
print(f"Schema complejo: {complex_constraints} restricciones")
print(f"Diferencia: {complex_constraints - simple_constraints}x más complejo")

print("\nImplicaciones para XGrammar:")
print(f"- Schema simple → FSM pequeña (~{simple_constraints * 2} estados estimados)")
print(f"- Schema complejo → FSM grande (~{complex_constraints * 3} estados estimados)")
print("- FSM más grande = más memoria, pero más restricciones = menos errores")

# Validar un ejemplo contra ambos
example = {
    "name": "my_kernel",
    "value": 3.14
}

print("\nEjemplo:", json.dumps(example))
print(f"¿Válido para schema simple? Sí (tiene 'name' requerido)")
print(f"¿Válido para schema complejo? No (falta kernel_name, block_size, dtype)")

# Ejercicio: ¿Cuál schema usarías para generar kernels Triton?
print("\nReflexión:")
print("Para kernels Triton, necesitas balance:")
print("- Muy simple → LLM genera código inválido")
print("- Muy complejo → LLM tarda mucho en generar")
print("Objetivo: restricciones mínimas necesarias para validez sintáctica")
```

**Análisis de resultados**:
- El schema simple tiene menos restricciones → FSM más pequeña → generación más rápida
- El schema complejo tiene muchas restricciones → FSM más grande → generación más lenta PERO más correcta
- En la práctica, para kernels Triton necesitas el schema complejo porque la sintaxis es estricta

**Tarea adicional**: Modifica `complex_schema` para agregar una restricción más (ej: un campo `description` opcional). Observa cómo cambia el conteo de restricciones.

---

## Referencias

- XGrammar. [Efficient, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar). GitHub.
- Willard, B. & Louf, R. (2023). [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702). arXiv.
