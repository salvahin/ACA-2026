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

# Lectura 6: Sampling y Constrained Decoding

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers plotly xgrammar pydantic accelerate
    print('Dependencias instaladas!')
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/06_sampling_constrained_decoding.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Identificar el problema de salidas inválidas (JSON mal formado, código con errores sintácticos) en LLMs sin restricciones
- Aplicar constrained decoding mediante máscaras de logits para garantizar salidas válidas
- Utilizar gramáticas formales (CFG) con frameworks como XGrammar para restricciones complejas
- Evaluar el trade-off entre calidad de salida y garantía de validez en constrained decoding
- Implementar bitmask simple para restricciones de vocabulario limitado
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[Decoding Strategies in Large Language Models](https://www.youtube.com/watch?v=d_ixjvAHA1E)** - Explicación gráfica de top-p, top-k, y temperature en la selección de tokens.
```


## Introducción

En la lectura anterior, vimos cómo generar texto token por token usando diferentes estrategias (greedy, top-K, top-P). Pero ¿qué pasa cuando necesitas que el modelo genere en un formato específico? Por ejemplo:

- JSON válido (llaves y valores correctos)
- Código compilable (sintaxis correcta)
- Respuestas estructuradas (formato específico)

Sin restricciones, el modelo a menudo genera JSON mal formado o código con errores sintácticos. **Constrained decoding** es la solución: modifica dinámicamente qué tokens puede generar para garantizar que el resultado respete restricciones.

---

## Parte 1: Repaso de Decoding Libre

Antes de hablar de restricciones, recordemos cómo funciona decoding sin restricciones:

```
Paso 1: Transformer produce logits para 50,257 tokens
        logits = [2.1, 0.5, 3.2, 1.1, ..., -5.2]

Paso 2: Aplica temperatura y convierte a probabilidades
        P = softmax(logits / T)

Paso 3: Aplica Top-P (nucleus) sampling
        - Ordena por probabilidad descendente
        - Suma hasta alcanzar P_acumulada > 0.9
        - Renormaliza y muestrea

Paso 4: Genera siguiente token
        Token elegido → Agregua a secuencia

Paso 5: Repite desde Paso 1 hasta [END] o límite de longitud
```

El modelo es **libre** de elegir cualquier token (con probabilidades aprendidas).

:::{figure} diagrams/sampling_strategies.png
:name: fig-sampling-strategies
:alt: Comparación de estrategias de sampling en generación de texto
:align: center
:width: 90%

**Figura 1:** Estrategias de sampling: Greedy (siempre el más probable), Top-K (muestrea de los K más probables), Top-P/Nucleus (muestrea de los tokens que acumulan masa de probabilidad p), Temperatura (controla la aleatoriedad distribuyendo o concentrando la distribución).
:::

---

## Parte 2: El Problema - Salidas Inválidas

:::{figure} images/AI_06_01_Quality_Constraint_Tradeoff.jpeg
:name: fig-quality-tradeoff
:alt: Trade-off entre calidad y restricciones
:align: center
:width: 90%

**Figura 2:** Trade-off Calidad vs Restricciones - más restricciones garantizan formato válido pero pueden reducir la calidad del contenido.
:::

### Ejemplo 1: JSON

```
Pregunta: "Extrae nombre y edad. Devuelve JSON."

Respuesta del modelo SIN restricciones:
{
  "nombre": "Juan
  "edad": 30,
  "ciudad": "Madrid"
  "ocupacion: "Ingeniero"
}

Problemas:
- "Juan sin cierre de comillas
- Falta comilla en "ocupacion
- Clave ocupacion no solicitada
```

El código que intenta parsear esto falla:

```{code-cell} ipython3
import json

# Ejemplo de JSON malformado
respuesta = """{
  "nombre": "Juan
  "edad": 30,
  "ciudad": "Madrid"
  "ocupacion: "Ingeniero"
}"""

try:
    data = json.loads(respuesta)
except json.JSONDecodeError as e:
    print(f"Error: {e}")
```

### Ejemplo 2: Código

```python
Pregunta: "Escribe una función que retorna el factorial"

Respuesta SIN restricciones:
def factorial(n):
    if n == 0
        return 1
    else
        return n * factorial(n - 1)

Problemas:
- Falta ":" después de "if n == 0"
- Indentación inconsistente
```

Este código no compila:

```
SyntaxError: expected ':' (line 2)
```

### El Coste de Regeneración

```{admonition} 🤔 Reflexiona
:class: hint
¿Por qué es costoso regenerar múltiples veces? Piensa en términos de tokens consumidos (que se pagan) y latencia (tiempo de espera del usuario). ¿Cuál es más crítico en tu aplicación?
```

Una solución ingenua: si el resultado es inválido, pide al modelo que intente de nuevo.

```
Intento 1: Resultado inválido → Descarta
Intento 2: Resultado inválido → Descarta
Intento 3: Resultado inválido → Descarta
Intento 4: Válido ✓

Costo: 4× el tiempo y tokens de salida
```

Esto es muy costoso a escala.

---

## Parte 3: Constrained Decoding - Enfoque Conceptual


:::{figure} images/AI_06_04_Constrained_Decoding_Pipeline_Steps.jpeg
:name: fig-constrained-decoding
:alt: Pipeline de Constrained Decoding con gramáticas
:align: center
:width: 90%

**Figura 3:** Pipeline de Constrained Decoding - filtrando tokens inválidos según gramática especificada.
:::

### La Idea

En lugar de permitir cualquier token:

```
Paso 1: Calcula qué tokens PODRÍAN ser válidos en esta posición
        - Si acabas de escribir "{", el próximo debe ser una comilla (para key)
        - Si estás dentro de un string, no puedes cerrar con "}"
        - Si escribiste "if", debes escribir ":" en algún momento

Paso 2: Crea una "máscara" de tokens válidos
        valid_mask = [1, 0, 1, 0, 0, 1, ...]
                     (1 = válido, 0 = inválido)

Paso 3: Anula logits de tokens inválidos
        logits_masked[i] = logits[i] si valid_mask[i] == 1
                          = -∞         si valid_mask[i] == 0

Paso 4: Aplica softmax (tokens con -∞ → probabilidad 0)
        P = softmax(logits_masked)

Paso 5: Muestrea normalmente
        (nunca selecciona tokens con P=0)
```

**Resultado:** El modelo elige el mejor token que sigue siendo válido.

---

## Parte 4: LogitsProcessor - Implementación

### LogitsProcessor Genérica

Frameworks como Hugging Face implementan esto con `LogitsProcessor`:

```python
class ConstrainedLogitsProcessor:
    def __call__(self, input_ids, scores):
        """
        input_ids: tokens generados hasta ahora
        scores: logits sin procesar para el siguiente token

        Retorna: scores modificados
        """
        # Identifica tokens válidos basado en input_ids
        valid_tokens = self.get_valid_tokens(input_ids)

        # Anulan logits de tokens inválidos
        scores[~valid_tokens] = -float("inf")

        return scores
```

### Ejemplo: JSON Válido

```python
import json
from transformers import LogitsProcessor

class JSONLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_valid_tokens(self, input_ids):
        # Convierte tokens a texto
        text = self.tokenizer.decode(input_ids)

        # Determina qué caracteres son válidos en JSON
        # Basado en el estado actual (dentro de string, dentro de objeto, etc.)

        valid_chars = self.get_valid_json_chars(text)
        valid_token_ids = [
            i for i, token in enumerate(self.tokenizer.vocab.items())
            if token in valid_chars
        ]

        return valid_token_ids
```

### Estados en JSON

```
Estado 1: Esperando { inicial
  válidos: { solo

Estado 2: Dentro del objeto, esperando key
  válidos: ", nombre de variable, }

Estado 3: Dentro de string (clave)
  válidos: caracteres, ", (pero no }))

Estado 4: Después de ", esperando :
  válidos: :

Estado 5: Después de :, esperando valor
  válidos: ", número, true, false, null, {, [

etc.
```

---

## Parte 5: XGrammar - Constrained Decoding Avanzado

:::{figure} diagrams/constrained_decoding.png
:name: fig-constrained-decoding-xgrammar
:alt: Flujo de constrained decoding con XGrammar mostrando la máscara de tokens válidos
:align: center
:width: 90%

**Figura 4:** XGrammar en acción: la gramática se compila en una máscara de bitmask que, en cada paso de decodificación, bloquea los tokens que producirían una salida inválida según la gramática especificada.
:::

XGrammar es un framework especializado en constrained decoding. Permite especificar restricciones mediante **gramáticas formales**.

### Gramáticas sin Contexto (CFG)

```
JSON Grammar (simplificada):

json ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" (string ":" value ("," string ":" value)*)? "}"
array ::= "[" (value ("," value)*)? "]"
value ::= object | array | string | number | "true" | "false" | "null"
string ::= "\"" ([\x20-\x21\x23-\x5B\x5D-\x7E] | "\\\"")* "\""
number ::= [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
```

Esto define **exactamente** qué es un JSON válido.

### Cómo XGrammar lo Usa

```
1. Parser de la gramática → Autómata finito determinista (DFA)
2. En cada paso de generación:
   - Estado actual en DFA: "dentro de objeto, esperando key"
   - Tokens válidos desde ese estado: ["\"", "}"]
   - Anulan logits de otros tokens
   - Muestrea normalmente
3. Transición de estado: si genera "\", nuevo estado = "dentro de string"
4. Repite hasta alcanzar estado de aceptación (JSON válido)
```

### Ventajas de XGrammar

```
✓ Garantiza salida válida (respeta la gramática 100%)
✓ Flexible: soporta cualquier CFG
✓ Integración con llama.cpp y otros servidores

Ejemplos:
- JSON: grammar = load_json_grammar()
- SQL: grammar = load_sql_grammar()
- Python: grammar = load_python_grammar()
```

---

## Parte 6: Bitmask - Constrained Decoding Sencillo

Para restricciones simples, una máscara de bits es suficiente:

### Caso 1: Tokens Específicos

```python
# Solo permite tokens que son dígitos
valid_tokens = [
    tokenizer.encode("0")[0],
    tokenizer.encode("1")[0],
    tokenizer.encode("2")[0],
    # ...,
    tokenizer.encode("9")[0],
]

bitmask = [False] * tokenizer.vocab_size
for token_id in valid_tokens:
    bitmask[token_id] = True

# Durante decoding:
logits[~bitmask] = -float("inf")
```

### Caso 2: Palabras de Vocabulario Pequeño

```python
# Genera solo respuestas de un conjunto limitado
allowed_responses = ["Sí", "No", "Quizás"]
allowed_token_ids = [tokenizer.encode(resp)[0] for resp in allowed_responses]

bitmask = [False] * vocab_size
for token_id in allowed_token_ids:
    bitmask[token_id] = True
```

**Ventaja:** Muy rápido, ningún overhead computacional
**Desventaja:** Solo funciona para restricciones simples

:::{figure} diagrams/temperature_sampling.png
:name: fig-temperature-sampling
:alt: Efecto de la temperatura en la distribución de probabilidades del siguiente token
:align: center
:width: 90%

**Figura 5:** Temperatura en sampling: T < 1 concentra la distribución (más determinístico), T = 1 mantiene la distribución original, T > 1 aplana la distribución (más aleatorio y creativo). Para código: T ≈ 0 garantiza salidas consistentes.
:::

```{admonition} 🎮 Simulación Interactiva: Top-K vs Top-P Sampling
:class: tip

Compara visualmente las diferentes estrategias de sampling.
```

```{code-cell} ipython3
# Comparación Top-K vs Top-P
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)
probs = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01])
tokens = [f'tok_{i}' for i in range(len(probs))]
```

A continuación establecemos los límites correspondientes para Top-K (los *k* tokens más probables aislándolos) y Top-P (recortando el subconjunto de tokens que suman al menos una masa probabilística acumulada de *p*). 

```{code-cell} ipython3
fig = make_subplots(rows=1, cols=3, subplot_titles=('Original', 'Top-K (K=3)', 'Top-P (P=0.9)'))

# Original
fig.add_trace(go.Bar(x=tokens, y=probs, marker_color='steelblue'), row=1, col=1)

# Top-K
top_k = 3
top_k_probs = np.where(np.argsort(-probs) < top_k, probs, 0)
top_k_probs = top_k_probs / top_k_probs.sum()
colors_k = ['green' if p > 0 else 'lightgray' for p in top_k_probs]
fig.add_trace(go.Bar(x=tokens, y=top_k_probs, marker_color=colors_k), row=1, col=2)

# Top-P
cumsum = np.cumsum(np.sort(probs)[::-1])
cutoff_idx = np.searchsorted(cumsum, 0.9) + 1
threshold = np.sort(probs)[::-1][min(cutoff_idx, len(probs)-1)]
top_p_probs = np.where(probs >= threshold, probs, 0)
top_p_probs = top_p_probs / top_p_probs.sum()
colors_p = ['orange' if p > 0 else 'lightgray' for p in top_p_probs]
fig.add_trace(go.Bar(x=tokens, y=top_p_probs, marker_color=colors_p), row=1, col=3)

fig.update_layout(height=350, showlegend=False, title_text="Estrategias de Sampling")
fig.show()
```

### Beam Search: Exploración Sistemática

**Beam Search** es una estrategia determinística que mantiene múltiples "hipótesis" (secuencias candidatas) en paralelo, explorando el espacio de manera más exhaustiva que greedy decoding.

```
Greedy: Siempre elige el token más probable
        → Puede quedar atrapado en secuencias subóptimas

Beam Search (beam_width=3):
        → Mantiene las 3 mejores secuencias en cada paso
        → Explora múltiples caminos simultáneamente
        → Al final, elige la secuencia con mayor probabilidad total
```

**Algoritmo paso a paso:**

```
Entrada: "El gato"
Beam width: 2

Paso 1 (generar palabra 3):
  Candidatos: "saltó" (P=0.4), "corrió" (P=0.3), "durmió" (P=0.2)
  Beams activos:
    Beam 1: "El gato saltó" (score=0.4)
    Beam 2: "El gato corrió" (score=0.3)

Paso 2 (generar palabra 4):
  Para Beam 1 ("El gato saltó"):
    "alto" (P=0.5), "lejos" (P=0.3)
  Para Beam 2 ("El gato corrió"):
    "rápido" (P=0.6), "lejos" (P=0.2)

  Scores totales:
    "El gato saltó alto": 0.4 × 0.5 = 0.20
    "El gato saltó lejos": 0.4 × 0.3 = 0.12
    "El gato corrió rápido": 0.3 × 0.6 = 0.18
    "El gato corrió lejos": 0.3 × 0.2 = 0.06

  Beams activos (top 2):
    Beam 1: "El gato saltó alto" (score=0.20)
    Beam 2: "El gato corrió rápido" (score=0.18)

Resultado final: "El gato saltó alto"
```

```{code-cell} ipython3
import numpy as np

def beam_search_demo(initial_probs, beam_width=2, max_steps=3):
    """
    Demostración simplificada de beam search.
    """
    vocab = ['saltó', 'corrió', 'durmió', 'alto', 'lejos', 'rápido', '.']

    # Inicializar beams
    beams = [{'tokens': [], 'score': 1.0}]

    print(f"Beam Search (width={beam_width})")
    print("=" * 50)

    for step in range(max_steps):
        all_candidates = []

        for beam in beams:
            # Simular probabilidades del siguiente token
            np.random.seed(len(beam['tokens']) + 42)
            probs = np.random.dirichlet(np.ones(len(vocab)))

            for i, (token, prob) in enumerate(zip(vocab, probs)):
                new_beam = {
                    'tokens': beam['tokens'] + [token],
                    'score': beam['score'] * prob
                }
                all_candidates.append(new_beam)

        # Ordenar por score y quedarse con los mejores
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        beams = all_candidates[:beam_width]

        print(f"\nPaso {step + 1}:")
        for i, beam in enumerate(beams):
            print(f"  Beam {i+1}: {' '.join(beam['tokens'])} (score={beam['score']:.4f})")

    return beams[0]

# Ejecutar demo
best = beam_search_demo(None, beam_width=3, max_steps=3)
print(f"\n✓ Mejor secuencia: {' '.join(best['tokens'])}")
```

**Comparación de estrategias:**

| Estrategia | Determinista | Explora múltiples | Uso típico |
|------------|--------------|-------------------|------------|
| Greedy | ✓ | ✗ | Respuestas factuales |
| Top-K | ✗ | ✗ | Texto creativo |
| Top-P | ✗ | ✗ | Balance creatividad/coherencia |
| Beam Search | ✓ | ✓ | Traducción, resumen |

```{admonition} 🔑 Cuándo usar Beam Search
:class: tip

**Usa Beam Search cuando:**
- Necesitas la "mejor" respuesta objetiva (traducción, transcripción)
- La creatividad no es deseable
- Puedes pagar el costo computacional (beam_width × más lento)

**Evita Beam Search cuando:**
- Necesitas diversidad (chatbots, escritura creativa)
- Recursos limitados (cada beam es una secuencia completa en memoria)
- El problema no tiene una "respuesta correcta" clara
```

---

## Parte 7: Trade-offs - Calidad vs Restricciones

:::{figure} images/AI_06_02_JSON_State_Machine_Valid_Transitions.jpeg
:name: fig-json-state-machine
:alt: Máquina de estados para JSON válido
:align: center
:width: 90%

**Figura 6:** Máquina de Estados JSON - el constrained decoding sigue transiciones válidas para garantizar sintaxis correcta.
:::

### El Dilema

```
Sin restricciones:
- Mejor calidad (el modelo elige libremente)
- Puede generar salida inválida
- Requiere reintento o parseo manual

Con restricciones estrictas:
- Salida garantizada válida
- Posiblemente menor calidad (forzado a tokens sub-óptimos)
- Sin overhead de reintento
```

### Ejemplo: JSON con Número

```
Prompt: "¿Cuántos años tienes? (responde como {\"edad\": <número>})"

Sin restricciones:
P("7") = 0.7     ← Mejor probabilidad
P("siete") = 0.2
P("07") = 0.05

Con restricción JSON:
P("7") = 0.7     ← Se permite, se elige
P("siete") = 0    ← Anulado (no es número en JSON)
P("07") = 0.05   ← Se permite, segunda opción

Resultado: Mismo token elegido (7)
Sin penalización de calidad en este caso.
```

### Otro Ejemplo: Creatividad

```
Prompt: "Escribe un haiku"

Sin restricciones:
El modelo puede usar cualquier estructura
Potencialmente más creativo

Con restricción (estructura haiku):
línea 1: 5 sílabas
línea 2: 7 sílabas
línea 3: 5 sílabas

El modelo DEBE seguir esto
Menos flexibilidad, pero garantizado haiku
```

:::{figure} images/AI_06_03_Quality_vs_Constraint_Strictness.jpeg
:name: fig-quality-strictness
:alt: Calidad vs nivel de restricción
:align: center
:width: 90%

**Figura 7:** Calidad vs Restricción - restricciones más estrictas garantizan formato pero pueden afectar creatividad.
:::

### Recomendaciones

```
Usa restricciones CUANDO:
- Necesitas salida estructurada (JSON, XML, SQL)
- Dominio tiene reglas claras (código, formato)
- Costo de reintento es alto (tokens = dinero)

Evita restricciones CUANDO:
- Necesitas máxima creatividad
- Restricciones excluyen buenas respuestas
- Costo computacional de constrained decoding > reintento
```

---

## Parte 8: Ejemplo Práctico con XGrammar en Colab

```{admonition} 💻 Recursos necesarios
:class: tip
Este ejemplo requiere **GPU** para ejecutarse:
- GPU recomendada: T4 (16GB) o superior
- Modelo: Qwen2.5-1.5B-Instruct (~3GB en FP16)
- Tiempo de carga inicial: ~30 segundos
- Tiempo por inferencia: ~2-5 segundos
```

### Escenario de Extracción de Datos

Queremos extraer información de una reseña de película mediante un LLM, pero garantizando algorítmicamente que el modelo devuelva un JSON perfectamente estructurado para nuestra base de datos:

```json
{
    "sentimiento": "positivo",
    "calificacion": 9,
    "recomendado": true
}
```

### Paso 1: Verificar GPU y Definir Esquema

```{code-cell} ipython3
import torch

print("Verificación de recursos:")
print(f"  GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️ No hay GPU disponible. Este ejemplo requiere GPU.")
```

#### ¿Qué es Pydantic?

**Pydantic** es una librería de Python para validación de datos usando anotaciones de tipo. En lugar de escribir código manual para verificar que los datos tienen el formato correcto, defines una clase con tipos y Pydantic se encarga de:

1. **Validar** que los datos cumplen el esquema
2. **Convertir** tipos automáticamente (ej: `"123"` → `123`)
3. **Generar JSON Schema** para usar con otras herramientas

```python
# Sin Pydantic (manual y propenso a errores)
def validar_resena(data):
    if not isinstance(data.get("sentimiento"), str):
        raise ValueError("sentimiento debe ser string")
    if not isinstance(data.get("calificacion"), int):
        raise ValueError("calificacion debe ser int")
    # ... más validaciones manuales

# Con Pydantic (declarativo y robusto)
class ResenaInfo(pydantic.BaseModel):
    sentimiento: str
    calificacion: int
    recomendado: bool
```

La magia para constrained decoding: Pydantic genera automáticamente un **JSON Schema** que XGrammar compila a un autómata finito, garantizando que el LLM solo genere JSON válido según ese esquema.

```{code-cell} ipython3
import pydantic
import json

class ResenaInfo(pydantic.BaseModel):
    """Esquema para extracción de información de reseñas."""
    sentimiento: str   # "positivo", "negativo", "neutro"
    calificacion: int  # 1-10
    recomendado: bool  # true/false

# Mostrar el JSON Schema generado automáticamente
print("JSON Schema generado por Pydantic:")
print(json.dumps(ResenaInfo.model_json_schema(), indent=2))
```

```{admonition} 💡 Ventajas de Pydantic + XGrammar
:class: tip

1. **Declarativo**: Defines QUÉ quieres, no CÓMO validarlo
2. **Type hints**: El mismo código sirve para IDE autocompletado y validación
3. **Composable**: Puedes anidar modelos (`List[ResenaInfo]`, `Optional[str]`)
4. **Estándar**: JSON Schema es un estándar abierto, no propietario
```

### Paso 2: Cargar Modelo y Compilar Gramática

```{code-cell} ipython3
import xgrammar as xgr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Cargando modelo: {model_id}")
print("(Esto puede tomar ~30 segundos la primera vez...)")

config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("✓ Modelo cargado")

# Compilar gramática XGrammar desde el esquema Pydantic
print("\nCompilando gramática XGrammar...")
# Convertir tokenizer de HuggingFace a formato XGrammar (usando vocab_size del config)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = grammar_compiler.compile_json_schema(ResenaInfo.model_json_schema())
print("✓ Gramática compilada (DFA creado)")
```

### Paso 3: Función de Extracción con Constrained Decoding

```{code-cell} ipython3
def extraer_info_resena(review_text: str, use_grammar: bool = True) -> str:
    """
    Extrae información estructurada de una reseña.

    Args:
        review_text: Texto de la reseña
        use_grammar: Si True, usa constrained decoding con XGrammar

    Returns:
        JSON string con la información extraída
    """
    messages = [
        {"role": "system", "content": "Extrae la información de la reseña. Devuelve SOLO el JSON con campos: sentimiento, calificacion, recomendado."},
        {"role": "user", "content": review_text}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # IMPORTANTE: Crear un nuevo LogitsProcessor para cada llamada
    # El LogitsProcessor mantiene estado interno que se modifica durante la generación
    if use_grammar:
        logits_processor = [xgr.contrib.hf.LogitsProcessor(compiled_grammar)]
    else:
        logits_processor = None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extraer solo los tokens generados
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
```

### Paso 4: Probar con Ejemplos Reales

```{code-cell} ipython3
ejemplos = [
    "¡Película increíble! Me encantó cada minuto. 9/10. Totalmente recomendada.",
    "Muy aburrida, no la terminaría nunca. 3/10. No la recomiendo para nada.",
    "Estuvo bien, nada especial. 6/10. Quizás si no tienes nada mejor que ver."
]

print("Extracción de Información con Constrained Decoding (XGrammar)")
print("=" * 70)

for i, review in enumerate(ejemplos, 1):
    print(f"\nEjemplo {i}:")
    print(f"  Reseña: \"{review}\"")

    resultado = extraer_info_resena(review, use_grammar=True)
    print(f"  JSON extraído: {resultado}")

    # Validar con Pydantic
    try:
        info = ResenaInfo.model_validate_json(resultado)
        print(f"  ✓ Validación Pydantic exitosa")
        print(f"    - Sentimiento: {info.sentimiento}")
        print(f"    - Calificación: {info.calificacion}")
        print(f"    - Recomendado: {info.recomendado}")
    except Exception as e:
        print(f"  ✗ Error de validación: {e}")
```

### Paso 5: Comparación CON vs SIN Constrained Decoding

```{code-cell} ipython3
print("Comparación: CON vs SIN Constrained Decoding")
print("=" * 70)

review_test = "Película regular, algunos momentos buenos pero otros aburridos. 5/10."

print(f"\nReseña de prueba: \"{review_test}\"\n")

# SIN constrained decoding (puede generar JSON inválido)
print("SIN constrained decoding (3 intentos):")
print("-" * 40)
for i in range(3):
    resultado = extraer_info_resena(review_test, use_grammar=False)
    try:
        ResenaInfo.model_validate_json(resultado)
        status = "✓ válido"
    except:
        status = "✗ inválido"
    # Mostrar primeros 80 caracteres
    display = resultado[:80].replace('\n', ' ')
    print(f"  {i+1}. {display}... [{status}]")

# CON constrained decoding (siempre genera JSON válido)
print("\nCON constrained decoding (3 intentos):")
print("-" * 40)
for i in range(3):
    resultado = extraer_info_resena(review_test, use_grammar=True)
    try:
        ResenaInfo.model_validate_json(resultado)
        status = "✓ válido"
    except:
        status = "✗ inválido"
    display = resultado[:80].replace('\n', ' ')
    print(f"  {i+1}. {display}... [{status}]")

print("\n💡 Observa cómo XGrammar garantiza JSON válido en TODOS los intentos.")
```

```{admonition} 🔑 Puntos clave del ejemplo
:class: note

1. **Pydantic** define el esquema de datos de forma declarativa
2. **XGrammar** compila el esquema a un autómata finito determinista (DFA)
3. **XGrammarLogitsProcessor** intercepta los logits en cada paso de generación
4. Tokens que producirían JSON inválido reciben probabilidad 0
5. El modelo mantiene creatividad dentro de las restricciones gramaticales
```

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **Trade-off calidad:** ¿En qué casos crees que constrained decoding degrada significativamente la calidad de la salida?

2. **Gramáticas:** ¿Cómo escribirías una gramática para código Python válido? ¿Qué lo haría más complejo que JSON?

3. **Híbrido:** ¿Podrías combinar constrained decoding (garantiza validez) con reintento (mejora calidad)? ¿Cómo?

### Ejercicios Prácticos:

1. **Diseña una máscara de bits:**
   ```
   Crea una máscara que solo permite respuestas de verdadero/falso
   (asume tokenizer.encode("verdadero")[0] = 500, "falso"[0] = 501)
   ```

2. **Escribe una mini-gramática:**
   ```
   Escribe una gramática para números positivos:
   número ::= [1-9][0-9]* | "0"

   ¿Permite "007"? ¿Permite "0"? ¿Por qué?
   ```

3. **Análisis de overhead:**
   ```
   Estimas que:
   - Constrained decoding agrega 10% de overhead
   - Reintento promedio 1.5 intentos (50% de fallo)

   Para 10,000 solicitudes:
   - Costo sin restricción: 10,000 * 1.5 = 15,000 generaciones
   - Costo con restricción: 10,000 * 1.1 = 11,000 generaciones

   ¿Vale la pena agregar restricciones?
   ```

4. **Reflexión escrita (350 palabras):** "Imagina que estás construyendo un sistema que genera SQL automáticamente. SQL mal formado puede dañar la base de datos. ¿Usarías constrained decoding, reintento manual, o una combinación? ¿Por qué?"

---

## Puntos Clave

- **Problema:** LLMs sin restricciones generan salida inválida (JSON mal formado, código con errores)
- **Solución:** Constrained decoding - anulan logits de tokens inválidos
- **Bitmask:** Simple, rápido; para restricciones simples (vocabulario limitado)
- **LogitsProcessor:** Flexible; puede verificar estados complejos
- **XGrammar:** Gramáticas formales; garantiza validez de salida
- **Trade-off:** Restricciones garantizan validez pero pueden reducir calidad ligeramente
- **Recomendación:** Usa cuando costo de reintento > overhead de constrained decoding

---

## Errores Comunes

```{admonition} ⚠️ Errores frecuentes
:class: warning

1. **Máscara incorrecta**: Anular logits con 0 en lugar de -inf. Usa `logits[~mask] = -float("inf")`, no `logits[~mask] = 0`.
2. **Olvidar renormalizar**: Después de anular logits, softmax renormaliza automáticamente. No lo hagas manualmente.
3. **Gramáticas incompletas**: Una gramática con reglas faltantes puede bloquear la generación (no tokens válidos disponibles).
4. **Overhead subestimado**: Verificar tokens válidos en cada paso puede ser lento. Cachea cuando sea posible.
```

## Ejercicio Práctico: Implementar Bitmask Simple

```{code-cell} ipython3
import torch
import torch.nn.functional as F

# 1. Definir el entorno del tokenizador
vocab_size = 50257  # Tamaño típico de GPT-2

# 2. Simular logits emitidos por el modelo (scores sin normalizar)
torch.manual_seed(42)
logits = torch.randn(vocab_size)

# TAREA: Solo permitir respuestas "Sí" (token 43521) o "No" (token 2949)
allowed_tokens = [43521, 2949]
```

Para aplicar nuestro *constrained decoding* duro (restringir exhaustivamente vocabulario completo), crearemos un tensor booleano y forzaremos implacablemente los logits no permitidos hacia `-infinito`.

```{code-cell} ipython3
# 3. Crear matriz de la máscara (bitmask) inicializandola en falso
bitmask = torch.zeros(vocab_size, dtype=torch.bool)

for token_id in allowed_tokens:
    bitmask[token_id] = True

# Aplicar máscara
logits_masked = logits.clone()
logits_masked[~bitmask] = -float("inf")

# 6. Transformar en Espacio de Probabilidad usando Softmax
probs_original = F.softmax(logits, dim=-1)
probs_masked = F.softmax(logits_masked, dim=-1)
```

Veamos los resultados probabilísticos ahora que hemos forzado agresivamente la distribución y re-aplicado el regulador *Softmax*.

```{code-cell} ipython3
# 7. Imprimir Auditoría Visual Comparativa
print("SIN restricciones:")
print(f"  Top 5 tokens más probables originalmente: {torch.topk(probs_original, 5).indices.tolist()}")
print(f"  Confirmar suma global de probabilidades: {probs_original.sum():.4f}")

print("\nCON máscara (solo 'Sí' o 'No'):")
print(f"  Tokens permitidos: {allowed_tokens}")
print(f"  Probabilidad re-ajustada del 'Sí' (43521): {probs_masked[43521]:.4f}")
print(f"  Probabilidad re-ajustada del 'No' (2949): {probs_masked[2949]:.4f}")
print(f"  Suma validada de P() del vocabulario restrictivo: {probs_masked.sum():.4f}")

# 8. Muestreo seguro 
sampled_token = torch.multinomial(probs_masked, 1).item()
print(f"\nToken finalmente seleccionado durante inferencia: {sampled_token}")
print(f"  ¿Es un token esperado?: {sampled_token in allowed_tokens}")
```

```{admonition} ✅ Verifica tu comprensión
:class: note
1. ¿Por qué usar -float("inf") en lugar de 0 para anular logits de tokens inválidos?
2. Explica el trade-off fundamental de constrained decoding: validez vs calidad.
3. ¿Cuándo usarías bitmask simple vs XGrammar con CFG completa?
4. Diseña una estrategia híbrida: constrained decoding + reintento. ¿Cuándo sería útil?
```

## Resumen

```{admonition} Resumen
:class: important
**Conceptos clave:**
- LLMs sin restricciones generan salidas inválidas frecuentemente (JSON mal formado, código con errores)
- Constrained decoding anula logits de tokens inválidos usando máscara, forzando salida válida
- Bitmask simple: para vocabulario limitado (ej: "Sí"/"No"); rápido, sin overhead
- LogitsProcessor: flexible, verifica estados complejos; XGrammar usa gramáticas formales (CFG)
- Trade-off: restricciones garantizan validez pero pueden reducir calidad al forzar tokens sub-óptimos
- Usa cuando costo de reintento > overhead de verificación (especialmente en producción a escala)

**Para la siguiente lectura:** Prepárate para fine-tuning y evaluación. Veremos cómo adaptar modelos pre-entrenados a tareas específicas con LoRA/QLoRA y cómo evaluar rigurosamente evitando benchmark contamination.
```

---

## Referencias

- Holtzman, A. et al. (2020). [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR 2020.
- XGrammar. [Efficient, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar). GitHub.
- Willard, B. & Louf, R. (2023). [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702). arXiv.
- Scholak, T. et al. (2021). [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding](https://arxiv.org/abs/2109.05093). EMNLP.
