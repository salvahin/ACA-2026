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

# Lectura 5: BERT, GPTs y Tokenización


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/05_bert_gpts_tokenizacion.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Distinguir entre BERT (comprensión bidireccional) y GPT (generación autoregresiva) según arquitectura y uso
- Explicar cómo funciona Masked Language Modeling (MLM) en el pre-entrenamiento de BERT
- Comprender la filosofía de escalamiento de GPT y las capacidades emergentes
- Aplicar tokenización BPE/WordPiece/SentencePiece para convertir texto a tokens
- Analizar el impacto del tokenizador en costo computacional y contexto disponible
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[Word Embedding and Tokenization (StatQuest)](https://www.youtube.com/watch?v=viZrOnJclY0)** - Animaciones sencillas y didácticas sobre cómo dividimos texto para los modelos de lenguaje.
```

```{admonition} 🔧 Herramienta Interactiva
:class: seealso

**[BertViz](https://github.com/jessevig/bertviz)** - Herramienta para visualizar cómo los *Attention Heads* de BERT y GPT conectan las palabras e infieren contexto. Requiere ejecutarse en un Jupyter/Colab notebook.
```


## Introducción

Hasta ahora entiendes la arquitectura Transformer. Pero, ¿cómo se materializó en modelos que cambiaron el mundo? BERT revolucionó NLP con comprensión bidireccional. GPT demostró que escalar funciona. Y todo comienza con **tokenización**: convertir texto en números que el modelo entiende.

Esta lectura conecta arquitectura con implementaciones reales y el proceso fundamental de tokenización.

---

## Parte 1: BERT - Comprensión Bidireccional

### El Problema Pre-BERT

Antes de 2018, los modelos de lenguaje eran **unidireccionales**:

```
Oración: "El banco está cerca del río"

Modelo unidireccional (izquierda→derecha):
  "banco" se predice solo con contexto de "El"
  No sabe si es banco financiero o banco de río

Modelo bidireccional (BERT):
  "banco" se predice con contexto completo
  Ve "río" → entiende que es banco de agua
```

### Arquitectura BERT

```
BERT = Transformer Encoder (bidireccional)

Entrada:  [CLS] El banco está cerca del río [SEP]
           ↓     ↓    ↓     ↓     ↓    ↓   ↓
        Embeddings + Positional + Segment
           ↓     ↓    ↓     ↓     ↓    ↓   ↓
        ┌─────────────────────────────────────┐
        │      12-24 Transformer Layers       │
        │      (Multi-Head Self-Attention)    │
        └─────────────────────────────────────┘
           ↓     ↓    ↓     ↓     ↓    ↓   ↓
        Representaciones contextualizadas
```

### Pre-entrenamiento de BERT

**Tarea 1: Masked Language Modeling (MLM)**

```
Original:  "El gato [MASK] sobre la mesa"
Objetivo:  Predecir "salta"

→ Fuerza comprensión bidireccional
→ 15% de tokens se enmascaran
```

**Tarea 2: Next Sentence Prediction (NSP)**

```
Oración A: "El perro ladró"
Oración B: "Tenía hambre"  → IsNext (relacionadas)
Oración B: "París es bonito" → NotNext (no relacionadas)
```

### Variantes de BERT

```
BERT-base:    110M parámetros, 12 layers, 768 hidden
BERT-large:   340M parámetros, 24 layers, 1024 hidden

RoBERTa:      Sin NSP, más datos, entrenamiento más largo
ALBERT:       Parámetros compartidos, más eficiente
DistilBERT:   66% más pequeño, 97% performance
```

---

## Parte 2: GPT - Generación Autoregresiva a Escala

### Filosofía GPT

Mientras BERT comprende, GPT genera. La apuesta: **escalar funciona**.

```
GPT-1 (2018):   117M parámetros
GPT-2 (2019):   1.5B parámetros  → "Demasiado peligroso para publicar"
GPT-3 (2020):   175B parámetros → Few-shot learning emerge
GPT-4 (2023):   ~1.7T parámetros (estimado)
```

### Arquitectura GPT

```
GPT = Transformer Decoder (unidireccional)

Diferencia clave con BERT:
  - Masked self-attention (solo ve tokens anteriores)
  - Entrenado para predecir siguiente token
  - Genera texto autorregressivamente

Entrada:  "El gato"
Proceso:
  P(siguiente | "El gato") → "salta"
  P(siguiente | "El gato salta") → "sobre"
  ...
```

### Emergent Abilities

A cierta escala, emergen capacidades no entrenadas explícitamente:

```
Escala pequeña (<10B):
  - Completa texto
  - Traduce (si entrenado)

Escala media (10-100B):
  - Few-shot learning
  - Razonamiento básico

Escala grande (>100B):
  - Chain-of-thought
  - Razonamiento complejo
  - Código
```

### GPT para Código

```{code-cell} ipython3
# CodeX/Codex: GPT fine-tuned en código
# Base de GitHub Copilot

# Prompt: "# Función que ordena una lista"
# Output:
def sort_list(lst):
    return sorted(lst)

# Ejemplo de uso
print(sort_list([3, 1, 4, 1, 5, 9, 2, 6]))
```

---

## Parte 3: Tokenización

### ¿Por Qué Tokenizar?

Los modelos no entienden texto. Necesitan números.

```
Texto:     "Hello world"
Tokens:    [15496, 995]    # IDs numéricos
Embeddings: [[0.1, -0.3, ...], [0.5, 0.2, ...]]  # Vectores
```

### Estrategias de Tokenización

**1. Word-level (palabras)**
```
"I love programming" → ["I", "love", "programming"]

Problemas:
  - Vocabulario enorme (100k+ palabras)
  - OOV (out-of-vocabulary): "unforgettable" → [UNK]
  - No maneja typos: "progrmaing" → [UNK]
```

**2. Character-level (caracteres)**
```
"Hello" → ["H", "e", "l", "l", "o"]

Problemas:
  - Secuencias muy largas
  - Pierde significado semántico
  - Ineficiente para transformers
```

**3. Subword (BPE, WordPiece, SentencePiece)**
```
"unforgettable" → ["un", "forget", "table"]
"programming"   → ["program", "ming"]

Ventajas:
  - Vocabulario manejable (~30-50k)
  - Maneja palabras nuevas
  - Balance entre word y character
```

### BPE (Byte Pair Encoding)

Algoritmo paso a paso:

```python
# Corpus inicial
vocabulary = {'l o w </w>': 5, 'l o w e r </w>': 2,
              'n e w e s t </w>': 6, 'w i d e s t </w>': 3}

# Paso 1: Contar pares de caracteres
pairs = {('e', 's'): 9, ('s', 't'): 9, ('t', '</w>'): 9, ...}

# Paso 2: Merge el par más frecuente
'es' se convierte en un token
vocabulary = {'l o w </w>': 5, 'l o w e r </w>': 2,
              'n e w es t </w>': 6, 'w i d es t </w>': 3}

# Repetir hasta alcanzar tamaño de vocabulario deseado
```

:::{figure} diagrams/bpe_tokenization.png
:name: fig-bpe
:alt: Proceso de Byte Pair Encoding para tokenización
:align: center
:width: 90%

**Figura 3:** Byte Pair Encoding (BPE) - cómo se fusionan caracteres para crear tokens subpalabra.
:::

### Implementación con tiktoken (OpenAI)

```{code-cell} ipython3
import tiktoken

# Encoder para GPT-4
enc = tiktoken.encoding_for_model("gpt-4")

text = "def hello_world():"
tokens = enc.encode(text)
print(tokens)  # [755, 23748, 11645, 33529, 25]
print(len(tokens))  # 5 tokens

# Decodificar
decoded = enc.decode(tokens)
print(decoded)  # "def hello_world():"

# Ver tokens individuales
for token in tokens:
    print(f"{token} → '{enc.decode([token])}'")
# 755 → 'def'
# 23748 → ' hello'
# 11645 → '_world'
# 33529 → '()'
# 25 → ':'
```

### Tokenización de Código

El código tiene características especiales:

```{code-cell} ipython3
# Código Triton
code = """
@triton.jit
def add_kernel(x_ptr, y_ptr, n):
    pid = tl.program_id(0)
"""

# Con tiktoken (cl100k_base)
tokens = enc.encode(code)
print(f"Tokens: {len(tokens)}")  # ~25-30 tokens

# Observaciones:
# - "@triton" puede ser 2 tokens: "@" + "triton"
# - "program_id" puede ser: "program" + "_id"
# - Indentación consume tokens
```

### SentencePiece (Google)

Alternativa a BPE, usado en T5, LLaMA:

```{code-cell} ipython3
:tags: [skip-execution]

import sentencepiece as spm

# Entrenar modelo
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe'  # o 'unigram'
)

# Usar modelo
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

tokens = sp.encode_as_pieces("Hello world")
# ['▁Hello', '▁world']  # ▁ marca inicio de palabra
```

### Comparación de Tokenizadores

| Aspecto | BPE (GPT) | WordPiece (BERT) | SentencePiece |
|---------|-----------|------------------|---------------|
| Merge | Más frecuente | Maximiza likelihood | Unigram/BPE |
| Prefijo | Ninguno | ## para continuación | ▁ para inicio |
| Usado en | GPT, CodeX | BERT, DistilBERT | T5, LLaMA |

```{admonition} 🎮 Simulación Interactiva: Visualización de Tokenización BPE
:class: tip

Observa cómo se descompone una palabra en tokens BPE.
```

```{code-cell} ipython3
# Simulación visual de tokenización BPE
import plotly.graph_objects as go

# Ejemplo de tokenización
texto = "unhappiness"
tokens_bpe = ["un", "##happy", "##ness"]
colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig = go.Figure()

# Mostrar texto original
fig.add_annotation(x=0.5, y=0.8, text=f"Texto: '{texto}'",
                   showarrow=False, font=dict(size=20))

# Mostrar tokens
for i, (token, color) in enumerate(zip(tokens_bpe, colores)):
    fig.add_shape(type="rect", x0=i*0.3+0.05, y0=0.3, x1=i*0.3+0.3, y1=0.5,
                  fillcolor=color, opacity=0.7)
    fig.add_annotation(x=i*0.3+0.175, y=0.4, text=token,
                       showarrow=False, font=dict(size=14, color='white'))

fig.update_layout(
    title="Visualización de Tokenización BPE",
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[0, 1]),
    height=300
)
fig.show()
```

---

## Parte 4: De Texto a Modelo

### Pipeline Completo

```
Texto crudo
    ↓
Tokenización (BPE/SentencePiece)
    ↓
Token IDs [15496, 995, ...]
    ↓
Embedding lookup (vocab_size × hidden_dim)
    ↓
Positional encoding
    ↓
Transformer layers
    ↓
Logits (vocab_size)
    ↓
Softmax → Probabilidades
    ↓
Sampling/Decoding
    ↓
Token ID → Detokenizar → Texto
```

### Vocabulario y Embeddings

```{code-cell} ipython3
# En PyTorch
import torch
import torch.nn as nn

vocab_size = 50257  # GPT-2
hidden_dim = 768

embedding = nn.Embedding(vocab_size, hidden_dim)

# Token ID → Vector
token_id = torch.tensor([15496])  # "Hello"
vector = embedding(token_id)  # Shape: [1, 768]
print(f"Vector shape: {vector.shape}")
```

### Impacto del Tokenizador

```
Mismo texto, diferentes tokenizadores:

Texto: "tl.load(x_ptr + offsets)"

GPT-4 (cl100k):  ~8 tokens
LLaMA:           ~10 tokens
BERT:            ~12 tokens

Impacto:
  - Más tokens = más costo ($)
  - Más tokens = más lento
  - Más tokens = menos contexto disponible
```

---

## Parte 5: Logits, Softmax y Cross-Entropy

### De Logits a Probabilidades

```{code-cell} ipython3
import torch
import torch.nn.functional as F

# Logits: output crudo del modelo (sin normalizar)
logits = torch.tensor([2.0, 1.0, 0.1])

# Softmax: convierte a probabilidades
probs = F.softmax(logits, dim=-1)
print(probs)
# tensor([0.659, 0.242, 0.099])
# Suma = 1.0
print(f"Suma: {probs.sum()}")

# Interpretación:
# Token 0: 65.9% probabilidad
# Token 1: 24.2% probabilidad
# Token 2:  9.9% probabilidad
```

### Temperature

```{code-cell} ipython3
def softmax_with_temp(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)

logits = torch.tensor([2.0, 1.0, 0.1])

# T=1.0 (normal)
print(f"T=1.0: {softmax_with_temp(logits, 1.0)}")  # [0.659, 0.242, 0.099]

# T=0.5 (más confiado)
print(f"T=0.5: {softmax_with_temp(logits, 0.5)}")  # [0.836, 0.142, 0.022]

# T=2.0 (más uniforme)
print(f"T=2.0: {softmax_with_temp(logits, 2.0)}")  # [0.506, 0.307, 0.187]
```

### Cross-Entropy Loss

```{code-cell} ipython3
# Función de pérdida para entrenamiento
loss_fn = nn.CrossEntropyLoss()

# Logits del modelo (batch=1, vocab_size=3)
logits = torch.tensor([[2.0, 1.0, 0.1]])

# Target: el token correcto es el índice 0
target = torch.tensor([0])

loss = loss_fn(logits, target)
print(f"Loss: {loss.item():.3f}")
# loss ≈ 0.417

# Interpretación:
# - Loss bajo = modelo confiado en respuesta correcta
# - Loss alto = modelo equivocado o inseguro
```

---

## Ejercicios Prácticos

### Ejercicio 1: Explorar Tokenización

```{code-cell} ipython3
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

# Tokeniza estos snippets de Triton
snippets = [
    "@triton.jit",
    "tl.load(x_ptr + offsets, mask=mask)",
    "pid = tl.program_id(axis=0)",
]

for s in snippets:
    tokens = enc.encode(s)
    print(f"'{s}' → {len(tokens)} tokens")
    for t in tokens:
        print(f"  {t}: '{enc.decode([t])}'")
```

### Ejercicio 2: Comparar Tokenizadores

Compara cómo diferentes tokenizadores manejan código Triton. ¿Cuál es más eficiente?

### Ejercicio 3: Visualizar Attention

Usa BertViz para visualizar qué tokens "atienden" a cuáles en un snippet de código.

---

## Preguntas de Reflexión

1. ¿Por qué BERT usa [MASK] en lugar de predecir el siguiente token como GPT?

2. Si entrenaras un tokenizador específico para código Triton, ¿qué tokens "especiales" incluirías?

3. ¿Cómo afecta la elección de tokenizador al costo de usar un LLM para generar kernels?

---

## Recursos

- **Paper BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers"
- **Paper GPT-3**: Brown et al. "Language Models are Few-Shot Learners"
- **BPE Original**: Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units"
- **tiktoken**: github.com/openai/tiktoken
- **Hugging Face Tokenizers**: huggingface.co/docs/tokenizers

---

## Errores Comunes

```{admonition} ⚠️ Errores frecuentes
:class: warning

1. **Confundir logits con probabilidades**: Los logits son scores sin normalizar. Aplica softmax para obtener probabilidades.
2. **Temperature mal entendida**: T < 1 = más determinista, T > 1 = más aleatorio. T=0 es greedy.
3. **Tokenización con espacios**: Muchos tokenizadores agregan espacio inicial (" hello" vs "hello").
```

## Ejercicio Práctico: Visualizar Temperatura

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Logits de ejemplo (scores del modelo)
logits = np.array([2.0, 1.0, 0.5, 0.1])
tokens = ['the', 'a', 'an', 'one']

def softmax_with_temp(logits, temperature):
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()

# Comparar diferentes temperaturas
temperatures = [0.5, 1.0, 2.0]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, T in zip(axes, temperatures):
    probs = softmax_with_temp(logits, T)
    ax.bar(tokens, probs)
    ax.set_title(f'Temperature = {T}')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilidad')

plt.tight_layout()
plt.savefig('temperature_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# Interpretación
print("T=0.5: Distribución concentrada (más determinista)")
print("T=1.0: Distribución original")
print("T=2.0: Distribución más uniforme (más aleatorio)")
```

---

```{admonition} ✅ Verifica tu comprensión
:class: note
1. ¿Por qué BERT usa [MASK] en lugar de predecir el siguiente token como GPT?
2. Explica con tus palabras qué son las "capacidades emergentes" en modelos grandes como GPT-3.
3. Si entrenaras un tokenizador específico para código Triton, ¿qué tokens "especiales" incluirías?
4. ¿Cómo afecta la elección de tokenizador al costo de usar un LLM? Piensa en términos de tokens generados.
```

## Resumen

```{admonition} Resumen
:class: important
**Conceptos clave:**
- BERT (Transformer Encoder): comprensión bidireccional mediante MLM y NSP; excelente para clasificación y QA
- GPT (Transformer Decoder): generación autoregresiva; escalar modelos produce capacidades emergentes (few-shot, reasoning)
- Tokenización convierte texto → IDs numéricos; estrategias: word-level (vocabulario enorme), character-level (secuencias largas), subword (balance óptimo)
- BPE/WordPiece/SentencePiece descomponen palabras en sub-palabras (~30-50k vocab), manejando palabras nuevas eficientemente
- Temperatura controla distribución de probabilidades; T<1 más determinístico, T>1 más aleatorio

**Para la siguiente lectura:** Prepárate para explorar constrained decoding. Veremos cómo forzar que los LLMs generen en formatos específicos (JSON, código válido) usando gramáticas.
```

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Devlin, J. et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805). NAACL 2019.
- Radford, A. et al. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI.
- Brown, T. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS 2020.
- Sennrich, R., Haddow, B., & Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL 2016.
- Kudo, T. & Richardson, J. (2018). [SentencePiece: A Simple and Language Independent Subword Tokenizer](https://arxiv.org/abs/1808.06226). EMNLP 2018.
- Wei, J. et al. (2022). [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682). TMLR.
