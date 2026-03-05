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

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



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

```{admonition} 📚 Prerequisito
:class: note
Antes de esta lección debes haber leido:
- **Lectura 2:** Fundamentos de Deep Learning — Redes neuronales y activaciones
- **Lectura 3:** Generación Autoregresiva — Tokenización BPE, sampling y temperatura
- **Lectura 4:** Arquitectura Transformer — Atención, multi-head, encoder vs decoder
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

### WordPiece en Python Puro (BERT-style)

Al igual que vimos BPE desde cero en la Lectura 3, aquí implementamos **WordPiece** para entender la diferencia clave: BPE elige el par más **frecuente**; WordPiece elige el par que **maximiza la probabilidad** del corpus (score = freq(AB) / (freq(A) × freq(B))).

```{code-cell} ipython3
from collections import Counter
import math

def wordpiece_score(pair_count, token_counts):
    """
    Score de WordPiece: prefiere pares cuya combinación es
    mucho más probable que sus partes por separado.
    
    score(A,B) = count(AB) / (count(A) * count(B))
    """
    a, b = pair_count
    return token_counts[a + b] / (token_counts[a] * token_counts[b] + 1e-10)

def get_pairs(vocab):
    """Obtiene todos los pares de tokens adyacentes con sus frecuencias."""
    pairs = Counter()
    for word, freq in vocab.items():
        tokens = word.split()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += freq
    return pairs

def merge_pair(vocab, pair):
    """Fusiona el par ganador en todo el vocabulario."""
    merged = ' '.join(pair)
    replacement = ''.join(pair)
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = word.replace(merged, replacement)
        new_vocab[new_word] = freq
    return new_vocab
```

Con estas funciones auxiliares establecidas, podemos diseñar el bucle principal de entrenamiento. Nota cómo el criterio de selección es la fórmula matemática de máxima verosimilitud (likelihood) en lugar de la pura frecuencia usada en BPE.

```{code-cell} ipython3
def wordpiece_train(corpus, num_merges=8):
    """
    Entrenamiento simplificado de WordPiece.
    Diferencia vs BPE: el criterio de selección del par.
    """
    # Tokenización inicial: caracteres + marcador ## para continuaciones
    vocab = {}
    for word in corpus.split():
        chars = list(word)
        # BERT-style: el primer carácter es normal, los demás van con ##
        chars_wp = [chars[0]] + ['##' + c for c in chars[1:]]
        key = ' '.join(chars_wp)
        vocab[key] = vocab.get(key, 0) + 1
    
    print("=== WordPiece Simplificado ===")
    print(f"Vocabulario inicial: {set(' '.join(vocab.keys()).split())}")
    print()
    
    merge_history = []
    
    for step in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        
        # Contar tokens individuales
        token_counts = Counter()
        for word, freq in vocab.items():
            for tok in word.split():
                token_counts[tok] += freq
        
        # Seleccionar par con mayor score probabilistico de WordPiece
        best_pair = max(pairs, key=lambda p: wordpiece_score(p, token_counts))
        best_score = wordpiece_score(best_pair, token_counts)
        merged_token = ''.join(best_pair)
        
        print(f"Paso {step+1}: merge {best_pair} → '{merged_token}'  (score={best_score:.4f})")
        merge_history.append((best_pair, merged_token))
        vocab = merge_pair(vocab, best_pair)
    
    final_vocab = set()
    for word in vocab:
        final_vocab.update(word.split())
    
    print(f"\nVocabulario final ({len(final_vocab)} tokens):")
    print(sorted(final_vocab))
    return vocab, merge_history
```

A continuación simularemos el modelo procesando un subconjunto de palabras que podrían encontrarse iterando código Triton.

```{code-cell} ipython3
# Corpus de prueba con términos de código Triton
corpus_triton = "pointer offset mask load store pointer offset load mask pointer"
vocab_final, history = wordpiece_train(corpus_triton, num_merges=6)
```

```{admonition} 🔑 BPE vs WordPiece: la diferencia clave
:class: important
| | BPE | WordPiece |
|---|---|---|
| **Elige el par** | más frecuente | que maximiza `P(corpus)` |
| **Resultado** | vocabulario compacto | mejor manejo de palabras raras |
| **Modelos** | GPT-2, CodeX | BERT, DistilBERT |

En la práctica, ambos producen vocabularios similares para textos grandes. La diferencia importa en vocabularios pequeños y palabras infrecuentes.
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

```{admonition} 📚 Revisado en Lectura 3
:class: seealso
Estos conceptos se desarrollaron en detalle **con numpy desde cero** en la
**Lectura 3: Generación Autoregresiva** (Partes 2, 3 y 5). Aquí los repasamos
utilizando la API nativa de **PyTorch** para conectar con la práctica real.
```

```{code-cell} ipython3
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Logits → Softmax → Probabilidades ---
logits = torch.tensor([2.0, 1.0, 0.1])
probs  = F.softmax(logits, dim=-1)
print(f"Logits:         {logits.numpy()}")
print(f"Probabilidades: {probs.detach().numpy().round(3)}")
# tensor([0.659, 0.242, 0.099]) — suman a 1.0
print(f"Suma: {probs.sum().item():.4f}")
```

La temperatura (Temperature scaling) nos permite modificar la confianza matemática resultante antes de hacer el muestreo de tokens:

```{code-cell} ipython3
# --- Temperature scaling ---
def softmax_temp(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)

print("\nEfecto de la temperatura (ver Lectura 3 para detalles):")
for T in [0.5, 1.0, 2.0]:
    print(f"  T={T}: {softmax_temp(logits, T).detach().numpy().round(3)}")
```

Finalmente, evaluamos el error del modelo comparando pre-dicciones con la respuesta esperada mediante Cross-Entropy Loss:

```{code-cell} ipython3
# --- Cross-Entropy Loss (usada durante entrenamiento) ---
loss_fn = nn.CrossEntropyLoss()
logits_batch = torch.tensor([[2.0, 1.0, 0.1]])  # batch=1, vocab_size=3
target = torch.tensor([0])                        # token correcto = índice 0
loss = loss_fn(logits_batch, target)
print(f"\nCross-Entropy Loss (target=token 0): {loss.item():.3f}")
# Más bajo = modelo más confiado en la respuesta correcta
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
```

Como podemos constatar analíticamente, reducir `T` por debajo de 1 endurece el modelo.

```{code-cell} ipython3
# Interpretación de la visualización
print("T=0.5: Distribución concentrada (Top-K amplificado, menos riesgoso/más determinista)")
print("T=1.0: Distribución probabilística estadísticamente fiel al modelo original")
print("T=2.0: Distribución plana atenuada (muy riesgoso/más alucinaciones y dispersión)")
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

---

## Lecturas Recomendadas

- **D2L: BERT** - [Capítulo 15.8](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html). Profundiza en la arquitectura y el pre-entrenamiento de BERT.
- **D2L: Subword Embedding** - [Capítulo 15.6](https://d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html). Explicación detallada de BPE, WordPiece y SentencePiece.
