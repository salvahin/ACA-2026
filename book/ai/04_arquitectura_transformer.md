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

# Lectura 4: Arquitectura Transformer

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Explicar el mecanismo de scaled dot-product attention usando Query, Key y Value
- Comprender por qué multi-head attention captura diferentes tipos de relaciones
- Aplicar codificación posicional para incorporar información de orden en secuencias
- Distinguir entre arquitecturas encoder (BERT) y decoder (GPT) según el tipo de tarea
- Identificar los componentes de un bloque Transformer (atención, FFN, layer norm, conexiones residuales)
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[Attention in transformers, visually explained (3Blue1Brown)](https://www.youtube.com/watch?v=eMlx5fFNoYc)** - Desglose geométrico interactivo del mecanismo de atención.
```

```{admonition} 🔧 Herramienta Interactiva
:class: seealso

**[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** - Visualización interactiva de GPT-2: ingresa texto y observa cómo Self-Attention procesa la información token por token.
```

## Contexto
Dominarás la arquitectura Transformer, fundamento de todos los LLMs modernos. Comprenderás el mecanismo de atención, multi-head attention, codificación posicional y la diferencia entre encoders (BERT) y decoders (GPT).

## Introducción

Los Transformers revolucionaron el procesamiento de lenguaje natural. Publicados en 2017 ("Attention is All You Need"), esta arquitectura es el fundamento de todos los LLMs modernos: GPT, BERT, Claude, Llama, etc.

En esta lectura entenderemos cómo funcionan Transformers desde sus componentes básicos hasta la arquitectura completa. La clave es el **mecanismo de atención**, que permitió a los modelos ver relaciones entre palabras distantes sin procesarlas secuencialmente.

---

## Parte 1: Motivación - El Problema de las Secuencias Largas

### El Desafío de RNNs

Recuerda de la lectura anterior:

```
Entrada: "El gato saltó sobre la cerca porque oyó un ruido en el jardín"

RNN:
t=1: "El" → Estado_1
t=2: "gato" + Estado_1 → Estado_2
t=3: "saltó" + Estado_2 → Estado_3
...
t=9: "jardín" + Estado_8 → Estado_9
```

**Problema:** Para que "jardín" influya en cómo interpretamos "El" (palabra 1), el gradiente debe viajar hacia atrás a través de 8 pasos. Cada paso multiplica por valores pequeños (desvanecimiento de gradientes).

Además, procesamiento secuencial es lento. Si la frase tiene 1000 palabras, debes hacer 1000 pasos seriales.

### La Solución: Atención

¿Qué pasa si, en lugar de procesar palabra por palabra, permitimos que cada palabra "atienda" a todas las demás palabras simultáneamente?

```
Pregunta: "¿Qué palabras son relevantes para entender 'saltó'?"

Respuesta (después de atención):
- "gato" es muy relevante (30% de atención)
- "cerca" es relevante (20% de atención)
- "el" es poco relevante (5% de atención)
- "jardín" tiene algo de relevancia (15% de atención)
- etc.

Luego, representación de "saltó" = 30% de "gato" + 20% de "cerca" + ... (promedio ponderado)
```

Esto permite que: 1) Cada palabra vea todo el contexto inmediatamente, 2) Las conexiones lejanas son directas (sin múltiples pasos), 3) Podemos procesar en paralelo.

---

## Parte 2: El Mecanismo de Atención Escalada por Puntuación

### Conceptos Previos: Embeddings

Las palabras se representan como vectores numéricos llamados **embeddings**.

```
"gato" → [0.2, -0.5, 0.8, 0.1, ...]  (vector de dimensión 768 para GPT-style)
"perro" → [0.3, -0.4, 0.7, 0.15, ...] (similar a "gato")
"matriz" → [0.9, 0.2, -0.1, -0.5, ...] (diferente)
```

Palabras similares tienen embeddings similares. Esto se aprende durante el entrenamiento.

### La Operación de Atención: Queries, Keys, Values

Imagina que quieres entender la palabra "saltó". Usas tres operaciones:

```
1. QUERY (Pregunta): "¿Qué estoy intentando entender?"
   Query para "saltó" = proyecto_a_query(embedding["saltó"])

2. KEY (Clave): "¿De qué información estoy informando?"
   Keys para todas las palabras: [proyecto_a_key(emb[i]) para cada palabra i]

3. VALUE (Valor): "¿Qué información proporciono?"
   Values para todas las palabras: [proyecto_a_value(emb[i]) para cada palabra i]
```

**Matemáticamente:**
```
Q = X @ W_Q    (Query: una proyección linear de la entrada)
K = X @ W_K    (Key: otra proyección linear)
V = X @ W_V    (Value: una tercera proyección linear)
```

donde X es la matriz de embeddings de entrada y W_Q, W_K, W_V son matrices de peso aprendidas.

### Puntuación y Softmax

```
Paso 1: Calcula similitud entre query y cada key

Similitud("saltó", "gato") = Query["saltó"] · Key["gato"]  (producto punto)
Similitud("saltó", "perro") = Query["saltó"] · Key["perro"]
Similitud("saltó", "matriz") = Query["saltó"] · Key["matriz"]
...

Paso 2: Escala por √d_k (stabilidad numérica)

scores = Q @ K^T / √d_k

donde d_k es la dimensión de las keys (ej: 64)
```

¿Por qué √d_k? Sin escala, los productos punto tienen varianza grande, causando softmax con picos muy agudos. La escala los mantiene estables.

```
Paso 3: Convierte scores a probabilidades con softmax

attention_weights = softmax(scores)

Ejemplo (tokens: "el", "gato", "saltó", "cerca"):
Para "saltó":
  scores = [0.1, 2.5, 0.0, 1.8]
  attention_weights = softmax(...) = [0.02, 0.72, 0.01, 0.25]

Interpretación: al procesar "saltó", atiende 72% a "gato", 25% a "cerca", etc.
```

### Agregación de Valores

```
output = attention_weights @ V

= 0.02 * Value["el"] + 0.72 * Value["gato"] + 0.01 * Value["saltó"] + 0.25 * Value["cerca"]
```

La representación actualizada de "saltó" es un promedio ponderado de los values basado en las similitudes calculadas.

### Fórmula Completa

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

Esta es la fórmula más importante de los Transformers. La verás en cada configuración.

```{admonition} 🤔 Reflexiona
:class: hint
¿Por qué escalamos por √d_k? Piensa qué pasaría con softmax si los productos punto tuvieran valores muy grandes (ej: 100 o -100). ¿Cómo afectaría esto a los gradientes durante entrenamiento?
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """Softmax numerically stable"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(Q, K, V):
    """
    Scaled Dot-Product Attention
    Q: Query matrix (seq_len, d_k)
    K: Key matrix (seq_len, d_k)
    V: Value matrix (seq_len, d_v)
    """
    d_k = Q.shape[-1]

    # Calcular scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Aplicar softmax
    attention_weights = softmax(scores)

    # Agregar values
    output = attention_weights @ V

    return output, attention_weights

# Ejemplo simple: 4 palabras
np.random.seed(42)
seq_len = 4
d_k = 8  # Dimensión de las keys/queries
d_v = 8  # Dimensión de los values

# Crear embeddings simulados para 4 palabras: "el", "gato", "saltó", "cerca"
words = ['el', 'gato', 'saltó', 'cerca']

# Generar Q, K, V (normalmente vienen de proyecciones lineales)
Q = np.random.randn(seq_len, d_k) * 0.5
K = np.random.randn(seq_len, d_k) * 0.5
V = np.random.randn(seq_len, d_v) * 0.5

# Aplicar atención
output, attn_weights = attention(Q, K, V)

print("Mecanismo de Atención - Ejemplo Práctico")
print("=" * 60)
print(f"\nEntrada: {words}")
print(f"Dimensión d_k (Query/Key): {d_k}")
print(f"Dimensión d_v (Value): {d_v}")

print("\nMatriz de Pesos de Atención:")
print(f"{'':>10}", end='')
for word in words:
    print(f"{word:>10}", end='')
print()
print("-" * 50)

for i, word_i in enumerate(words):
    print(f"{word_i:>10}", end='')
    for j in range(seq_len):
        print(f"{attn_weights[i, j]:>10.3f}", end='')
    print(f"  (suma={attn_weights[i].sum():.3f})")

print("\nInterpretación:")
print("- Cada fila muestra cómo una palabra 'atiende' a todas las palabras")
print("- Los valores son probabilidades (suman a 1.0)")
print("- Valores altos = mayor relevancia")

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap de pesos de atención
im = axes[0].imshow(attn_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
axes[0].set_xticks(range(seq_len))
axes[0].set_yticks(range(seq_len))
axes[0].set_xticklabels(words)
axes[0].set_yticklabels(words)
axes[0].set_xlabel('Key (atendiendo a...)', fontsize=11)
axes[0].set_ylabel('Query (palabra actual)', fontsize=11)
axes[0].set_title('Matriz de Pesos de Atención', fontsize=12, weight='bold')

# Añadir valores en las celdas
for i in range(seq_len):
    for j in range(seq_len):
        text = axes[0].text(j, i, f'{attn_weights[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=axes[0], label='Peso de Atención')

# Visualización de atención para una palabra específica
word_idx = 2  # "saltó"
axes[1].bar(words, attn_weights[word_idx], color='steelblue', alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Peso de Atención', fontsize=11)
axes[1].set_title(f'Atención de "{words[word_idx]}" a todas las palabras', fontsize=12, weight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 1.0)

# Añadir valores en las barras
for i, (word, weight) in enumerate(zip(words, attn_weights[word_idx])):
    axes[1].text(i, weight + 0.02, f'{weight:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Mostrar cálculo paso a paso para un ejemplo
print("\n\nCálculo Paso a Paso (Query de 'saltó' atendiendo a todas):")
print("=" * 60)
q_salto = Q[2]  # Query para "saltó"
print(f"1. Query 'saltó': {q_salto[:3].round(2)}... (dim={d_k})")

print(f"\n2. Scores (similitud con cada Key):")
scores = q_salto @ K.T / np.sqrt(d_k)
for i, word in enumerate(words):
    print(f"   Score con '{word}': {scores[i]:.4f}")

print(f"\n3. Softmax (convertir a probabilidades):")
for i, word in enumerate(words):
    print(f"   Atención a '{word}': {attn_weights[2, i]:.4f} ({attn_weights[2, i]*100:.1f}%)")

print(f"\n4. Output = weighted sum de Values")
print(f"   Output 'saltó': {output[2][:3].round(2)}... (dim={d_v})")
```

---

## Parte 3: Atención Multi-Cabeza

Una cabeza de atención captura un tipo de relación. ¿Qué pasa si queremos múltiples tipos?

```
Cabeza 1: Atención a sujetos gramaticales
  "el perro saltó" → Cabeza 1 atiende "el" a "perro"

Cabeza 2: Atención a objetos
  "el perro saltó sobre la cerca" → Cabeza 2 atiende "saltó" a "cerca"

Cabeza 3: Atención a dependencias léxicas
  "saltó" → Cabeza 3 atiende a tiempos verbales relacionados ("salta", "saltará")
```

**Implementación:**

```
Para cada cabeza h (ej: 8 cabezas):
  Q_h = X @ W_Q^h
  K_h = X @ W_K^h
  V_h = X @ W_V^h

  head_h = Attention(Q_h, K_h, V_h)

Concatena todas las cabezas:
  multi_head = concat(head_1, head_2, ..., head_8)

Proyección linear de salida:
  output = multi_head @ W_O
```

El modelo aprende automáticamente a asignar diferentes tipos de atención a diferentes cabezas. Es una forma elegante de paralelizar múltiples tipos de análisis.

```{admonition} 📚 Conexión
:class: seealso
Multi-head attention es la razón por la que modelos como GPT y BERT pueden entender relaciones complejas en el lenguaje. En la Lectura 5 veremos cómo BERT usa esto de forma bidireccional mientras GPT lo usa causalmente (solo tokens anteriores).
```

```{admonition} 🎮 Simulación Interactiva: Transformer Explainer
:class: tip

Explora cómo funciona un Transformer paso a paso:

<iframe src="https://poloclub.github.io/transformer-explainer/" width="100%" height="700px" style="border:1px solid #ddd; border-radius:8px;"></iframe>

*Desarrollado por Georgia Tech. Escribe tu propio texto y observa cómo se calcula la atención.*
```

---

## Parte 4: Codificación Posicional

Un problema: la atención ignora orden. Si mezclas las palabras:

```
Original: "gato saltó cerca"
Mezclado: "saltó cerca gato"

Sin codificación posicional, la atención vería exactamente lo mismo
(los mismos embeddings, solo en diferente orden).
```

**Solución:** Suma una **codificación posicional** a cada embedding que encode la posición.

```
embedding_final_pos_0 = embedding["gato"] + pos_encoding[0]
embedding_final_pos_1 = embedding["saltó"] + pos_encoding[1]
embedding_final_pos_2 = embedding["cerca"] + pos_encoding[2]
```

**Fórmula de Vaswani (original):**

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

donde:
- pos es la posición en la secuencia (0, 1, 2, ...)
- i es el índice de la dimensión (0, 1, 2, ..., d_model/2)
- d_model es la dimensión total del embedding (ej: 768)
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_len, d_model):
    """
    Genera codificaciones posicionales usando seno y coseno
    max_len: longitud máxima de secuencia
    d_model: dimensión del modelo
    """
    pe = np.zeros((max_len, d_model))

    position = np.arange(0, max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Aplicar seno a posiciones pares
    pe[:, 0::2] = np.sin(position * div_term)
    # Aplicar coseno a posiciones impares
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

# Generar codificaciones posicionales
max_len = 100
d_model = 128

pe = positional_encoding(max_len, d_model)

print("Codificación Posicional de Transformers")
print("=" * 60)
print(f"Longitud máxima de secuencia: {max_len}")
print(f"Dimensión del modelo: {d_model}")
print(f"\nForma de PE: {pe.shape}")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Heatmap completo de codificaciones posicionales
im = axes[0, 0].imshow(pe.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
axes[0, 0].set_xlabel('Posición en la Secuencia', fontsize=11)
axes[0, 0].set_ylabel('Dimensión del Embedding', fontsize=11)
axes[0, 0].set_title('Codificaciones Posicionales (Heatmap)', fontsize=12, weight='bold')
plt.colorbar(im, ax=axes[0, 0], label='Valor')

# 2. Primeras 10 dimensiones para las primeras 50 posiciones
axes[0, 1].imshow(pe[:50, :10].T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
axes[0, 1].set_xlabel('Posición', fontsize=11)
axes[0, 1].set_ylabel('Dimensión', fontsize=11)
axes[0, 1].set_title('Zoom: Primeras 10 dimensiones, 50 posiciones', fontsize=12, weight='bold')
plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])

# 3. Patrones sinusoidales para diferentes dimensiones
positions = np.arange(max_len)
dims_to_plot = [0, 4, 16, 64]
for dim in dims_to_plot:
    axes[1, 0].plot(positions, pe[:, dim], label=f'Dim {dim}', linewidth=1.5)

axes[1, 0].set_xlabel('Posición en la Secuencia', fontsize=11)
axes[1, 0].set_ylabel('Valor de PE', fontsize=11)
axes[1, 0].set_title('Patrones Sinusoidales por Dimensión', fontsize=12, weight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Similitud entre posiciones (producto punto)
# Calcular similitud entre posición 0 y todas las demás
pos_0 = pe[0]
similarities = [np.dot(pos_0, pe[i]) / (np.linalg.norm(pos_0) * np.linalg.norm(pe[i]))
                for i in range(max_len)]

axes[1, 1].plot(positions, similarities, 'b-', linewidth=2)
axes[1, 1].set_xlabel('Posición', fontsize=11)
axes[1, 1].set_ylabel('Similitud con Posición 0', fontsize=11)
axes[1, 1].set_title('Similitud Posicional (Cosine Similarity)', fontsize=12, weight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Mostrar valores específicos
print("\nEjemplos de Codificaciones Posicionales:")
print("-" * 60)
for pos in [0, 1, 5, 10]:
    print(f"Posición {pos}: {pe[pos, :5].round(3)}... (primeras 5 dims)")

print("\n¿Por qué funciona?")
print("-" * 60)
print("1. Frecuencias diferentes capturan relaciones a diferentes escalas")
print("2. Seno/Coseno permiten interpolar posiciones no vistas")
print("3. Posiciones relativas mantienen patrones consistentes")
print("4. El modelo puede aprender a usar estas señales posicionales")

# Demostrar adición a embeddings
print("\n\nAdición de PE a Embeddings:")
print("-" * 60)
# Simular un embedding de palabra
word_embedding = np.random.randn(d_model) * 0.5
pos = 5

print(f"Embedding original (pos {pos}): {word_embedding[:5].round(3)}...")
print(f"PE para posición {pos}:         {pe[pos, :5].round(3)}...")
final_embedding = word_embedding + pe[pos]
print(f"Embedding final:                {final_embedding[:5].round(3)}...")
```

Esto crea patrones sinusoidales a diferentes frecuencias, permitiendo al modelo aprender relaciones posicionales.

**Nota moderna:** Algunos modelos nuevos usan codificaciones posicionales aprendidas o relativas (Rotary Embeddings en Llama 2). Pero el concepto es igual.

---

## Parte 5: Normalización y Feed-Forward

El Transformer no es solo atención. Tiene componentes adicionales cruciales:

### Layer Normalization

```
Problema: Durante el entrenamiento, la distribución de valores
en cada capa cambia. Esto ralentiza el aprendizaje (covariate shift).

Solución: Normaliza la media y varianza en cada capa.

Normalización de capa:
x_normalized = (x - mean(x)) / sqrt(var(x) + ε)

Luego aprende parámetros γ y β que reescalan:
output = γ * x_normalized + β
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """
    Layer Normalization
    x: input tensor (puede ser multidimensional)
    gamma, beta: parámetros aprendibles (escala y desplazamiento)
    eps: pequeño valor para estabilidad numérica
    """
    # Calcular media y varianza
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    # Normalizar
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Aplicar transformación afín (si se proporciona)
    if gamma is not None and beta is not None:
        x_norm = gamma * x_norm + beta

    return x_norm, mean, var

# Demostración
np.random.seed(42)

# Simular activaciones de una capa con distribución variable
batch_size = 5
d_model = 8

# Activaciones sin normalizar (diferentes escalas)
activations = np.random.randn(batch_size, d_model)
activations[0] *= 10  # Una muestra con valores grandes
activations[1] *= 0.1  # Una muestra con valores pequeños

print("Layer Normalization - Demostración")
print("=" * 60)
print(f"\nActivaciones originales (batch_size={batch_size}, d_model={d_model}):")
print(activations.round(2))

# Aplicar Layer Norm
gamma = np.ones(d_model)  # Parámetro de escala
beta = np.zeros(d_model)  # Parámetro de desplazamiento
activations_norm, means, variances = layer_norm(activations, gamma, beta)

print(f"\nDespués de Layer Normalization:")
print(activations_norm.round(2))

print(f"\nEstadísticas por muestra:")
print(f"{'Muestra':<10} {'Media Original':<20} {'Var Original':<20} {'Media Norm':<15} {'Var Norm':<15}")
print("-" * 80)
for i in range(batch_size):
    orig_mean = np.mean(activations[i])
    orig_var = np.var(activations[i])
    norm_mean = np.mean(activations_norm[i])
    norm_var = np.var(activations_norm[i])
    print(f"{i:<10} {orig_mean:<20.4f} {orig_var:<20.4f} {norm_mean:<15.6f} {norm_var:<15.4f}")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribución antes de normalización
for i in range(batch_size):
    axes[0, 0].plot(activations[i], marker='o', label=f'Muestra {i}', alpha=0.7)
axes[0, 0].set_xlabel('Dimensión', fontsize=11)
axes[0, 0].set_ylabel('Valor', fontsize=11)
axes[0, 0].set_title('Activaciones ANTES de Layer Norm', fontsize=12, weight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)

# 2. Distribución después de normalización
for i in range(batch_size):
    axes[0, 1].plot(activations_norm[i], marker='o', label=f'Muestra {i}', alpha=0.7)
axes[0, 1].set_xlabel('Dimensión', fontsize=11)
axes[0, 1].set_ylabel('Valor', fontsize=11)
axes[0, 1].set_title('Activaciones DESPUÉS de Layer Norm', fontsize=12, weight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)

# 3. Histogramas comparativos
axes[1, 0].hist(activations.flatten(), bins=30, alpha=0.7, color='red',
                edgecolor='black', label='Original')
axes[1, 0].set_xlabel('Valor', fontsize=11)
axes[1, 0].set_ylabel('Frecuencia', fontsize=11)
axes[1, 0].set_title('Distribución Original', fontsize=12, weight='bold')
axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].hist(activations_norm.flatten(), bins=30, alpha=0.7, color='green',
                edgecolor='black', label='Normalizado')
axes[1, 1].set_xlabel('Valor', fontsize=11)
axes[1, 1].set_ylabel('Frecuencia', fontsize=11)
axes[1, 1].set_title('Distribución Normalizada', fontsize=12, weight='bold')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n\nBeneficios de Layer Normalization:")
print("-" * 60)
print("1. Estabiliza el entrenamiento al mantener distribuciones consistentes")
print("2. Permite usar tasas de aprendizaje más altas")
print("3. Reduce la dependencia de la inicialización de pesos")
print("4. Actúa como regularizador (reduce overfitting)")
print("5. Cada muestra se normaliza independientemente (importante en NLP)")

# Comparación con Batch Normalization
print("\n\nLayer Norm vs Batch Norm:")
print("-" * 60)
print("LAYER NORM (usado en Transformers):")
print("  - Normaliza a través de features (dentro de cada muestra)")
print("  - Independiente del batch size")
print("  - Mejor para secuencias de longitud variable")
print("\nBATCH NORM (usado en CNNs):")
print("  - Normaliza a través del batch (para cada feature)")
print("  - Depende del batch size")
print("  - Mejor para visión computacional")
```

La normalización de capa ayuda a que el entrenamiento sea más estable y rápido.

### Red Feed-Forward (FFN)

Después de la atención multi-cabeza, cada Transformer agrega una red neuronal simple:

```
FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2

Es decir:
1. Proyecta a dimensión más alta (típicamente 4x) con ReLU
2. Proyecta de vuelta a dimensión original

Ejemplo (d_model=768):
  x (768) → W_1 → (3072) → ReLU → W_2 → (768)
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def feed_forward_network(x, W1, b1, W2, b2):
    """
    Feed-Forward Network de 2 capas con ReLU
    x: input (d_model,)
    W1: weights capa 1 (d_model, d_ff)
    b1: bias capa 1 (d_ff,)
    W2: weights capa 2 (d_ff, d_model)
    b2: bias capa 2 (d_model,)
    """
    # Capa 1: expandir a dimensión mayor
    hidden = relu(x @ W1 + b1)

    # Capa 2: proyectar de vuelta a dimensión original
    output = hidden @ W2 + b2

    return output, hidden

# Configuración
np.random.seed(42)
d_model = 64  # Dimensión del modelo
d_ff = 256    # Dimensión de la capa intermedia (4x)

# Inicializar pesos
W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)

# Input simulado (salida de atención)
x = np.random.randn(d_model)

# Forward pass
output, hidden = feed_forward_network(x, W1, b1, W2, b2)

print("Feed-Forward Network (FFN) en Transformers")
print("=" * 60)
print(f"\nArquitectura:")
print(f"  Input:      d_model = {d_model}")
print(f"  Hidden:     d_ff = {d_ff} (factor de expansión: {d_ff/d_model:.1f}x)")
print(f"  Output:     d_model = {d_model}")

print(f"\nPesos:")
print(f"  W1 shape: {W1.shape} ({W1.size:,} parámetros)")
print(f"  W2 shape: {W2.shape} ({W2.size:,} parámetros)")
print(f"  Total FFN parámetros: {W1.size + W2.size + d_ff + d_model:,}")

print(f"\nEjemplo Forward Pass:")
print(f"  Input:  {x[:5].round(3)}... (primeros 5 valores)")
print(f"  Hidden: {hidden[:5].round(3)}... (después de ReLU, primeros 5)")
print(f"  Output: {output[:5].round(3)}... (primeros 5 valores)")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Arquitectura visual
ax = axes[0, 0]
ax.axis('off')

# Dibujar neuronas
input_y = np.linspace(0.1, 0.9, min(10, d_model))
hidden_y = np.linspace(0.05, 0.95, min(20, d_ff))
output_y = np.linspace(0.1, 0.9, min(10, d_model))

# Input layer
for y in input_y:
    circle = plt.Circle((0.1, y), 0.02, color='lightblue', ec='black', zorder=3)
    ax.add_patch(circle)

# Hidden layer
for y in hidden_y:
    circle = plt.Circle((0.5, y), 0.015, color='lightgreen', ec='black', zorder=3)
    ax.add_patch(circle)

# Output layer
for y in output_y:
    circle = plt.Circle((0.9, y), 0.02, color='lightcoral', ec='black', zorder=3)
    ax.add_patch(circle)

# Conexiones (muestra solo algunas)
for yi in input_y[::2]:
    for yh in hidden_y[::4]:
        ax.plot([0.12, 0.485], [yi, yh], 'gray', alpha=0.2, linewidth=0.5, zorder=1)

for yh in hidden_y[::4]:
    for yo in output_y[::2]:
        ax.plot([0.515, 0.88], [yh, yo], 'gray', alpha=0.2, linewidth=0.5, zorder=1)

# Labels
ax.text(0.1, 0.05, f'Input\n({d_model})', ha='center', fontsize=10, weight='bold')
ax.text(0.5, 0.02, f'Hidden + ReLU\n({d_ff})', ha='center', fontsize=10, weight='bold')
ax.text(0.9, 0.05, f'Output\n({d_model})', ha='center', fontsize=10, weight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Arquitectura FFN', fontsize=12, weight='bold')

# 2. Distribución de activaciones
axes[0, 1].hist(x, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Input')
axes[0, 1].hist(output, bins=30, alpha=0.7, color='red', edgecolor='black', label='Output')
axes[0, 1].set_xlabel('Valor', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].set_title('Distribución de Activaciones', fontsize=12, weight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Efecto de ReLU en capa oculta
pre_relu = x @ W1 + b1
post_relu = hidden

axes[1, 0].scatter(range(len(pre_relu[:50])), pre_relu[:50],
                  alpha=0.6, s=20, label='Antes de ReLU', color='orange')
axes[1, 0].scatter(range(len(post_relu[:50])), post_relu[:50],
                  alpha=0.6, s=20, label='Después de ReLU', color='green')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('Neurona (primeras 50)', fontsize=11)
axes[1, 0].set_ylabel('Activación', fontsize=11)
axes[1, 0].set_title('Efecto de ReLU en Capa Oculta', fontsize=12, weight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Comparación input vs output
axes[1, 1].plot(x[:50], 'b-', linewidth=2, label='Input', alpha=0.7)
axes[1, 1].plot(output[:50], 'r-', linewidth=2, label='Output', alpha=0.7)
axes[1, 1].set_xlabel('Dimensión (primeras 50)', fontsize=11)
axes[1, 1].set_ylabel('Valor', fontsize=11)
axes[1, 1].set_title('Input vs Output del FFN', fontsize=12, weight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n\nPropósito del FFN:")
print("-" * 60)
print("1. Introduce NO-LINEALIDAD adicional (crucial para capacidad del modelo)")
print("2. Procesa cada posición INDEPENDIENTEMENTE (no hay interacción entre tokens)")
print("3. Expande y comprime información (bottleneck ayuda a generalizar)")
print("4. Factor 4x es estándar, pero varía (GPT-3 usa ~4x, algunos usan hasta 8x)")

sparsity = np.sum(hidden == 0) / len(hidden) * 100
print(f"\nEsparsidad en capa oculta (% de ceros por ReLU): {sparsity:.1f}%")
```

Esta red **introduce no-linealidad adicional** y permite aprender transformaciones más complejas por token.

### Residual Connections

```
En lugar de:
  x' = Atention(x)

Usa:
  x' = x + Attention(x)
```

Las conexiones residuales ayudan a los gradientes a fluir a través de redes profundas y estabiliza el entrenamiento.

---

## Parte 6: Bloques y Capas del Transformer

### Un Bloque Transformer Completo

```
Entrada: x (secuencia de embeddings)
         ↓
[Multi-Head Attention]
         ↓
[Add + LayerNorm]  (residual connection + normalización)
         ↓
[Feed-Forward Network]
         ↓
[Add + LayerNorm]  (residual connection + normalización)
         ↓
Salida: x' (secuencia transformada)
```

:::{figure} diagrams/transformer_block.png
:name: fig-transformer-block
:alt: Arquitectura completa de un bloque Transformer
:align: center
:width: 90%

**Figura 5:** Bloque Transformer Completo - atención, normalización y red feed-forward.
:::

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Funciones auxiliares
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn_weights = softmax(scores)
    output = attn_weights @ V
    return output, attn_weights

class TransformerBlock:
    """Bloque Transformer simplificado"""
    def __init__(self, d_model, d_ff, n_heads):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Pesos de atención (simplificado: 1 cabeza)
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

        # Pesos FFN
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass del bloque Transformer
        x: (seq_len, d_model)
        """
        # 1. Multi-Head Attention (simplificado a 1 cabeza)
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        attn_output, attn_weights = attention(Q, K, V, mask)
        attn_output = attn_output @ self.W_O

        # 2. Add & Norm (primera residual connection)
        x = x + attn_output  # Residual connection
        x = layer_norm(x)    # Layer normalization

        # Guardar estado intermedio
        after_attn = x.copy()

        # 3. Feed-Forward Network
        ffn_hidden = relu(x @ self.W1 + self.b1)
        ffn_output = ffn_hidden @ self.W2 + self.b2

        # 4. Add & Norm (segunda residual connection)
        x = x + ffn_output   # Residual connection
        x = layer_norm(x)    # Layer normalization

        return x, attn_weights, after_attn

# Configuración
np.random.seed(42)
seq_len = 6
d_model = 64
d_ff = 256
n_heads = 1

# Input: secuencia de embeddings
words = ['El', 'gato', 'saltó', 'sobre', 'la', 'cerca']
x_input = np.random.randn(seq_len, d_model) * 0.5

# Crear máscara causal (para decoder)
causal_mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)

# Crear bloque Transformer
transformer_block = TransformerBlock(d_model, d_ff, n_heads)

# Forward pass
x_output, attn_weights, after_attn = transformer_block.forward(x_input, mask=causal_mask)

print("Bloque Transformer Completo - Demostración")
print("=" * 60)
print(f"\nConfiguración:")
print(f"  Secuencia: {words}")
print(f"  seq_len: {seq_len}")
print(f"  d_model: {d_model}")
print(f"  d_ff: {d_ff}")
print(f"  n_heads: {n_heads} (simplificado)")

print(f"\nFormas de tensores:")
print(f"  Input:           {x_input.shape}")
print(f"  Después de Attn: {after_attn.shape}")
print(f"  Output final:    {x_output.shape}")

print(f"\nEstadísticas:")
print(f"  Input - media: {x_input.mean():.4f}, std: {x_input.std():.4f}")
print(f"  Output - media: {x_output.mean():.4f}, std: {x_output.std():.4f}")

# Visualización
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Máscara causal
ax1 = fig.add_subplot(gs[0, 0])
mask_display = causal_mask.copy()
mask_display[mask_display < -1e8] = 0
mask_display[mask_display == 0] = 1
im1 = ax1.imshow(mask_display, cmap='RdYlGn', aspect='auto')
ax1.set_xticks(range(seq_len))
ax1.set_yticks(range(seq_len))
ax1.set_xticklabels(words, rotation=45)
ax1.set_yticklabels(words)
ax1.set_title('Máscara Causal\n(1=visible, 0=bloqueado)', fontsize=11, weight='bold')
plt.colorbar(im1, ax=ax1)

# 2. Pesos de atención
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(attn_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax2.set_xticks(range(seq_len))
ax2.set_yticks(range(seq_len))
ax2.set_xticklabels(words, rotation=45)
ax2.set_yticklabels(words)
ax2.set_title('Pesos de Atención\n(con máscara causal)', fontsize=11, weight='bold')
plt.colorbar(im2, ax=ax2)

# 3. Diagrama de flujo
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
flow_steps = [
    'Input',
    '↓',
    'Multi-Head\nAttention',
    '↓',
    'Add + Norm',
    '↓',
    'Feed-Forward',
    '↓',
    'Add + Norm',
    '↓',
    'Output'
]
for i, step in enumerate(flow_steps):
    y = 1 - i * 0.09
    if '↓' in step:
        ax3.text(0.5, y, step, ha='center', va='center', fontsize=14)
    else:
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2)
        ax3.text(0.5, y, step, ha='center', va='center', fontsize=10, weight='bold', bbox=bbox_props)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('Flujo del Bloque', fontsize=11, weight='bold')

# 4-6. Distribuciones de activaciones
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(x_input.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
ax4.set_xlabel('Valor', fontsize=10)
ax4.set_ylabel('Frecuencia', fontsize=10)
ax4.set_title('Distribución: Input', fontsize=11, weight='bold')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(after_attn.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
ax5.set_xlabel('Valor', fontsize=10)
ax5.set_ylabel('Frecuencia', fontsize=10)
ax5.set_title('Distribución: Después de Atención', fontsize=11, weight='bold')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax5.grid(True, alpha=0.3, axis='y')

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(x_output.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
ax6.set_xlabel('Valor', fontsize=10)
ax6.set_ylabel('Frecuencia', fontsize=10)
ax6.set_title('Distribución: Output Final', fontsize=11, weight='bold')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax6.grid(True, alpha=0.3, axis='y')

# 7-9. Embeddings por token
ax7 = fig.add_subplot(gs[2, :])
for i, word in enumerate(words):
    ax7.plot(x_input[i, :30], alpha=0.5, linewidth=1, label=f'{word} (input)')
    ax7.plot(x_output[i, :30], alpha=0.8, linewidth=2, linestyle='--', label=f'{word} (output)')

ax7.set_xlabel('Dimensión (primeras 30)', fontsize=11)
ax7.set_ylabel('Valor', fontsize=11)
ax7.set_title('Transformación de Embeddings: Input → Output', fontsize=12, weight='bold')
ax7.legend(ncol=3, fontsize=8)
ax7.grid(True, alpha=0.3)
ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.suptitle('Bloque Transformer: Visualización Completa', fontsize=14, weight='bold', y=0.995)
plt.show()

print("\n\nComponentes Clave:")
print("-" * 60)
print("1. ATENCIÓN: Permite que cada token 'vea' otros tokens")
print("2. RESIDUAL CONNECTIONS: x + sublayer(x) → ayuda al flujo de gradientes")
print("3. LAYER NORM: Estabiliza distribuciones → entrenamiento más rápido")
print("4. FFN: Introduce no-linealidad y capacidad expresiva")
print("5. MÁSCARA CAUSAL: En decoders, previene mirar hacia el futuro")

print("\n\nPor qué funciona:")
print("-" * 60)
print("- Atención captura DEPENDENCIAS entre tokens")
print("- FFN procesa cada token INDEPENDIENTEMENTE")
print("- Residual connections permiten APILAR muchos bloques")
print("- Layer norm mantiene ESTABILIDAD durante entrenamiento")
print("- Combinado: modelo muy profundo y expresivo")
```

### Encoder vs Decoder

```
ENCODER:
- Puede ver todo el contexto simultáneamente
- Salida: representaciones contextualizadas
- Ejemplo: BERT (clasificación, preguntas-respuestas)

DECODER:
- Solo puede ver palabras anteriores (causal masking)
- Razón: genera token por token, no debería "mirar adelante"
- Salida: predicción del siguiente token
- Ejemplo: GPT (generación de texto)
```

:::{figure} diagrams/encoder_vs_decoder.png
:name: fig-encoder-decoder
:alt: Comparación de arquitectura encoder vs decoder
:align: center
:width: 90%

**Figura 6:** Encoder vs Decoder - diferencias en visibilidad de contexto y aplicaciones.
:::

**Causal Masking:**

```
Posición 0 puede atender a: [0]
Posición 1 puede atender a: [0, 1]
Posición 2 puede atender a: [0, 1, 2]
...

Esto se logra poniendo scores -∞ para posiciones futuras,
así softmax los convierte en 0.
```

### Stacking de Capas

Los modelos modernos apilan múltiples bloques:

```
Entrada
  ↓
[Bloque 1]
  ↓
[Bloque 2]
  ↓
... (12, 24, 32, ... bloques)
  ↓
Salida
```

Más bloques = mayor capacidad pero mayor costo computacional.

---

## Parte 7: La Arquitectura Completa Original

Para un modelo como GPT:

```
Entrada (texto):        "el gato saltó"
         ↓
[Tokenización]          [el] [gato] [saltó]
         ↓
[Token Embedding]       [e_1] [e_2] [e_3]  (768-dimensional)
         ↓
[Pos Encoding]          [e_1 + PE_0] [e_2 + PE_1] [e_3 + PE_2]
         ↓
[N Bloques Transformer]
  Cada bloque:
    - Multi-Head Attention
    - Add + LayerNorm
    - FFN
    - Add + LayerNorm
         ↓
[Output Projection]     [logits para 50,000 tokens]
         ↓
[Softmax]               [probabilidades]
         ↓
Predicción:             Siguiente token más probable
```

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **¿Por qué la atención multi-cabeza es mejor que una sola cabeza?** ¿Qué perdería un modelo con una sola cabeza de atención?

2. **Causal masking:** Explica por qué un modelo como GPT que genera texto token por token necesita causal masking durante el entrenamiento.

3. **Comparación:** ¿Cuál es la ventaja fundamental de Transformers sobre RNNs? ¿Hay alguna ventaja de RNNs que los Transformers no tengan?

### Ejercicios Prácticos:

1. **Atención manual:** Tienes embeddings para "el", "gato", "saltó":
   ```
   "el" → [1, 0]
   "gato" → [0.9, 0.1]
   "saltó" → [0.5, 0.5]

   W_Q = W_K = W_V = Identidad (para simplificar)

   Calcula la atención de "saltó" a todas las palabras:
   1. Query para "saltó"
   2. Keys para todas
   3. Scores (producto punto)
   4. Softmax
   5. Agregación de values
   ```

2. **Codificación posicional:** Calcula PE(pos=0, 2i=0) y PE(pos=1, 2i=0) con d_model=512.

3. **Reflexión escrita (350 palabras):** "Los Transformers reemplazaron las RNNs porque procesamiento paralelo + atención directa > procesamiento secuencial. Pero ¿por qué aún investigadores estudian modelos secuenciales? ¿Hay escenarios donde secuencial es mejor?"

---

```{admonition} ✅ Verifica tu comprensión
:class: note
1. ¿Por qué la atención multi-cabeza es mejor que una sola cabeza? ¿Qué perdería un modelo con una sola cabeza?
2. Explica con tus palabras por qué un modelo como GPT necesita causal masking durante el entrenamiento.
3. ¿Cuál es la ventaja fundamental de Transformers sobre RNNs? ¿Hay alguna ventaja de RNNs que los Transformers no tengan?
4. Sin codificación posicional, ¿por qué "El gato persiguió al perro" sería idéntico a "El perro persiguió al gato" para el modelo?
```

## Resumen

```{admonition} Resumen
:class: important
**Conceptos clave:**
- Mecanismo de atención permite que cada token "vea" todos los demás tokens mediante Query, Key, Value
- Fórmula fundamental: Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
- Multi-head attention captura diferentes tipos de relaciones en paralelo (sintaxis, semántica, etc.)
- Codificación posicional (sinusoidal o aprendida) agrega información de orden, crucial para entender secuencias
- Bloque Transformer completo: Atención → Add&Norm → FFN → Add&Norm
- Encoder (BERT): atención bidireccional para comprensión; Decoder (GPT): atención causal para generación

**Para la siguiente lectura:** Prepárate para conocer modelos específicos como BERT y GPT, y profundizar en tokenización (BPE, WordPiece). Veremos cómo convertir texto en números que estos Transformers procesan.
```

## Puntos Clave

- **Atención:** Cada elemento consulta (Q) a todos los demás elementos, usando claves (K) y valores (V)
- **Fórmula:** Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
- **Multi-head:** Múltiples cabezas capturan diferentes tipos de relaciones
- **Codificación posicional:** Agrega información de posición (sin ella, el orden no importaría)
- **Normalización + FFN:** Estabilizan entrenamiento e introducen no-linealidad
- **Conexiones residuales:** Permiten entrenar modelos profundos
- **Encoder vs Decoder:** Encoder ve todo contexto; Decoder solo tokens anteriores (causal)

---

## Referencias

- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Ba, J., Kiros, J., & Hinton, G. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). arXiv:1607.06450.
- He, K. et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). ICLR 2015.

