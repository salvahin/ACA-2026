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

# Solucionario de Ejercicios

> Este documento contiene las soluciones a los ejercicios prácticos de cada lectura del módulo de Inteligencia Artificial.

---

## Lectura 1: Introducción al Aprendizaje Automático

### Ejercicio: Softmax Manual

**Problema:** Dado logits = [2.0, 1.0, 0.5], calcula las probabilidades usando softmax.

**Solución:**

```python
import numpy as np

logits = np.array([2.0, 1.0, 0.5])

# Paso 1: Calcular exponenciales
exp_logits = np.exp(logits)
# exp([2.0, 1.0, 0.5]) = [7.389, 2.718, 1.649]

# Paso 2: Sumar exponenciales
sum_exp = np.sum(exp_logits)
# sum = 7.389 + 2.718 + 1.649 = 11.756

# Paso 3: Dividir cada exponencial por la suma
probs = exp_logits / sum_exp
# probs = [0.628, 0.231, 0.140]

print(f"Probabilidades: {probs.round(3)}")
# Verificar: sum(probs) = 1.0
```

**Respuesta:** P = [0.628, 0.231, 0.140]

---

### Ejercicio: Identificar Paradigmas

**Problema:** Para diagnóstico médico, recomendaciones de productos y detección de fraude, identifica el paradigma más apropiado.

**Solución:**

| Problema | Paradigma Recomendado | Justificación |
|----------|----------------------|---------------|
| Diagnóstico médico | Discriminativo (supervisado) | Clasificación binaria/multiclase con datos etiquetados por expertos. Se necesita P(enfermedad\|síntomas). |
| Recomendaciones | Generativo o Híbrido | Modela preferencias del usuario P(items\|usuario). Collaborative filtering o modelos generativos. |
| Detección de fraude | Discriminativo + No supervisado | Clasificación supervisada si hay etiquetas. Detección de anomalías (no supervisado) para patrones nuevos. |

---

## Lectura 2: Fundamentos de Deep Learning

### Ejercicio: Calcular Parámetros de Red

**Problema:** Calcula el número de parámetros para una red [784, 128, 64, 10].

**Solución:**

```python
def count_params(layers):
    total = 0
    for i in range(len(layers) - 1):
        weights = layers[i] * layers[i+1]
        biases = layers[i+1]
        total += weights + biases
    return total

layers = [784, 128, 64, 10]
# Capa 1: 784 * 128 + 128 = 100,480
# Capa 2: 128 * 64 + 64 = 8,256
# Capa 3: 64 * 10 + 10 = 650
# Total: 109,386

print(f"Total parámetros: {count_params(layers):,}")
```

**Respuesta:** 109,386 parámetros

---

### Ejercicio: Identificar Overfitting

**Problema:** Training loss = 0.01, Validation loss = 2.5. ¿Qué está pasando?

**Solución:**

Esto es un caso claro de **overfitting**:

1. **Training loss muy bajo (0.01):** El modelo memorizó los datos de entrenamiento
2. **Validation loss muy alto (2.5):** El modelo no generaliza a datos nuevos
3. **Ratio:** val_loss / train_loss = 250x indica sobreajuste severo

**Mitigaciones:**
- Agregar dropout (0.3-0.5)
- Usar early stopping
- Aumentar datos de entrenamiento
- Reducir complejidad del modelo
- Aplicar regularización L2

---

## Lectura 3: Generación Autoregresiva

### Ejercicio: Softmax con Temperatura

**Problema:** Logits = [2.0, 0.5, 1.5]. Calcula softmax para T=0.5, T=1.0 y T=2.0.

**Solución:**

```python
import numpy as np

def softmax_temp(logits, T):
    scaled = logits / T
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()

logits = np.array([2.0, 0.5, 1.5])

print("T=0.5 (determinístico):", softmax_temp(logits, 0.5).round(3))
# [0.817, 0.018, 0.165] - muy concentrado en el máximo

print("T=1.0 (original):", softmax_temp(logits, 1.0).round(3))
# [0.576, 0.128, 0.296] - distribución original

print("T=2.0 (aleatorio):", softmax_temp(logits, 2.0).round(3))
# [0.422, 0.211, 0.366] - más uniforme
```

**Respuesta:**
- T=0.5: [0.817, 0.018, 0.165]
- T=1.0: [0.576, 0.128, 0.296]
- T=2.0: [0.422, 0.211, 0.366]

---

### Ejercicio: BPE Simulado

**Problema:** Documento: "aaabaaab". Aplica BPE paso a paso.

**Solución:**

```
Documento inicial: a a a b a a a b

Paso 1: Contar pares
  "aa" aparece 4 veces (posiciones 1-2, 2-3, 5-6, 6-7)
  "ab" aparece 2 veces (posiciones 3-4, 7-8)
  "ba" aparece 1 vez (posición 4-5)

  Par más común: "aa" → reemplazar con X

Paso 2: Aplicar reemplazo
  "aaabaaab" → "XabXab"
  Vocabulario: {a, b, X="aa"}

Paso 3: Contar pares nuevamente
  "Xa" aparece 2 veces
  "ab" aparece 2 veces
  "bX" aparece 1 vez

  Empate: elegir "Xa" → reemplazar con Y

Paso 4: Aplicar reemplazo
  "XabXab" → "YbYb"
  Vocabulario: {a, b, X="aa", Y="Xa"="aaa"}

Estado final: "YbYb" con vocab={a, b, X, Y}
```

**Respuesta:** Vocabulario final: {a, b, aa, aaa}

---

### Ejercicio: Greedy vs Top-P

**Problema:**
- Caso 1: [0.9, 0.05, 0.03, 0.02]
- Caso 2: [0.2, 0.2, 0.2, 0.2, 0.2]

**Solución:**

| Caso | Estrategia Recomendada | Justificación |
|------|------------------------|---------------|
| Caso 1 | **Greedy** | Token dominante (0.9). Greedy es eficiente y correcto. Top-P con p=0.9 también funcionaría. |
| Caso 2 | **Top-P (p=0.9)** | Distribución uniforme. Greedy elegiría arbitrariamente. Top-P permite explorar opciones equivalentes. |

---

## Lectura 4: Arquitectura Transformer

### Ejercicio: Atención Manual

**Problema:** Embeddings: "el"=[1,0], "gato"=[0.9,0.1], "saltó"=[0.5,0.5]. Calcula atención de "saltó" con W_Q=W_K=W_V=I.

**Solución:**

```python
import numpy as np

# Embeddings
el = np.array([1, 0])
gato = np.array([0.9, 0.1])
salto = np.array([0.5, 0.5])

# Con W_Q = W_K = W_V = Identidad:
Q_salto = salto  # [0.5, 0.5]
K = np.array([el, gato, salto])  # Stack de keys
V = K  # Igual que keys

# Paso 1: Scores (producto punto)
d_k = 2
scores = Q_salto @ K.T / np.sqrt(d_k)
# scores = [0.5*1 + 0.5*0, 0.5*0.9 + 0.5*0.1, 0.5*0.5 + 0.5*0.5] / sqrt(2)
# scores = [0.5, 0.5, 0.5] / 1.414 = [0.354, 0.354, 0.354]

# Paso 2: Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

attn_weights = softmax(scores)
# Como todos los scores son iguales: [0.333, 0.333, 0.333]

# Paso 3: Agregar values
output = attn_weights @ V
# output = 0.333*[1,0] + 0.333*[0.9,0.1] + 0.333*[0.5,0.5]
# output = [0.8, 0.2]

print(f"Scores: {scores.round(3)}")
print(f"Attention weights: {attn_weights.round(3)}")
print(f"Output: {output.round(3)}")
```

**Respuesta:**
- Scores: [0.354, 0.354, 0.354]
- Weights: [0.333, 0.333, 0.333]
- Output: [0.8, 0.2]

---

### Ejercicio: Codificación Posicional

**Problema:** Calcula PE(pos=0, 2i=0) y PE(pos=1, 2i=0) con d_model=512.

**Solución:**

```python
import numpy as np

d_model = 512

# Fórmula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
# Para 2i = 0:

def PE_sin(pos, i, d_model):
    return np.sin(pos / (10000 ** (2*i / d_model)))

# PE(pos=0, 2i=0)
pe_0_0 = PE_sin(0, 0, 512)
# sin(0 / 10000^0) = sin(0) = 0

# PE(pos=1, 2i=0)
pe_1_0 = PE_sin(1, 0, 512)
# sin(1 / 10000^0) = sin(1) = 0.8414

print(f"PE(pos=0, 2i=0) = {pe_0_0:.4f}")
print(f"PE(pos=1, 2i=0) = {pe_1_0:.4f}")
```

**Respuesta:**
- PE(pos=0, 2i=0) = 0.0000
- PE(pos=1, 2i=0) = 0.8414

---

## Lectura 5: BERT, GPTs y Tokenización

### Ejercicio: Comparar Tokenizadores

**Problema:** ¿Cuántos tokens genera cada tokenizador para "tl.load(x_ptr + offsets)"?

**Solución:**

```python
# GPT-2 tokenizer (tiktoken)
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens_gpt2 = enc.encode("tl.load(x_ptr + offsets)")
# Resultado: ~8-10 tokens (depende de espacios)

# BERT tokenizer
from transformers import BertTokenizer
bert = BertTokenizer.from_pretrained("bert-base-uncased")
tokens_bert = bert.tokenize("tl.load(x_ptr + offsets)")
# Resultado: ~12-15 tokens (WordPiece más agresivo)
```

**Respuesta:** GPT-2 es más eficiente para código (~8 tokens vs ~12 tokens de BERT).

---

## Lectura 6: Sampling y Constrained Decoding

### Ejercicio: Diseña una Máscara de Bits

**Problema:** Crea máscara para solo permitir "verdadero" (500) o "falso" (501).

**Solución:**

```python
import torch

vocab_size = 50257
allowed_tokens = [500, 501]  # "verdadero", "falso"

# Crear máscara
bitmask = torch.zeros(vocab_size, dtype=torch.bool)
for token_id in allowed_tokens:
    bitmask[token_id] = True

# Aplicar a logits
logits = torch.randn(vocab_size)
logits_masked = logits.clone()
logits_masked[~bitmask] = -float("inf")

# Verificar
probs = torch.softmax(logits_masked, dim=-1)
assert probs[500] + probs[501] == 1.0  # Solo estos dos tokens
```

---

### Ejercicio: Mini-Gramática

**Problema:** número ::= [1-9][0-9]* | "0". ¿Permite "007"? ¿"0"?

**Solución:**

- **"007":** ❌ NO permitido. No comienza con [1-9] y no es exactamente "0".
- **"0":** ✅ SÍ permitido. Coincide con la alternativa "0".
- **"42":** ✅ SÍ permitido. Coincide con [1-9][0-9]* (4 seguido de 2).
- **"00":** ❌ NO permitido. No es "0" exacto ni comienza con [1-9].

---

### Ejercicio: Análisis de Overhead

**Problema:** Constrained +10% overhead vs 50% fallo (1.5 reintentos promedio).

**Solución:**

```
Para 10,000 solicitudes:

Sin restricción:
  Intentos totales = 10,000 × 1.5 = 15,000 generaciones

Con restricción:
  Intentos totales = 10,000 × 1.1 = 11,000 generaciones

Ahorro = 15,000 - 11,000 = 4,000 generaciones (26.7% menos)

Conclusión: SÍ vale la pena usar constrained decoding.
```

---

## Lectura 7: Fine-Tuning y Evaluación

### Ejercicio: Fine-Tuning Budget

**Problema:** Presupuesto $500. Opciones: Full ($200), LoRA ($50), QLoRA ($10).

**Solución:**

**Recomendación: QLoRA ($10) + múltiples experimentos**

| Estrategia | Costo | Uso del presupuesto restante |
|------------|-------|------------------------------|
| QLoRA × 5 configuraciones | $50 | $450 para evaluación, datos adicionales |
| LoRA × 2 configuraciones | $100 | $400 para producción |
| Full × 1 | $200 | $300 para iteraciones |

**Mejor estrategia:** Usar QLoRA para explorar hiperparámetros (learning rate, rank, módulos), luego LoRA con la mejor configuración para modelo final.

---

### Ejercicio: Dataset Split

**Problema:** 10,000 ejemplos. ¿Mejor split?

**Solución:**

| Split | Pros | Contras |
|-------|------|---------|
| 8000/1000/1000 | Balanceado | - |
| 9000/500/500 | Más datos de entrenamiento | Validation/test pequeños, menos confiable |
| 7000/2000/1000 | Validation grande para hyperparams | Menos datos de entrenamiento |

**Recomendación:** **8000/1000/1000 (opción a)**

- Suficientes datos de entrenamiento (8K)
- Validation suficiente para early stopping (1K)
- Test suficiente para evaluación final (1K)

---

### Ejercicio: Pass@k Aproximación

**Problema:** Si Pass@1 = 70%, ¿cuál es Pass@5?

**Solución:**

```python
# Fórmula aproximada: Pass@k ≈ 1 - (1 - Pass@1)^k

pass_1 = 0.70
k = 5

pass_k = 1 - (1 - pass_1) ** k
# pass_5 = 1 - (0.30)^5
# pass_5 = 1 - 0.00243
# pass_5 = 0.9976 ≈ 99.8%

print(f"Pass@5 ≈ {pass_k:.1%}")
```

**Respuesta:** Pass@5 ≈ 99.8%

---

### Ejercicio: Benchmark Analysis

**Problema:** SQuAD 90% F1, Preguntas internas 65% F1. ¿Contaminación?

**Solución:**

**Posible contaminación.** Investigar:

1. **Verificar overlap:** ¿Los datos de fine-tuning contienen ejemplos de SQuAD?
2. **Verificar pre-entrenamiento:** ¿El modelo base fue entrenado con Wikipedia (fuente de SQuAD)?
3. **Analizar errores:** ¿En qué tipo de preguntas falla en datos internos?

**Diagnóstico probable:** El modelo memorizó patrones de SQuAD pero no aprendió a generalizar a nuevos dominios.

---

## Lectura 8: MLOps y Visualización

### Ejercicio: Checklist de Reproducibilidad

**Problema:** Enumera los elementos mínimos para reproducir un experimento.

**Solución:**

```markdown
☐ Git commit hash del código
☐ Versión de dependencias (requirements.txt o environment.yml)
☐ Random seeds (Python, NumPy, PyTorch)
☐ Configuración de hiperparámetros (YAML o JSON)
☐ Hash del dataset o versión en DVC
☐ Hardware utilizado (GPU model, CUDA version)
☐ Comando exacto de ejecución
☐ Logs de entrenamiento
☐ Métricas finales
```

---

## Soluciones de Preguntas de Comprensión

### Lectura 1

1. **Discriminativo vs Generativo matemáticamente:**
   - Discriminativo: P(Y|X) - probabilidad de etiqueta dado input
   - Generativo: P(X,Y) = P(X|Y)P(Y) - distribución conjunta

2. **Perceptrón y XOR:**
   XOR no es linealmente separable. Una línea recta no puede separar {(0,0), (1,1)} de {(0,1), (1,0)}.

3. **Importancia de P(Y|X) vs P(X,Y):**
   - Si solo clasificas: P(Y|X) es suficiente y más eficiente
   - Si generas datos o detectas anomalías: necesitas P(X,Y)

4. **Aprendizaje clásico vs deep learning:**
   - Pocos datos (<10K): aprendizaje clásico (Random Forest, SVM)
   - Interpretabilidad crítica: árboles de decisión
   - Recursos limitados: modelos simples

### Lectura 4

1. **Multi-head vs single-head:**
   Multi-head captura diferentes tipos de relaciones (sintaxis, semántica, posición). Una cabeza tendería a colapsar en un solo patrón.

2. **Causal masking en GPT:**
   Durante entrenamiento, el modelo debe predecir el siguiente token sin "hacer trampa" mirando el futuro. Sin máscara, el loss sería trivial.

3. **Transformers vs RNNs:**
   - Transformers: paralelización, conexiones directas a largo plazo
   - RNNs: menor memoria para secuencias muy largas, más eficiente en edge devices

4. **Sin codificación posicional:**
   "El gato persiguió al perro" y "El perro persiguió al gato" tendrían la misma representación porque la atención ignora orden.

---

*Última actualización: Marzo 2026*
