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

# Glosario Técnico

> Definiciones de términos clave utilizados en el módulo de Inteligencia Artificial.

---

## A

**Activation Function (Función de Activación)**
: Función no lineal aplicada a la salida de una neurona. Ejemplos: ReLU, sigmoid, tanh. Sin activaciones no lineales, una red profunda colapsa a una transformación lineal simple.

**Attention (Atención)**
: Mecanismo que permite a cada token "ver" todos los demás tokens en una secuencia. Calcula pesos de relevancia mediante Query, Key, Value: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`.

**Autoencoder**
: Arquitectura que comprime datos a una representación latente (encoder) y los reconstruye (decoder). Útil para aprender representaciones y reducción de dimensionalidad.

**Autoregressive (Autoregresivo)**
: Modelo que genera secuencias un elemento a la vez, condicionando cada predicción en los elementos anteriores. GPT es autoregresivo: predice el siguiente token basándose en todos los anteriores.

---

## B

**Backpropagation (Retropropagación)**
: Algoritmo para calcular gradientes en redes neuronales. Propaga el error desde la salida hacia la entrada usando la regla de la cadena, permitiendo actualizar pesos eficientemente.

**Batch**
: Subconjunto de datos procesado en una iteración de entrenamiento. El tamaño del batch afecta estabilidad de gradientes y uso de memoria.

**Beam Search**
: Estrategia de decodificación que mantiene múltiples hipótesis (secuencias candidatas) en paralelo. Más exhaustivo que greedy pero computacionalmente costoso.

**BERT (Bidirectional Encoder Representations from Transformers)**
: Modelo Transformer encoder que ve contexto en ambas direcciones. Entrenado con Masked Language Modeling (MLM). Excelente para clasificación y comprensión.

**BPE (Byte Pair Encoding)**
: Algoritmo de tokenización que fusiona iterativamente los pares de caracteres más frecuentes. Balancea eficiencia (vocabulario pequeño) y flexibilidad (maneja palabras nuevas).

---

## C

**Causal Masking (Máscara Causal)**
: Técnica que previene que tokens futuros sean visibles durante el entrenamiento de modelos generativos. Implementada poniendo -∞ en posiciones futuras de la matriz de atención.

**Constrained Decoding (Decodificación Restringida)**
: Técnica que fuerza al modelo a generar salidas válidas (JSON, código) anulando logits de tokens inválidos en cada paso.

**Cross-Entropy Loss**
: Función de pérdida estándar para clasificación: `-log(P(clase_correcta))`. Penaliza predicciones incorrectas, especialmente cuando el modelo está muy confiado en la respuesta equivocada.

---

## D

**Decoder**
: Componente de Transformer que genera secuencias. Usa atención causal (solo ve tokens anteriores). GPT es un decoder.

**Discriminative Model (Modelo Discriminativo)**
: Modelo que aprende P(Y|X) - la probabilidad de una etiqueta dado el input. Clasifica pero no genera nuevos datos. Ejemplo: clasificador de spam.

**Dropout**
: Técnica de regularización que "apaga" aleatoriamente neuronas durante entrenamiento. Previene overfitting al forzar redundancia en las representaciones.

---

## E

**Embedding**
: Representación vectorial densa de un token, palabra o concepto. Palabras similares tienen embeddings similares (cercanos en el espacio vectorial).

**Encoder**
: Componente de Transformer que procesa input bidireccionalmente. BERT es un encoder. Produce representaciones contextualizadas.

**Epoch (Época)**
: Una pasada completa por todo el dataset de entrenamiento. El entrenamiento típicamente requiere múltiples épocas.

---

## F

**Feed-Forward Network (FFN)**
: Red neuronal densa de dos capas dentro de cada bloque Transformer. Expande a dimensión mayor (típicamente 4x), aplica ReLU, y proyecta de vuelta.

**Fine-Tuning**
: Proceso de continuar el entrenamiento de un modelo pre-entrenado con datos específicos de una tarea. Adapta el modelo general a un dominio particular.

**Forward Pass (Pase Hacia Adelante)**
: Propagación de datos desde la entrada hasta la salida de una red neuronal. Calcula predicciones usando los pesos actuales.

---

## G

**Generative Model (Modelo Generativo)**
: Modelo que aprende P(X,Y) o P(X) - la distribución de los datos. Puede generar nuevos ejemplos. GPT es generativo.

**Gradient (Gradiente)**
: Vector de derivadas parciales que indica la dirección de máximo aumento de una función. El entrenamiento mueve pesos en dirección opuesta al gradiente.

**Gradient Descent (Descenso de Gradiente)**
: Algoritmo de optimización que actualiza pesos en la dirección opuesta al gradiente para minimizar la pérdida.

**Greedy Decoding**
: Estrategia que siempre elige el token más probable. Rápido pero puede quedar atrapado en secuencias subóptimas.

---

## H

**Hallucination (Alucinación)**
: Generación de información falsa o inventada por un LLM. Ocurre cuando el modelo extrapola más allá de sus datos de entrenamiento.

**Hidden State (Estado Oculto)**
: Representación interna de una red en una capa intermedia. En RNNs, el estado oculto acumula información de tokens anteriores.

---

## L

**Layer Normalization**
: Técnica que normaliza activaciones dentro de cada capa (media≈0, varianza≈1). Estabiliza el entrenamiento de Transformers.

**Learning Rate (Tasa de Aprendizaje)**
: Hiperparámetro que controla el tamaño de los pasos de actualización de pesos. Muy alto causa divergencia; muy bajo causa convergencia lenta.

**Logits**
: Salida cruda del modelo antes de softmax. Valores sin normalizar que pueden ser positivos, negativos o mayores que 1.

**LoRA (Low-Rank Adaptation)**
: Técnica de fine-tuning eficiente que actualiza solo matrices de bajo rango (~1-2% de parámetros). 50x más eficiente que full fine-tuning.

**Loss Function (Función de Pérdida)**
: Métrica que cuantifica el error del modelo. El entrenamiento minimiza esta función. Ejemplos: cross-entropy, MSE.

**LSTM (Long Short-Term Memory)**
: Tipo de RNN con "puertas" que controlan el flujo de información. Mitiga el problema de vanishing gradients para secuencias largas.

---

## M

**Masked Language Modeling (MLM)**
: Objetivo de pre-entrenamiento de BERT. Enmascara tokens aleatorios y entrena al modelo a predecirlos usando contexto bidireccional.

**Multi-Head Attention**
: Atención con múltiples "cabezas" en paralelo, cada una capturando diferentes tipos de relaciones (sintácticas, semánticas, posicionales).

---

## O

**Optimizer (Optimizador)**
: Algoritmo que actualiza pesos basándose en gradientes. Adam es el más común; adapta tasas de aprendizaje por parámetro.

**Overfitting (Sobreajuste)**
: Cuando el modelo memoriza datos de entrenamiento pero no generaliza a datos nuevos. Indicado por bajo training loss pero alto validation loss.

---

## P

**Perplexity (Perplejidad)**
: Métrica de evaluación para modelos de lenguaje: `exp(cross_entropy)`. Menor perplexity = mejor modelo. Interpretable como "número de opciones igualmente probables".

**Positional Encoding (Codificación Posicional)**
: Señal añadida a embeddings para informar al modelo sobre la posición de cada token. Sin ella, el orden no importaría para la atención.

**Pre-training**
: Entrenamiento inicial en un corpus masivo de texto general (ej: internet). Produce un modelo base que luego se fine-tunea.

---

## Q

**QLoRA**
: LoRA combinado con cuantización INT4. Permite fine-tuning de modelos grandes en GPUs de consumidor (2GB de memoria).

**Quantization (Cuantización)**
: Reducir precisión de pesos (ej: float32 → int8). Reduce memoria y aumenta velocidad con pérdida mínima de calidad.

---

## R

**ReLU (Rectified Linear Unit)**
: Función de activación: `max(0, x)`. Simple, eficiente, y evita el problema de saturación de sigmoid/tanh.

**Residual Connection (Conexión Residual)**
: Añadir la entrada de una capa a su salida: `output = x + sublayer(x)`. Permite entrenar redes muy profundas.

**RNN (Recurrent Neural Network)**
: Red que procesa secuencias manteniendo un estado oculto. Procesa token por token secuencialmente.

---

## S

**Sampling**
: Proceso de seleccionar el siguiente token basándose en la distribución de probabilidad. Estrategias: greedy, top-K, top-P, beam search.

**Self-Attention (Auto-Atención)**
: Atención donde Query, Key y Value vienen de la misma secuencia. Cada token atiende a todos los demás tokens de la misma secuencia.

**Softmax**
: Función que convierte logits a probabilidades: `softmax(x_i) = exp(x_i) / Σexp(x_j)`. La salida suma 1.

**Supervised Learning (Aprendizaje Supervisado)**
: Paradigma donde el modelo aprende de datos etiquetados (pares input-output).

---

## T

**Temperature (Temperatura)**
: Hiperparámetro que controla la "agudeza" de la distribución softmax. T<1 más determinístico; T>1 más aleatorio.

**Token**
: Unidad básica de texto para un LLM. Puede ser una palabra, sub-palabra, o carácter dependiendo del tokenizador.

**Tokenizer (Tokenizador)**
: Componente que convierte texto a secuencia de tokens (IDs numéricos). BPE, WordPiece, SentencePiece son algoritmos comunes.

**Top-K Sampling**
: Estrategia que muestrea solo de los K tokens más probables. Filtra tokens improbables.

**Top-P (Nucleus) Sampling**
: Estrategia que muestrea del conjunto mínimo de tokens cuya probabilidad acumulada supera P. Más adaptativo que Top-K.

**Transformer**
: Arquitectura de red neuronal basada en atención, sin recurrencia. Base de todos los LLMs modernos (GPT, BERT, Claude).

---

## U

**Underfitting (Subajuste)**
: Cuando el modelo es demasiado simple para capturar patrones en los datos. Alto training loss y alto validation loss.

**Unsupervised Learning (Aprendizaje No Supervisado)**
: Paradigma donde el modelo encuentra estructura en datos sin etiquetas. Clustering, reducción de dimensionalidad.

---

## V

**Vanishing Gradient (Gradiente Desvaneciente)**
: Problema donde gradientes se vuelven muy pequeños en capas profundas, impidiendo el aprendizaje. Afecta RNNs con secuencias largas.

**Vocabulary (Vocabulario)**
: Conjunto de todos los tokens únicos que el modelo conoce. Típicamente 30K-100K tokens para LLMs.

---

## W

**Weight (Peso)**
: Parámetro aprendible de una red neuronal. Las conexiones entre neuronas tienen pesos que determinan la importancia de cada input.

**Word2Vec**
: Técnica clásica de embeddings que aprende representaciones de palabras basándose en co-ocurrencia. Predecesor de embeddings contextuales.

---

## X

**XGrammar**
: Framework para constrained decoding usando gramáticas formales (CFG). Garantiza salidas sintácticamente válidas (JSON, código).

---

*Última actualización: Marzo 2026*
