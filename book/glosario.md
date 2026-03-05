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

# Glosario del Curso ACA-2026

> Referencia rápida de los términos técnicos usados a lo largo del curso. Los términos están organizados por módulo. Haz clic en las referencias cruzadas para ir a la lección correspondiente.

---

## Módulo AI

**Atención (Attention)** — Mecanismo que permite a un modelo aprender cuánta importancia asignar a cada parte de la secuencia de entrada al generar una salida. `score(Q, K) = softmax(QKᵀ / √d_k) · V`. Ver [AI L04](ai/04_arquitectura_transformer.md).

**Atención Causal (Causal Attention)** — Variante de la atención donde cada token solo puede atender a tokens anteriores (no futuros). Se implementa con una máscara triangular superior `-inf`. Usada en GPT. Ver [AI L04](ai/04_arquitectura_transformer.md#atención-causal-gpt-style-vs-bidireccional-bert-style).

**Atención Multi-Cabeza (Multi-Head Attention)** — Aplicación paralela de múltiples mecanismos de atención, cada uno aprendiendo diferentes tipos de relaciones entre tokens. Ver [AI L04](ai/04_arquitectura_transformer.md).

**Backpropagation** — Algoritmo de entrenamiento que usa la regla de la cadena para calcular el gradiente de la pérdida respecto a cada peso de la red. Ver [AI L02](ai/02_fundamentos_deep_learning.md#parte-5-retropropagación-backpropagation).

**BPE (Byte Pair Encoding)** — Algoritmo de tokenización que combina iterativamente los pares de caracteres más frecuentes. Usado en GPT-2, CodeX. Ver [AI L03](ai/03_generacion_autoregresiva.md).

**Cross-Entropy Loss** — Función de pérdida para clasificación: `-Σ y_i · log(p_i)`. Penaliza fuertemente las predicciones incorrectas confiadas. Ver [AI L02](ai/02_fundamentos_deep_learning.md) y [AI L03](ai/03_generacion_autoregresiva.md).

**Embedding** — Representación vectorial densa de un token en un espacio de alta dimensión. Tokens similares tienen embeddings próximos. Ver [AI L04](ai/04_arquitectura_transformer.md).

**Generación Autoregresiva** — Estrategia de generación de texto donde el modelo produce un token a la vez, usando los tokens anteriores como contexto. Ver [AI L03](ai/03_generacion_autoregresiva.md).

**Layer Normalization** — Técnica de normalización que estabiliza el entrenamiento de redes profundas normalizando activaciones por cada muestra (no por batch). Ver [AI L04](ai/04_arquitectura_transformer.md).

**Logits** — Salida cruda (sin normalizar) de la última capa del modelo antes de aplicar softmax. Ver [AI L05](ai/05_bert_gpts_tokenizacion.md).

**Perplexidad (Perplexity)** — Métrica de evaluación de LLMs: `exp(cross-entropy)`. Menor perplexidad = mejor modelo. Un modelo perfecto tiene perplexity=1.

**Residual Connection** — Conexión que suma la entrada de una subcapa a su salida: `x + Sublayer(x)`. Permite entrenar redes muy profundas. Ver [AI L04](ai/04_arquitectura_transformer.md).

**Softmax** — Función que transforma un vector de logits en una distribución de probabilidades: `softmax(z_i) = exp(z_i) / Σ exp(z_j)`. Ver [AI L03](ai/03_generacion_autoregresiva.md).

**Temperatura (Temperature)** — Parámetro que controla la aleatoriedad de la generación: `probs = softmax(logits / T)`. T<1 = más determinista; T>1 = más creativo. Ver [AI L03](ai/03_generacion_autoregresiva.md).

**Token** — Unidad básica de texto para un LLM. Puede ser una palabra completa, parte de una palabra, o un carácter. Ver [AI L03](ai/03_generacion_autoregresiva.md) y [AI L05](ai/05_bert_gpts_tokenizacion.md).

**Transformer** — Arquitectura de red neuronal basada en atención, introducida en el paper "Attention is All You Need" (2017). Base de BERT, GPT y todos los LLMs modernos. Ver [AI L04](ai/04_arquitectura_transformer.md).

**WordPiece** — Algoritmo de tokenización que maximiza la probabilidad del corpus al fusionar pares; usado en BERT. Se diferencia de BPE en el criterio de selección del par. Ver [AI L05](ai/05_bert_gpts_tokenizacion.md).

---

## Módulo Compilers

**AST (Abstract Syntax Tree)** — Árbol que representa la estructura gramatical de un programa sin detalles de sintaxis concreta. Resultado del parser. Ver [Compilers L01](compilers/01-introduccion-compiladores.md).

**CFG (Context-Free Grammar)** — Gramática de Chomsky Tipo 2 donde cada producción tiene la forma `A → α`. Base de los lenguajes de programación y de XGrammar. Ver [Compilers L01](compilers/01-introduccion-compiladores.md).

**Constrained Decoding** — Técnica que restringe las probabilidades de siguiente token del LLM para garantizar que la salida sea válida según una gramática formal. Ver [Project 1 L04](project_1/04_xgrammar_constrained.md).

**DFA (Deterministic Finite Automaton)** — Autómata de estados finitos donde cada estado tiene exactamente una transición por símbolo. Reconoce lenguajes regulares (Tipo 3).

**EBNF (Extended Backus-Naur Form)** — Notación extendida de BNF para describir gramáticas, con operadores `?`, `*`, `+`, `|`. Usada para definir la gramática de kernels Triton.

**Jerarquía de Chomsky** — Clasificación de gramáticas en 4 tipos: Tipo 0 (irrestrictas), Tipo 1 (sensibles al contexto), Tipo 2 (libres de contexto), Tipo 3 (regulares). Ver [Compilers L01](compilers/01-introduccion-compiladores.md).

**Lexer / Tokenizer** — Primera fase del compilador: convierte texto en una secuencia de tokens léxicos (identificadores, literales, operadores).

**Parser** — Segunda fase del compilador: construye el AST a partir de la secuencia de tokens, verificando que sigan la gramática.

**XGrammar** — Librería de constrained decoding que compila gramáticas CFG en estructuras eficientes (FSM) para filtrar tokens durante la generación del LLM. Ver [Project 1 L04](project_1/04_xgrammar_constrained.md).

---

## Módulo Stats

**Cohen's d** — Medida de tamaño de efecto para comparar dos medias: `d = (μ₁ - μ₂) / σ_pooled`. Valores: pequeño=0.2, mediano=0.5, grande=0.8. Ver [Stats L08](stats/).

**Intervalo de Confianza (CI)** — Rango de valores que, con una cierta probabilidad (ej. 95%), contienen el parámetro poblacional verdadero.

**ANOVA** — Prueba estadística para comparar medias de 3+ grupos. Asume normalidad y homogeneidad de varianzas. Ver [Stats L06](stats/).

**Kruskal-Wallis** — Alternativa no paramétrica a ANOVA para 3+ grupos que no asumen distribución normal. Ver [Stats L06](stats/) y [uso-en-el-proyecto.md](stats/uso-en-el-proyecto.md).

**Prueba t (t-test)** — Prueba estadística para comparar dos medias: independiente (dos grupos distintos) o pareada (mismos sujetos en dos condiciones). Ver [Stats L03](stats/).

**p-valor** — Probabilidad de observar un resultado igual o más extremo que el actual, asumiendo que H₀ es verdadera. p < α (típico 0.05) → rechazar H₀.

**Shapiro-Wilk** — Prueba de normalidad (H₀: los datos vienen de una distribución normal). Usar antes de decidir entre t-test y Wilcoxon.

**Wilcoxon** — Prueba no paramétrica equivalente al t-test para datos no normales. Ver [uso-en-el-proyecto.md](stats/uso-en-el-proyecto.md).

---

## Módulo Project_1 / GPU

**Block Size (BLOCK_SIZE)** — Número de elementos procesados por cada thread block de Triton. Constante en tiempo de compilación (`tl.constexpr`).

**Grid** — Colección de thread blocks que ejecutan un kernel GPU en paralelo. Dimensiones definidas por el programador.

**Kernel GPU** — Función ejecutada en la GPU, con miles de threads en paralelo. En Triton: función decorada con `@triton.jit`.

**Program ID (pid)** — Identificador único de cada thread block en Triton: `pid = tl.program_id(axis=0)`.

**Triton** — Lenguaje de programación GPU de alto nivel (sobre CUDA) creado por OpenAI. Permite escribir kernels eficientes en Python. Ver [Project 1](project_1/index.md).

**tl.load / tl.store** — Operaciones fundamentales de Triton para leer/escribir memoria GPU con una mask de validez: `tl.load(ptr + offsets, mask=mask)`.

---

## Referencias Cruzadas Rápidas

| Concepto | Módulo | Lección |
|----------|--------|---------|
| Softmax + Temperature | AI | L03, L05 |
| Causal Masking | AI | L04 |
| Beam Search | AI | L03 |
| BERT vs GPT | AI | L05 |
| CFG + Jerarquía Chomsky | Compilers | L01 |
| XGrammar internals | Project 1 | L04 |
| t-test + Wilcoxon | Stats | L03 |
| ANOVA + Kruskal | Stats | L06 |
| Estadística → Proyecto | Stats | [uso-en-el-proyecto.md](stats/uso-en-el-proyecto.md) |

---

*Última actualización: Marzo 2026 — ACA: Grammar-Constrained GPU Kernel Generation*
