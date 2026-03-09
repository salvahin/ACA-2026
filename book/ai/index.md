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

# Módulo de Inteligencia Artificial

> **Curso:** ACA - Grammar-Constrained GPU Kernel Generation
> **Perfil del Profesor:** Experto en IA Clásica
> **Semanas:** 8 (+ 2 semanas para experimentos/final)

---

## Overview

Este módulo proporciona los fundamentos teóricos de Inteligencia Artificial y Deep Learning necesarios para entender la generación de código con LLMs. Está diseñado para profesores con experiencia en IA clásica.

**Nota:** El contenido específico de LLMs para código, serving, y sistemas agénticos se encuentra en el módulo Project 1.

---

## Mapa Conceptual del Módulo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MÓDULO DE IA                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  SEMANA 1   │───▶│  SEMANA 2   │───▶│  SEMANA 3   │───▶│  SEMANA 4   │   │
│  │  Historia   │    │  Deep       │    │  NLP &      │    │  Arquitect. │   │
│  │  IA Clásica │    │  Learning   │    │  Generación │    │  Transformer│   │
│  │  vs Gener.  │    │  Fundament. │    │  Autoregr.  │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        │                  │                  │                  │            │
│        ▼                  ▼                  ▼                  ▼            │
│   Paradigmas         Backprop           Softmax            Attention        │
│   Supervisado        Tensores           BPE/Tokens         Multi-head       │
│   No Supervisado     Optimizers         Sampling           Pos. Encoding    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  SEMANA 5   │───▶│  SEMANA 6   │───▶│  SEMANA 7   │───▶│  SEMANA 8   │   │
│  │  BERT &     │    │  Sampling & │    │  Fine-Tune  │    │  MLOps &    │   │
│  │  GPTs       │    │  Constrain. │    │  Evaluación │    │  Producción │   │
│  │             │    │  Decoding   │    │             │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        │                  │                  │                  │            │
│        ▼                  ▼                  ▼                  ▼            │
│   Encoder/Decoder    XGrammar           LoRA/QLoRA         MLflow           │
│   Tokenización       Beam Search        Pass@k             Reproducib.      │
│   Alucinaciones      Gramáticas         Perplexity         Monitoreo        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

PRERREQUISITOS:
  Semana 1 → ninguno (punto de entrada)
  Semana 2 → Semana 1
  Semana 3 → Semanas 1-2
  Semana 4 → Semanas 1-3
  Semanas 5-8 → Semanas 1-4 (base completa)
```

---

## Estructura del Módulo

### Semana 1: Introducción al Aprendizaje Automático
**Archivo:** `01_ia_clasica_vs_generativa.md`

- Historia de la IA: Taller de Dartmouth (1956)
- Paradigmas: supervisado, no supervisado, refuerzo
- Clasificación vs Regresión
- Regresión lineal y logística
- Autoencoders y VAEs
- Comparación modelos discriminativos vs generativos

---

### Semana 2: Fundamentos de Deep Learning
**Archivo:** `02_fundamentos_deep_learning.md`

- Del perceptrón a Deep Learning
- Historia de GPUs y AlexNet
- Operaciones matriciales y producto de Hadamard
- Forward pass y backward pass (regla de la cadena)
- Funciones de pérdida y optimizadores (SGD, Adam)
- Overfitting y regularización

---

### Semana 3: Fundamentos de Procesamiento de Lenguaje Natural
**Archivo:** `03_generacion_autoregresiva.md`

- Preprocesamiento de texto: tokenización, stemming
- N-gramas, one-hot encoding, vocabulario
- Embeddings (Word2Vec) y similitud semántica
- RNNs, LSTMs y el problema del gradient vanishing
- BERT y embeddings contextuales
- Introducción a generación autoregresiva

---

### Semana 4: Arquitectura Transformer
**Archivo:** `04_arquitectura_transformer.md`

- Multi-head attention (y por qué evita el colapso)
- Positional encoding
- Layer normalization y conexiones residuales
- Transformer como autoencoder
- Encoder vs Decoder
- "Attention Is All You Need" + The Illustrated Transformer

---

### Semana 5: GPTs y Modelos Generativos
**Archivo:** `05_bert_gpts_tokenizacion.md`

- Filosofía de escalamiento de GPT
- Construir un mini-GPT desde cero
- Generación autoregresiva y estrategias de muestreo
- El problema de las alucinaciones
- Tokenización avanzada: BPE, SentencePiece, tiktoken

---

### Semana 6: Sampling y Constrained Decoding
**Archivo:** `06_sampling_constrained_decoding.md`

- Estrategias de muestreo (greedy, top-k, top-p)
- Temperature y su efecto
- Constrained decoding
- Introducción a XGrammar

---

### Semana 7: Fine-Tuning y Evaluación
**Archivo:** `07_fine_tuning_evaluacion.md`

- Fine-tuning vs prompting
- LoRA/QLoRA: cómo funcionan
- Instruction tuning
- Métricas de evaluación
- LLM-as-Judge

---

### Semana 8: MLOps
**Archivo:** `08_mlops_visualizacion.md`

- ¿Qué es MLOps? Ciclo de vida y importancia
- MLOps vs LLMOps
- MLflow: tracking de experimentos
- MLflow con Databricks
- Model Registry y versionado
- Checklists de reproducibilidad

---

### Semanas 9-10: Experimentos y Final
*(Sin lectura asignada - tiempo para proyecto)*

---

## Estructura de Archivos

```
book/ai/
├── 01_ia_clasica_vs_generativa.md     # Introducción al ML, historia, clasificación/regresión
├── 02_fundamentos_deep_learning.md    # Perceptrón, backprop, GPUs, optimizadores
├── 03_generacion_autoregresiva.md     # NLP básico, BERT, embeddings, RNNs
├── 04_arquitectura_transformer.md     # Attention, multihead, autoencoder
├── 05_bert_gpts_tokenizacion.md       # GPTs, mini-GPT, alucinaciones
├── 06_sampling_constrained_decoding.md # Sampling, XGrammar
├── 07_fine_tuning_evaluacion.md       # LoRA, evaluación
├── 08_mlops_visualizacion.md          # MLOps, MLflow, Databricks
├── glossary.md                        # Glosario de términos técnicos
├── solutions.md                       # Solucionario de ejercicios
└── index.md                           # Este archivo
```

```{admonition} 📚 Recursos Adicionales
:class: seealso

- **[Glosario Técnico](glossary.md):** Definiciones de términos clave (logits, embeddings, attention, etc.)
- **[Solucionario](solutions.md):** Respuestas detalladas a los ejercicios de cada lectura
```

---

## Progresión Pedagógica

```
Semanas 1-2: Fundamentos de ML
├─ Historia de la IA (Dartmouth 1956)
├─ Clasificación vs Regresión
├─ Regresión lineal y logística
├─ Redes neuronales y backpropagation
└─ GPUs y la revolución del Deep Learning

Semanas 3-4: De NLP a Transformers
├─ Preprocesamiento de texto y n-gramas
├─ Embeddings y Word2Vec
├─ RNNs, LSTMs y gradient vanishing
├─ BERT y embeddings contextuales
├─ Arquitectura Transformer completa
└─ Multi-head attention y por qué funciona

Semanas 5-6: Generación y Control
├─ GPTs y escalamiento
├─ Mini-GPT desde cero
├─ Alucinaciones y mitigación
├─ Estrategias de muestreo
└─ Constrained decoding

Semanas 7-8: Práctica y Producción
├─ Fine-tuning con LoRA
├─ Evaluación de modelos
├─ MLOps y MLflow
└─ Databricks y Model Registry

Semanas 9-10: Proyecto Final
├─ Experimentos
└─ Presentación
```

---

## Relación con Otros Módulos

- **Project 1:** Aplicación práctica de LLMs para código
- **Compilers:** Gramáticas para constrained decoding
- **Stats:** Evaluación estadística de modelos
- **Project 2:** GPU computing para inferencia

---

**Última actualización:** Marzo 2026
**Versión:** 3.0
**Idioma:** Español
