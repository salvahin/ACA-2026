# Módulo de Inteligencia Artificial

> **Curso:** TC3002B - Grammar-Constrained GPU Kernel Generation
> **Perfil del Profesor:** Experto en IA Clásica
> **Semanas:** 8 (+ 2 semanas para experimentos/final)

---

## Overview

Este módulo proporciona los fundamentos teóricos de Inteligencia Artificial y Deep Learning necesarios para entender la generación de código con LLMs. Está diseñado para profesores con experiencia en IA clásica.

**Nota:** El contenido específico de LLMs para código, serving, y sistemas agénticos se encuentra en el módulo Project 1.

---

## Estructura del Módulo

### Semana 1: IA Clásica vs Generativa
**Archivo:** `01_IA_Clasica_vs_Generativa.md`

- ML clásico: supervisado, no supervisado, refuerzo
- Regresión Lineal, Regresión logística
- Comparación ML vs DL vs LLMs
- Casos de uso para cada paradigma

---

### Semana 2: Fundamentos de Deep Learning
**Archivo:** `02_Fundamentos_Deep_Learning.md`

- Del perceptrón a Deep Learning
- Backpropagation
- Entrenamiento e hiperparámetros
- Funciones de activación y optimizadores

---

### Semana 3: Generación Autoregresiva
**Archivo:** `03_Generacion_Autoregresiva.md`

- Auto-encoders y redes recurrentes
- Generación autoregresiva
- IA Generativa vs Discriminativa
- Introducción a Transformers

---

### Semana 4: Arquitectura Transformer
**Archivo:** `04_Arquitectura_Transformer.md`

- Multi-head attention
- Positional encoding
- Layer normalization
- Encoder vs Decoder
- "Attention Is All You Need"

---

### Semana 5: BERT, GPTs y Tokenización
**Archivo:** `05_BERT_GPTs_Tokenizacion.md`

- BERT: comprensión bidireccional
- GPT: generación a escala
- Logits → softmax → probabilidades
- Cross-entropy loss
- Tokenización: BPE, SentencePiece, tiktoken

---

### Semana 6: Sampling y Constrained Decoding
**Archivo:** `06_Sampling_Constrained_Decoding.md`

- Estrategias de muestreo (greedy, top-k, top-p)
- Temperature y su efecto
- Constrained decoding
- Introducción a XGrammar

---

### Semana 7: Fine-Tuning y Evaluación
**Archivo:** `07_Fine_Tuning_Evaluacion.md`

- Fine-tuning vs prompting
- LoRA/QLoRA: cómo funcionan
- Instruction tuning
- Métricas de evaluación
- LLM-as-Judge

---

### Semana 8: MLOps y Visualización
**Archivo:** `08_MLOps_Visualizacion.md`

- Logging de hiperparámetros y métricas
- Weights & Biases / MLflow
- Versionado de experimentos
- Visualización de resultados
- Reportes automatizados

---

### Semanas 9-10: Experimentos y Final
*(Sin lectura asignada - tiempo para proyecto)*

---

## Estructura de Archivos

```
lecturas/AI/
├── 01_IA_Clasica_vs_Generativa.md     # ML clásico vs LLMs
├── 02_Fundamentos_Deep_Learning.md    # Perceptrón, backprop
├── 03_Generacion_Autoregresiva.md     # RNNs, autoregresivo
├── 04_Arquitectura_Transformer.md     # Attention, encoder/decoder
├── 05_BERT_GPTs_Tokenizacion.md       # BERT, GPT, BPE
├── 06_Sampling_Constrained_Decoding.md # Sampling, XGrammar
├── 07_Fine_Tuning_Evaluacion.md       # LoRA, evaluación
├── 08_MLOps_Visualizacion.md          # Tracking, visualización
└── README.md                          # Este archivo
```

---

## Progresión Pedagógica

```
Semanas 1-2: Fundamentos
├─ ML clásico
└─ Deep Learning

Semanas 3-4: Transformers
├─ Generación autoregresiva
└─ Arquitectura detallada

Semanas 5-6: Modelos y Decoding
├─ BERT/GPT/Tokenización
└─ Sampling y constraints

Semanas 7-8: Práctica
├─ Fine-tuning
└─ MLOps

Semanas 9-10: Proyecto
├─ Experimentos
└─ Presentación final
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
