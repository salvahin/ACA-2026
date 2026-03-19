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

Este módulo te proporciona los fundamentos teóricos de Inteligencia Artificial y Deep Learning necesarios para entender cómo funcionan los modelos de lenguaje (LLMs) y la generación de código. Aprenderás desde los conceptos básicos hasta técnicas avanzadas como fine-tuning y MLOps.

## Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:

- [ ] Distinguir entre modelos discriminativos y generativos, y explicar cuándo usar cada uno
- [ ] Implementar y entrenar redes neuronales básicas con backpropagation
- [ ] Explicar la arquitectura Transformer y el mecanismo de atención
- [ ] Aplicar técnicas de sampling y constrained decoding para controlar la generación de texto
- [ ] Realizar fine-tuning eficiente con LoRA y evaluar modelos con métricas apropiadas
- [ ] Implementar pipelines de MLOps para tracking de experimentos

## Prerequisitos

- Programación en Python (intermedio)
- Álgebra lineal básica (vectores, matrices)
- Cálculo (derivadas, regla de la cadena)
- Probabilidad y estadística básica

## Temas por Semana

| Semana | Tema | Descripción |
|:------:|------|-------------|
| 1 | [IA Clásica vs Generativa](01_ia_clasica_vs_generativa) | Historia de la IA, paradigmas de aprendizaje, modelos discriminativos vs generativos |
| 2 | [Fundamentos Deep Learning](02_fundamentos_deep_learning) | Redes neuronales, backpropagation, funciones de pérdida, optimizadores |
| 3 | [Generación Autoregresiva](03_generacion_autoregresiva) | NLP básico, embeddings, RNNs/LSTMs, introducción a generación de texto |
| 4 | [Arquitectura Transformer](04_arquitectura_transformer) | Atención, multi-head attention, positional encoding, encoder vs decoder |
| 5 | [BERT y GPTs](05_bert_gpts_tokenizacion) | Modelos pre-entrenados, tokenización (BPE), scaling laws |
| 6 | [Sampling y Constrained Decoding](06_sampling_constrained_decoding) | Estrategias de muestreo, temperature, XGrammar |
| 7 | [Fine-Tuning y Evaluación](07_fine_tuning_evaluacion) | LoRA/QLoRA, métricas de evaluación, LLM-as-Judge |
| 8 | [MLOps](08_mlops_visualizacion) | MLflow, tracking de experimentos, reproducibilidad |

## Relación con Otros Módulos

```{admonition} Conexiones
:class: tip

- **→ Project 1:** Aplica estos fundamentos para generar código con LLMs
- **→ Compilers:** Las gramáticas de este módulo se usan para constrained decoding
- **→ Stats:** Evaluación estadística de los modelos que desarrolles
- **→ Project 2:** GPU computing para inferencia eficiente
```

## Recursos Adicionales

- [Glosario Técnico](glossary) — Definiciones de términos clave
- [Solucionario](solutions) — Respuestas a ejercicios
