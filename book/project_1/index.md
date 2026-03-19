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

# Project 1: Grammar-Constrained LLM Code Generation

En este proyecto aprenderás a generar kernels Triton automáticamente usando LLMs con gramáticas que restringen la salida. Combinarás técnicas de prompt engineering, constrained decoding con XGrammar, y sistemas agénticos para crear un pipeline de generación de código GPU.

```{admonition} Primer Paso
:class: important
Antes de comenzar, ejecuta el notebook de verificación: **[00_verificacion_entorno.ipynb](../notebooks/00_verificacion_entorno.ipynb)**
```

## Objetivos de Aprendizaje

Al completar este proyecto, serás capaz de:

- [ ] Seleccionar y configurar LLMs especializados en código (CodeLlama, DeepSeek-Coder)
- [ ] Diseñar prompts efectivos para generación de código con técnicas de few-shot y chain-of-thought
- [ ] Implementar constrained decoding usando XGrammar y JSON Schema
- [ ] Crear gramáticas EBNF para la sintaxis de Triton (niveles L1-L4)
- [ ] Desplegar LLMs con vLLM/TGI optimizando latencia y throughput
- [ ] Construir sistemas agénticos con tool use y feedback loops

## Prerequisitos

- Python 3.9+
- Conceptos básicos de ML/LLMs (Módulo AI)
- Familiaridad con gramáticas (Módulo Compilers)
- Acceso a GPU (recomendado)

## Temas por Semana

| Semana | Tema | Descripción |
|:------:|------|-------------|
| 1 | [Setup e Intro LLMs](01_setup_intro_llms) | Configuración del entorno, introducción a LLMs para código |
| 2 | [LLMs para Código](02_llms_para_codigo) | CodeLlama, StarCoder, DeepSeek-Coder, Fill-in-the-Middle |
| 3 | [Prompt Engineering](03_prompt_engineering) | Few-shot, chain-of-thought, estructuración de prompts |
| 4 | [XGrammar y Constrained Decoding](04_xgrammar_constrained) | Pipeline de compilación, LogitsProcessor |
| 5 | [Gramática JSON](05_json_grammar) | JSON Schema para kernels, validación |
| 6 | [Gramática Triton L1-L2](06_gramatica_triton_l1l2) | Imports, decoradores, firma de funciones |
| 7 | [Gramática Triton L3-L4](07_gramatica_triton_l3l4) | Expresiones, llamadas API, control flow |
| 8 | [Serving LLMs](08a_serving_inference) | vLLM, TGI, batching, optimización |
| 9 | [Token Economics](09_token_economics_integracion) | Costos, integración gramática + modelo |
| 10 | [Sistemas Agénticos](10_agentic_systems) | Tool use, multi-agente, RAG |

## Relación con Otros Módulos

```{admonition} Conexiones
:class: tip

- **← AI:** Fundamentos teóricos de IA/DL que aplicas aquí
- **← Compilers:** Teoría de gramáticas y parsing para XGrammar
- **→ Project 2:** Los kernels generados aquí se optimizan en Project 2
- **→ Stats:** Análisis estadístico de experimentos de generación
```
