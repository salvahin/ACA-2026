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

# Project 1 - Grammar-Constrained LLM Code Generation

> **Módulo:** ACA - Grammar-Constrained GPU Kernel Generation
> **Perfil del Profesor:** LLM Researcher (experto en generación de código con LLMs)
> **Semanas:** 10

---

```{admonition} 🚀 Primer Paso Obligatorio
:class: important
Antes de comenzar cualquier actividad de este módulo, ejecuta el notebook de verificación de entorno:
**[00_verificacion_entorno.ipynb](../notebooks/00_verificacion_entorno.ipynb)** — Verifica dependencias, GPU disponible y modo CPU Fallback.
```

## Overview

Este módulo cubre la generación automática de kernels Triton usando gramáticas JSON Schema y modelos de lenguaje. El enfoque está en constrained decoding, diseño de gramáticas, y sistemas agénticos.

**Total de contenido:** 10 lecturas
**Tiempo estimado:** 15-20 horas de lectura
**Nivel:** Intermedio-Avanzado

---

## Estructura del Módulo

### Semana 1: Setup e Introducción a LLMs
**Archivo:** `01_Setup_Intro_LLMs.md`

- Configuración del entorno de desarrollo
- Introducción a LLMs para generación de código
- Clonación de repos, venv/conda, dependencias

---

### Semana 2: LLMs para Código
**Archivo:** `02_LLMs_para_Codigo.md`

- Modelos especializados: CodeLlama, StarCoder, DeepSeek-Coder
- Distribución de entrenamiento código vs texto
- Fill-in-the-Middle (FIM)
- Trade-offs calidad vs restricciones

---

### Semana 3: Prompt Engineering
**Archivo:** `03_Prompt_Engineering.md`

- Técnicas de prompting para código
- Few-shot examples
- Chain-of-thought para programación
- Estructuración de prompts efectivos

---

### Semana 4: XGrammar y Constrained Decoding
**Archivo:** `04_XGrammar_Constrained.md`

- Pipeline de compilación de XGrammar
- Gramática → Parser → AST → Compiler → FSM
- TokenizerInfo y LogitsProcessor
- Constrained decoding en práctica

---

### Semana 5: Gramática JSON
**Archivo:** `05_JSON_Grammar.md`

- Diseño de JSON Schema para kernels
- Compilación con XGrammar
- Tests positivos y negativos
- Validación sistemática

---

### Semana 6: Gramática Triton L1-L2
**Archivo:** `06_Gramatica_Triton_L1L2.md`

- L1: Imports obligatorios (triton, triton.language)
- L1: Decorador @triton.jit
- L2: Firma de función (parámetros, tipos)
- Especificación EBNF formal

---

### Semana 7: Gramática Triton L3-L4
**Archivo:** `07_Gramatica_Triton_L3L4.md`

- L3: Expresiones (aritmética, booleanas, indexación)
- L3: Llamadas a API (tl.load, tl.store, etc.)
- L4: Control flow (if, for, while)
- Generador recursivo de código

---

### Semana 8: Serving LLMs
**Archivo:** `08_Serving_LLMs.md`

- Infraestructura de serving (vLLM, TGI)
- Batching y optimización
- Latency vs throughput
- Deployment en producción

---

### Semana 9: Token Economics e Integración
**Archivo:** `09_Token_Economics_Integracion.md`

- Costos de tokens y optimización
- Integración gramática + modelo
- Pipeline end-to-end
- Validación de outputs

---

### Semana 10: Sistemas Agénticos
**Archivo:** `10_Agentic_Systems.md`

- Tool use y function calling
- Arquitecturas multi-agente
- RAG (Retrieval-Augmented Generation)
- Reasoning y planificación

---

## Estructura de Archivos

```
lecturas/Project_1/
├── 01_Setup_Intro_LLMs.md          # Setup + Intro LLMs
├── 02_LLMs_para_Codigo.md          # Modelos de código
├── 03_Prompt_Engineering.md        # Prompting
├── 04_XGrammar_Constrained.md      # XGrammar internals
├── 05_JSON_Grammar.md              # JSON Schema
├── 06_Gramatica_Triton_L1L2.md     # Estructura kernels
├── 07_Gramatica_Triton_L3L4.md     # Expresiones
├── 08a_serving_inference.md
├── 08b_serving_optimization.md
├── 09_token_economics_integracion.md # Economics
├── 10_Agentic_Systems.md           # Sistemas agénticos
└── README.md                       # Este archivo
```

---

## Progresión Pedagógica

```
Semanas 1-3: Fundamentos LLM
├─ Setup y entorno
├─ Modelos de código
└─ Prompt engineering

Semanas 4-5: Constrained Decoding
├─ XGrammar internals
└─ JSON Schema

Semanas 6-7: Gramáticas Triton
├─ Estructura (L1-L2)
└─ Expresiones (L3-L4)

Semanas 8-10: Producción
├─ Serving
├─ Economics
└─ Sistemas agénticos
```

---

## Requisitos Previos

- Python 3.9+
- Familiaridad con Git
- Conceptos básicos de ML/LLMs
- Acceso a GPU (recomendado)

---

## Relación con Otros Módulos

- **AI:** Fundamentos teóricos de IA/DL
- **Compilers:** Teoría de gramáticas y parsing
- **Project 2:** Implementación GPU y optimización
- **Stats:** Análisis estadístico de experimentos

---

**Última actualización:** Marzo 2026
**Versión:** 2.0
**Idioma:** Español
