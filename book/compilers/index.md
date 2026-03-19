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

# Módulo de Compiladores

Este módulo cubre la teoría de compiladores necesaria para entender cómo las gramáticas pueden restringir la generación de código. Desde autómatas finitos hasta parser generators, aprenderás las bases formales que hacen posible el constrained decoding en LLMs.

## Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:

- [ ] Diseñar autómatas finitos (DFA/NFA) para reconocer patrones en texto
- [ ] Escribir gramáticas libres de contexto (CFG) en notación BNF/EBNF
- [ ] Entender el pipeline de compilación de gramáticas a FSM en XGrammar
- [ ] Diseñar gramáticas para DSLs (Domain-Specific Languages)
- [ ] Identificar las limitaciones de las gramáticas libres de contexto
- [ ] Comparar y seleccionar parser generators apropiados para cada caso

## Prerequisitos

- Programación en Python
- Matemáticas discretas básicas (conjuntos, relaciones)
- Familiaridad con expresiones regulares

## Temas por Semana

| Semana | Tema | Descripción |
|:------:|------|-------------|
| 1 | [Introducción a Compiladores](01-introduccion-compiladores) | Fases de compilación, frontend vs backend |
| 2 | [Lenguajes Regulares y DFAs](02-lenguajes-regulares-dfas) | Expresiones regulares, autómatas deterministas |
| 3 | [NFAs y Conversión a DFA](03-nfas-conversion-dfa) | Autómatas no deterministas, algoritmo de subconjuntos |
| 4 | [Gramáticas Libres de Contexto](04-gramaticas-libres-contexto) | CFG, derivaciones, árboles de parseo, jerarquía de Chomsky |
| 5 | [BNF, EBNF y Parsing](05-bnf-ebnf-parsing) | Notaciones, LL/LR parsers, recursive descent |
| 6 | [Pipeline FSM de XGrammar](06-pipeline-fsm-xgrammar) | Compilación de gramáticas, integración con tokenizers |
| 7 | [Diseño de Gramáticas DSL](07-diseno-gramaticas-dsl) | Principios de diseño, usabilidad vs expresividad |
| 8 | [Compilación de Gramáticas](08-compilacion-gramaticas) | Generación de código, optimización, testing |
| 9 | [Limitaciones Context-Sensitive](09-limitaciones-context-sensitive) | Qué no pueden expresar las CFG, workarounds |
| 10 | [Parser Generators](10-parser-generators-reflexion) | ANTLR, Bison, PLY, comparativa |

## Relación con Otros Módulos

```{admonition} Conexiones
:class: tip

- **→ AI:** Las gramáticas se usan para constrained decoding en LLMs
- **→ Project 1:** Implementación de gramáticas Triton para generación de kernels
- **→ Project 2:** Restricciones de sintaxis GPU en gramáticas
```
