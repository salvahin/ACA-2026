# Módulo de Compiladores

> **Curso:** TC3002B - Grammar-Constrained GPU Kernel Generation
> **Semanas:** 10

---

## Overview

Este módulo cubre la teoría de compiladores necesaria para entender cómo las gramáticas pueden restringir la generación de código. Desde autómatas finitos hasta parser generators, proporciona las bases formales del proyecto.

---

## Estructura del Módulo

### Semana 1: Introducción a Compiladores
**Archivo:** `01-introduccion-compiladores.md`

- ¿Qué es un compilador?
- Fases de compilación
- Analogía: la cocina
- Frontend vs backend

---

### Semana 2: Lenguajes Regulares y DFAs
**Archivo:** `02-lenguajes-regulares-dfas.md`

- Expresiones regulares
- Autómatas finitos deterministas (DFA)
- Reconocimiento de patrones
- Implementación práctica

---

### Semana 3: NFAs y Conversión a DFA
**Archivo:** `03-nfas-conversion-dfa.md`

- Autómatas no deterministas (NFA)
- Algoritmo de subconjuntos
- Equivalencia NFA-DFA
- Trade-offs de implementación

---

### Semana 4: Gramáticas Libres de Contexto
**Archivo:** `04-gramaticas-libres-contexto.md`

- Definición formal de CFG
- Derivaciones y árboles de parseo
- Ambigüedad
- Jerarquía de Chomsky

---

### Semana 5: BNF, EBNF y Parsing
**Archivo:** `05-bnf-ebnf-parsing.md`

- Notación BNF y EBNF
- Parsing top-down vs bottom-up
- LL y LR parsers
- Recursive descent

---

### Semana 6: Pipeline FSM de XGrammar
**Archivo:** `06-pipeline-fsm-xgrammar.md`

- Compilación de gramáticas a FSM
- Pipeline de XGrammar
- Optimizaciones
- Integración con tokenizers

---

### Semana 7: Diseño de Gramáticas DSL
**Archivo:** `07-diseno-gramaticas-dsl.md`

- Principios de diseño de DSLs
- Gramáticas para lenguajes específicos
- Usabilidad vs expresividad
- Casos de estudio

---

### Semana 8: Compilación de Gramáticas
**Archivo:** `08-compilacion-gramaticas.md`

- De especificación a parser ejecutable
- Generación de código
- Optimización de gramáticas
- Testing de parsers

---

### Semana 9: Limitaciones Context-Sensitive
**Archivo:** `09-limitaciones-context-sensitive.md`

- Qué no pueden expresar las CFG
- Gramáticas sensibles al contexto
- Workarounds prácticos
- Semantic actions

---

### Semana 10: Parser Generators y Reflexión
**Archivo:** `10-parser-generators-reflexion.md`

- ANTLR, Bison, PLY
- Comparativa de herramientas
- Cuándo usar cada una
- El ecosistema real

---

## Progresión Pedagógica

```
Semanas 1-3: Autómatas
├─ Introducción
├─ DFAs
└─ NFAs

Semanas 4-5: Gramáticas
├─ CFG
└─ BNF/EBNF

Semanas 6-8: XGrammar
├─ Pipeline FSM
├─ Diseño DSL
└─ Compilación

Semanas 9-10: Avanzado
├─ Limitaciones
└─ Herramientas
```

---

## Relación con Otros Módulos

- **AI:** Constrained decoding usa gramáticas
- **Project 1:** Implementación de gramáticas Triton
- **Project 2:** Restricciones GPU en gramáticas

---

**Última actualización:** Marzo 2026
**Versión:** 1.0
**Idioma:** Español
