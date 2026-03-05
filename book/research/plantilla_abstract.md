# Plantilla: Abstract / Resumen

> **Instrucciones:** Completa cada sección entre `[corchetes]`. El abstract debe tener entre **150–250 palabras** en inglés (norma ACM/IEEE) y su traducción al español. Borra estas instrucciones antes de entregar.

---

## Abstract (English)

**[Título del paper en inglés]**

[**Context/Motivation sentence** — ¿Por qué es importante este problema? 1–2 oraciones.]

[**Gap sentence** — ¿Qué falta en la literatura actual? "However, existing approaches...".]

[**Contribution sentence** — ¿Qué propones tú? "In this paper, we present/propose/introduce...".]

[**Method sentence** — ¿Cómo lo hiciste? Menciona la técnica principal en 1 oración.]

[**Results sentence** — ¿Cuál es tu resultado más importante? Incluye al menos un número. "Our approach achieves X% improvement over baseline Y on benchmark Z".]

[**Conclusion sentence** — ¿Qué implica para el campo? "These results suggest that..."]

---

## Resumen (Español)

[Traducción del abstract anterior, misma estructura.]

---

## Palabras Clave / Keywords

**Español:** [término 1], [término 2], [término 3], [término 4], [término 5]  
**English:** [keyword 1], [keyword 2], [keyword 3], [keyword 4], [keyword 5]

> **Sugerencias para ACA-2026:** grammar-constrained generation, GPU kernels, Triton, large language models, constrained decoding, XGrammar, JSON Schema

---

## Checklist para el Abstract

- [ ] Tiene entre 150 y 250 palabras (contar con: `wc -w abstract.txt`)
- [ ] Menciona el problema claramente en la primera oración
- [ ] Menciona al menos UN número concreto en los resultados
- [ ] No usa jerga no definida ("our system", no "our novel efficient system")
- [ ] Versión en inglés revisada con herramienta de gramática (Grammarly, LanguageTool)
- [ ] Palabras clave elegidas con criterio (buscar en papers similares qué términos usan)

---

## Ejemplo Completo (ACA-2026)

**Grammar-Constrained Triton Kernel Generation with Large Language Models**

GPU computing demands highly optimized kernel implementations, yet writing correct Triton kernels requires expert knowledge of memory hierarchies and parallelism patterns. Existing code generation approaches lack formal correctness guarantees, producing syntactically invalid outputs up to 40% of the time. In this paper, we present a grammar-constrained decoding framework that enforces Triton kernel structure at generation time using XGrammar's context-free grammar engine. Our approach compiles a formal EBNF grammar of valid Triton programs into a finite-state machine that constrains token probabilities during inference. We evaluate our system on a benchmark of 120 kernel specifications spanning memory-bound and compute-bound patterns. Our method achieves 94.3% syntactic validity and 78.1% functional correctness, improving over unconstrained DeepSeek-Coder-7B baseline by +52.3 and +41.2 percentage points respectively. These results demonstrate that formal grammar constraints are a practical and effective approach to structured code generation with LLMs.

*(Palabra count: ~150 palabras — borde inferior aceptable para IEEE)*

---

*Este archivo es parte del módulo Research del curso ACA-2026.*
