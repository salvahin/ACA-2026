# Syllabus: ACA-2026

## Información General
**Curso:** TC3002B - ACA: Grammar-Constrained GPU Kernel Generation  
**Duración:** 10 semanas  
**Carga Horaria:** 12 horas presenciales (6 módulos × 2 horas) + 17 horas de estudio independiente.  
**Fechas:** 23 de marzo 2026 – 3 de junio 2026.  

---

## El Reto
¿Puedes enseñar a una IA a escribir código GPU ultra-optimizado? En este curso construirás un sistema que combina **Inteligencia Artificial**, **Teoría de Compiladores** y **GPU Computing** para generar automáticamente kernels de alto rendimiento con garantías formales de corrección mediante gramáticas formales.

### Objetivos de Aprendizaje
Al finalizar el curso, serás capaz de:
1. **Diseñar gramáticas formales** (BNF/EBNF) para restringir la generación de código.
2. **Implementar kernels GPU optimizados** con Triton y CUDA.
3. **Construir pipelines de constrained decoding** con LLMs (XGrammar).
4. **Evaluar y benchmarkear** el rendimiento de código generado y compararlo con baselines.
5. **Aplicar metodología de investigación** rigurosa y comunicar resultados técnicos.

---

## Módulos del Curso
El curso se divide en 6 módulos que se imparten de manera paralela:

| Módulo | Descripción | Lecturas |
|---|---|---|
| **Inteligencia Artificial** | Transformers, LLMs, tokenización y fine-tuning. | 8 |
| **Compiladores** | Lenguajes regulares, CFGs, BNF y parsing. | 10 |
| **Estadística** | Pruebas de hipótesis, power analysis y diseño experimental. | 10 |
| **Research** | Metodología, escritura académica y defensa de proyecto. | 10 |
| **Project 1** | Generación de código, prompt engineering y XGrammar. | 10 |
| **Project 2** | Arquitectura GPU, optimización de memoria y benchmarking. | 10 |

---

## Calendario Semanal

| Sem | Inicio | AI | Compiladores | Estadística | Research | Proyectos (1 & 2) |
|---|---|---|---|---|---|---|
| **1** | 23 Mar | IA Clásica vs Gen | Intro Compiladores | Probabilidad | Qué es un Proyecto | Setup & GPU Fund. |
| **2** | 8 Abr | Deep Learning | Lenguajes Regulares | Distribuciones | Revisión Biblio. | LLMs Código & CUDA |
| **3** | 15 Abr | Gen Autoregresiva | NFAs y DFAs | Pruebas Hipótesis | Planteamiento Prob. | Prompting & Memoria |
| **4** | 22 Abr | Transformer | Gramáticas CFG | Power Analysis | Marco Teórico | XGrammar & Triton |
| **5** | 29 Abr | Tokens & GPTs | BNF/EBNF | Reproducibilidad | Metodología | JSON & Adv Patterns |
| **6** | 6 May | Sampling | Pipeline FSM | Comparaciones Mult. | Estilo Académico | Triton L1-L2 & Corpus |
| **7** | 13 May | Fine-tuning | Diseño DSL | Pruebas No Param. | Resultados | Triton L3-L4 & Debug |
| **8** | 20 May | MLOps | Compilación Gram. | Tamaño de Efecto | Conclusiones | Serving & Baselines |
| **9** | 27 May | *Revisión* | Context-Sensitive | MLOps Stats | Revisión Estruct. | Token Econ. & Visual. |
| **10** | 3 Jun | *Final* | Parser Generators | Reporte Estad. | Preparación Defensa | Agents & Integración |

> [!WARNING]
> Existe un periodo de vacaciones entre la Semana 1 (23 Mar) y la Semana 2 (8 Abr).

---

## Stack Tecnológico
- **LLMs:** CodeLlama, DeepSeek, GPT-4.
- **Frameworks:** PyTorch, Triton, vLLM.
- **Gramáticas:** XGrammar, JSON Schema, EBNF.
- **GPU:** CUDA, NVIDIA Nsight.
- **MLOps/Research:** Weights & Biases, Docker, Git.

---

## Convención de Código
Seguimos la regla: **Español para comunicar, Inglés para programar.**
- **Código (variables, funciones, clases):** Inglés (`batch_size`, `compute_loss()`).
- **Comentarios y Docstrings:** Español (`# Calcular error`).
- **Strings de interfaz:** Español (`print("Iniciando entrenamiento...")`).

---

## Evaluación Final
El entregable principal es un **sistema funcional** de generación de kernels GPU con restricciones gramaticales, evaluado rigurosamente y documentado en un formato de tesis defendible.
