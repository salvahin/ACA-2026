# Alternativas Offline para Recursos de Video

Este documento lista los videos de YouTube recomendados en el curso y sus equivalentes **sin Internet**, útiles para entornos restringidos o sin
conexión.

> [!TIP]
> Todos los notebooks del curso incluyen el contenido esencial cubierto en cada video. Los videos son material **complementario**, no obligatorio.

---

## 🧠 Módulo AI (Aprendizaje Profundo)

| Lección | Video | Alternativa Offline |
|---------|-------|---------------------|
| AI L01 | [But what is a neural network? (3B1B)](https://www.youtube.com/watch?v=aircAruvnKk) | Cap. 1–3 de *Deep Learning* (Goodfellow) — [PDF libre](https://www.deeplearningbook.org/) |
| AI L02 | [Backpropagation (3B1B)](https://www.youtube.com/watch?v=Ilg3gGewQ5U) | Sección de retropropagación en el notebook `02_fundamentos_deep_learning.md` |
| AI L02 | [Gradient Descent (3B1B)](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Ejercicio del notebook: trazar la pérdida paso a paso con `matplotlib` |
| AI L03 | [Transformers explained (Andrej Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Paper original: *Attention Is All You Need* (Vaswani et al., 2017) [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| AI L04 | [Illustrated Transformer](https://www.youtube.com/watch?v=4Bdc55j80l8) | Sección `causal_attention()` en `04_arquitectura_transformer.md` |
| AI L05 | [BERT explained](https://www.youtube.com/watch?v=--TqwBvW9v8) | Paper BERT: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) |

---

## ⚙️ Módulo Compilers (Autómatas y Gramáticas)

| Lección | Video | Alternativa Offline |
|---------|-------|---------------------|
| C L02 | [DFA (Neso Academy)](https://www.youtube.com/watch?v=40i4PKpM0cE) | Simulador DFA Plotly en `02-lenguajes-regulares-dfas.md` — **Sin internet** |
| C L03 | [NFA to DFA (Neso Academy)](https://www.youtube.com/watch?v=--TqwBvW9v8) | Código de construcción de subconjuntos en `03-nfas-conversion-dfa.md` |
| C L04 | [Context-Free Grammars](https://www.youtube.com/watch?v=Rsc5znwR5FA) | Sección de CFGs y derivaciones en `04-gramaticas-libres-contexto.md` |
| C L05 | [LL(1) Parsing](https://www.youtube.com/watch?v=eC6Hd1hFvos) | Ejemplos de parsing tabla en `05-bnf-ebnf-parsing.md` |

---

## 📊 Módulo Stats (Estadística)

| Lección | Video | Alternativa Offline |
|---------|-------|---------------------|
| S L01 | [Probability Basics (Khan Academy)](https://www.youtube.com/watch?v=uzkc-qNVoOk) | Sección "Axiomas de Probabilidad" + Teorema de Bayes interactivo en `01-fundamentos-probabilidad.md` |
| S L02 | [Normal Distribution](https://www.youtube.com/watch?v=rzFX5NWojp0) | Código de Plotly para visualización de distribuciones en `02-distribuciones-descriptiva.md` |
| S L03 | [Hypothesis Testing](https://www.youtube.com/watch?v=0oc49DyA3hU) | Walkthrough interactivo t-test en `03-pruebas-hipotesis.md` |
| S L04 | [Statistical Power](https://www.youtube.com/watch?v=Rsc5znwR5FA) | Simulador power analysis en `04-poweranalysis-diseno.md` |

---

## 🔬 Módulo Research (Metodología)

| Lección | Video | Alternativa Offline |
|---------|-------|---------------------|
| R L01 | [How to Read a Paper](https://www.youtube.com/watch?v=733m6qBH-jI) | Guía de lectura en `02-revision-bibliografica.md` |
| R L02 | [Literature Review](https://www.youtube.com/watch?v=GrLsW-gABm8) | Plantilla de revisión bibliográfica en el módulo |
| R L04 | [Related Work Section](https://www.youtube.com/watch?v=hOofv-Yy1k0) | `plantilla_metodologia.md` y ejemplos en `04-marco-teorico.md` |

---

## 🚀 Módulo Project 1 (LLM Code Generation)

| Lección | Video | Alternativa Offline |
|---------|-------|---------------------|
| P1 L01 | [LLM Intro (Andrej Karpathy)](https://www.youtube.com/watch?v=7kZrrCsJdEM) | `01_setup_intro_llms.md` — Simulaciones interactivas Plotly sin GPU |
| P1 L04 | [Constrained Decoding](https://www.youtube.com/watch?v=QXjU9qTsYCc) | Paper XGrammar: [arXiv:2411.15100](https://arxiv.org/abs/2411.15100) |

---

## 📥 Cómo Descargar Videos para Uso Offline

```bash
# Instalar yt-dlp (recomendado sobre youtube-dl)
pip install yt-dlp

# Descargar un video en calidad máxima
yt-dlp "https://www.youtube.com/watch?v=aircAruvnKk" -o "%(title)s.%(ext)s"

# Descargar solo el audio (para escuchar mientras estudias)
yt-dlp -x --audio-format mp3 "URL" -o "audio/%(title)s.%(ext)s"

# Descargar una lista completa de videos del curso
yt-dlp --batch-file urls.txt -o "videos/%(title)s.%(ext)s"
```

> [!NOTE]
> Verifica los términos de servicio de YouTube antes de descargar. Los videos enlazados en este curso son de canales educativos (3Blue1Brown, Andrej Karpathy, Neso Academy) que típicamente permiten descarga personal para uso educativo offline.
