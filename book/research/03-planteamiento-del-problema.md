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

# Planteamiento del Problema: Convertir Curiosidad en Preguntas de Investigación

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/03-planteamiento-del-problema.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
# Este módulo no requiere instalaciones de Python adicionales.
# Asegúrate de haber ejecutado 00_verificacion_entorno.ipynb antes de iniciar el proyecto.
print("Módulo Research — sin dependencias de código adicionales.")
print('Dependencies installed!')
```

> **Módulo:** Research
> **Semana:** 3
> **Tiempo de lectura:** ~26 minutos

---

```{admonition} 🎬 Video Recomendado
:class: tip

**[Writing a Problem Statement](https://www.youtube.com/watch?v=bRrmNJx5GQw)** - Guía paso a paso para convertir una idea general en un planteamiento de problema específico, falsable y respondible para investigación académica.
```

## Introducción

Has leído papers. Has identificado patrones en el campo. Has notado algunas cosas que no tienen respuesta clara. Ahora viene la parte aterradora: **convertir toda esa curiosidad amorfa en preguntas de investigación formales y específicas**.

Muchos estudiantes aquí cometen un error crítico: quieren hacer "demasiado". Quieren resolver el problema fundamental de la generación de código, optimización de GPU, y cambiar el mundo todo en un proyecto de investigación de maestría. Mientras el entusiasmo es admirable, un proyecto de investigación fuerte normalmente aborda **una pregunta clara y bien delimitada**.

Esta lectura te enseña a tomar un área general de interés y refinarlo en 2-3 preguntas precisas, falsables, y respondibles. Es un proceso de refinamiento iterativo, no un epifanía instantáneo.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Escribir un planteamiento de problema que identifique una brecha específica en el conocimiento
2. Formular 3 preguntas de investigación (RQ1-RQ3) que sean específicas, falsables, y respondibles
3. Construir hipótesis formales que hagan predicciones comprobables
4. Evaluar si tu planteamiento es lo suficientemente delimitado para un proyecto de investigación

---

## De Interés General a Pregunta Específica: El Embudo de Refinamiento

Visualiza tu proceso de investigación como un embudo:

```
Área amplia: "GPU Kernel Generation with AI"
           ↓
Área más enfocada: "Grammar-constrained generation for GPU kernels"
           ↓
Tema específico: "Impact of grammar constraints on generation speed vs. safety"
           ↓
Pregunta de investigación: "Can we achieve <5% performance overhead while maintaining
                            grammar-constraint safety in GPU kernel generation?"
```

Cada nivel es más específico que el anterior. Tu objetivo es llegar al fondo del embudo.

:::{figure} diagrams/refinement_funnel.png
:name: fig-refinement-funnel
:alt: Diagrama en forma de embudo que muestra el proceso de refinamiento desde un área amplia de interés hasta preguntas de investigación específicas y falsables
:align: center
:width: 90%

**Figura 1:** Embudo de refinamiento para convertir intereses generales en preguntas de investigación específicas, falsables y respondibles.
:::

### Características de un Buen Planteamiento de Problema

Un planteamiento de problema fuerte tiene estas características:

**1. Basado en una brecha real**

No inventas problemas. Idealmente, tu revisión bibliográfica reveló que algo importante está sin responder.

- ❌ Débil: "Nadie ha combinado ever attention mechanisms con GPU kernel generation"
- ✅ Fuerte: "Investigaciones previas (Smith 2023, Johnson 2024) muestran que restricciones gramaticales mejoran corrección, pero el costo computacional no ha sido sistematicamente medido, limitando su adopción práctica"

**2. Importante y relevante**

¿A quién le importa tu pregunta?

- ❌ Débil: "¿Qué color debería ser la salida de depuración?"
- ✅ Fuerte: "¿Podemos reducir la tasa de kernels sinténticamente inválidos de 15% a <5% usando restricciones, sin sacrificar rendimiento?"

La respuesta afecta a ingenieros reales, investigadores, o productividad en la industria.

**3. Respondible con recursos que tienes**

No necesitas resolver todo, solo tu parte.

- ❌ Demasiado amplio: "¿Cómo revolucionamos toda la compilación de GPU?"
- ✅ Respondible: "¿Cómo impactan diferentes esquemas de restricción gramáticales en la velocidad de generación de kernels CUDA?"

**4. Expresado sin predisponer la respuesta**

Tu planteamiento no debería estar escondiendo tu conclusión deseada.

- ❌ Sesgado: "¿Por qué son mejores las restricciones gramaticales?"
- ✅ Neutral: "¿Cuál es el impacto de las restricciones gramaticales en la corrección vs. velocidad?"

> 💡 **Concepto clave:** Tu planteamiento debe ser una pregunta genuina cuya respuesta te sorprendería si fuera opuesta a tu intuición.

---

## Formulación de Preguntas de Investigación (RQ1, RQ2, RQ3)

Tus preguntas de investigación son más específicas que tu planteamiento general. Idealmente, tienes 2-4 de ellas, ordenadas de lo general a lo específico.

### Estructura recomendada

**RQ1: La pregunta primaria (la que la mayoría de tu proyecto de investigación responde)**

Ejemplo: "¿Pueden las restricciones gramaticales compiladas a GPU mejorar la calidad de kernels generados mientras mantienen overhead computacional bajo (<10%)?"

Esta es tu pregunta "grande". Muchas sub-investigaciones la informan.

**RQ2-RQ3: Preguntas subsidiarias (detalles que apoyan RQ1)**

Ejemplo:
- RQ2: "¿Cuál es el overhead exacto de diferentes tipos de restricciones gramaticales (EBNF, regex, CFG) en hardware moderno?"
- RQ3: "¿Existen clases de kernels para las cuales las restricciones gramaticales son particularmente beneficiosas o problemáticas?"

### Criterios para buenas RQs

**Específicas:**
- ❌ "¿Qué es mejor, A o B?" (demasiado vago, "mejor" en qué métrica?)
- ✅ "¿Reduce A la métrica X en comparación con B, controlando por variable Y?"

**Falsables:**
- ❌ "¿Son interesantes los kernels GPU?" (opinión, no investigación)
- ✅ "¿Producen kernels sinténticamente válidos >90% del tiempo?" (comprobable)

**Independientes pero relacionadas:**
- La mayoría no deberían solaparse completamente
- Pero RQ2 y RQ3 deberían informar RQ1

**Respondibles con tu metodología:**
- Antes de formular RQ2 sobre "modelos especializados", asegúrate de que puedas realmente entrenar esos modelos
- Si tu presupuesto solo permite experimentos en A100, no hagas una RQ sobre Hopper

### Ejemplo Completo: Nuestro Dominio

**Contexto:** Tu investigación investiga restricciones gramaticales en generación de kernels GPU

**Planteamiento de Problema:**
"Métodos actuales para generar kernels GPU seguros requieren expertise manual o generación sin restricciones con alta tasa de error. Mientras investigaciones recientes muestran que restricciones gramaticales mejoran corrección, su costo computacional y aplicabilidad a kernels complejos siguen siendo poco entendidos."

**RQ1:**
"¿Pueden sistemas de generación de kernels GPU enfatizando restricciones gramaticales alcanzar >90% de kernels sinténticamente válidos mientras mantienen latencia de generación <200ms por kernel en hardware GPU estándar?"

**RQ2:**
"¿Cómo varía el overhead computacional de las restricciones gramaticales según el tipo de restricción (especificación formal vs. heurísticos) y la complejidad del kernel?"

**RQ3:**
"¿Son los kernels generados bajo restricciones gramaticales igualmente seguros en cuanto a seguridad funcional (no race conditions, memoria acceso válido) como kernels escritos manualmente?"

---

## Construcción de Hipóproyecto de investigación Formales (Opcional pero Recomendado)

Además de RQs, puedes formular hipótesis estadísticas. Estas son especialmente útiles si tu investigación incluye comparaciones cuantitativas.

### Estructura de Hipóproyecto de investigación

**Hipóproyecto de investigación Nula (H0):** No hay diferencia, efecto o relación

H0: "La validez sintáctica de kernels generados bajo restricciones gramaticales es estadísticamente idéntica a kernels generados sin restricciones."

**Hipóproyecto de investigación Alternativa (H1):** Sí hay diferencia, efecto o relación

H1: "La validez sintáctica de kernels bajo restricciones es significativamente mayor que sin restricciones."

El punto de la hipótesis nula es que es tu punto de partida neutral. Tu experimento intenta rechazarla (mediante pruebas estadísticas). Si no la rechazas, entonces no tienes evidencia de un efecto.

### Hipóproyecto de investigación Prácticas vs. Estadísticas

A veces, tu hipótesis estadística (mayor, menor) no es suficientemente específica. Puedes ser más práctico:

**Hipóproyecto de investigación práctica:**
"El overhead computacional de restricciones gramaticales será menor al 10% en comparación con generación sin restricciones, haciendo el enfoque práctico para uso industrial."

Esto combina el aspecto estadístico (menor overhead) con una evaluación práctica (qué threshold es "práctico").

> 💡 **Concepto clave:** Las hipótesis formales no son obligatorias, pero te fuerczan a ser preciso sobre qué exactamente esperas encontrar.

---

## Evaluando tu Planteamiento: Auto-Crítica Constructiva

Antes de presentar tu planteamiento a tu supervisor, pasa esta lista de verificación:

**Especificidad**
- [ ] Puedo explicar mi pregunta en 1-2 oraciones sin vaguedad
- [ ] Alguien que no es del campo aún entiende lo que investigaré
- [ ] No contengo dos preguntas confundidas en una sola oración

**Justificación**
- [ ] Puedo señalar por lo menos 2 papers/fuentes que sugieren que esta pregunta es importante
- [ ] He explicado por qué respuestas anteriores son insuficientes
- [ ] La respuesta afectaría decisiones reales en la industria o investigación

**Delimitación**
- [ ] Mi RQ1 es respondible en 6 meses de trabajo
- [ ] Mis RQ2-3 son abordables pero no triviales
- [ ] He identificado qué explícitamente NO investigaré

**Falsabilidad**
- [ ] Puedo describir un resultado que refutaría mis hipótesis
- [ ] No he "movido los posts" (redefinido "éxito" para cualquier resultado)
- [ ] Mis métricas son objetivas, no basadas en opinión

**Claridad**
- [ ] Mi planteamiento leerá igual en 2 meses (¿está claro para mi yo futuro?)
- [ ] He evitado jerga innecesaria
- [ ] Mis preguntas no presuponen una respuesta particular

---

## Refinar Iterativamente con tu Supervisor

Tu primer borrador de planteamiento de problema probablemente será muy amplio. Esto es normal.

### Ciclo de Refinamiento Típico

**Semana 1:** Presentas tu planteamiento inicial
- Supervisor: "Interesante, pero esto son casi 3 proyecto de investigación separadas. Enfócate en esto."

**Semana 2:** Reescribes más enfocado
- Supervisor: "Mejor. Pero ahora tu RQ2 es trivial si tienes los datos. ¿Puedes hacer que sea más desafiante?"

**Semana 3:** Ajustas la complejidad de RQ2-3
- Supervisor: "Excelente. Ahora, ¿cómo específicamente medirás 'kernels seguros' en RQ3? Tu métrica es aún vaga."

**Semana 4:** Especificas métricas
- Supervisor: "Perfecto. Vamos adelante con esto."

Este ciclo típicamente toma 2-4 semanas. **Es tiempo bien invertido.** Un planteamiento confuso destruye toda la investigación que viene después.

---

## Documentación de tu Planteamiento

Una vez que tu supervisor ha aprobado tu planteamiento, documéntalo formalmente. Esto se convierte en la sección de "Introducción" y "Preguntas de Investigación" de tu proyecto de investigación.

**Estructura recomendada:**

```
## 1. Introducción (1.5 páginas)
   - Contexto general (¿por qué debería importarle a alguien?)
   - Trabajo anterior y sus limitaciones
   - Tu pregunta específica

## 2. Preguntas de Investigación (0.5 páginas)
   - RQ1 (primaria)
   - RQ2 (subsidiaria 1)
   - RQ3 (subsidiaria 2)

## 3. Contribuciones Propuestas (0.5 páginas)
   - Cómo tu trabajo aborda las RQs
   - Por qué esto es diferente de trabajo anterior
```

---

## Resumen

En esta lectura exploramos:

- **El embudo de refinamiento:** De interés general a preguntas precisas
- **Características de buenos planteamientos:** Basado en brechas reales, importante, respondible, neutral
- **Preguntas de Investigación (RQs):** RQ1 primaria más RQ2-3 subsidiarias, todas específicas y falsables
- **Hipóproyecto de investigación formales:** Hipóproyecto de investigación nulas y alternativas que puedes usar para estructura adicional
- **Auto-crítica:** Checklist de especificidad, justificación, delimitación

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Cuál es la diferencia entre un "planteamiento de problema" general y "preguntas de investigación"? ¿Por qué necesitas ambos?

2. ¿Por qué es importante que tu planteamiento sea "neutral" y no presuponga la respuesta? Proporciona un ejemplo.

3. ¿Qué hace una pregunta de investigación "respondible"? Contrasta con una pregunta que es demasiado amplia.

### Ejercicio práctico

**Fase 1: Planteamiento Inicial (45 minutos)**

Basado en tu revisión bibliográfica, escribe:
- Un planteamiento de problema (2-3 párrafos)
- Tu RQ primaria
- 2 preguntas de investigación subsidiarias

No necesita ser perfecta; es un borrador para obtener feedback.

**Fase 2: Auto-Crítica (20 minutos)**

Pasa tu borrador a través de la checklist de auto-crítica provista en el texto. Para cada ítem, escribe brevemente por qué pasas o no pasas.

**Fase 3: Refinamiento (30 minutos)**

Basado en tu auto-crítica, reescribe tu planteamiento. Intenta hacerlo:
- Más específico (menos vago)
- Más delimitado (menos amplio)
- Más justificado (cita papers que sugieren por qué esto importa)

**Fase 4: Discusión con Supervisor (30 minutos)**

Programa una breve sesión (o envía por email) con tu supervisor mostrando tu borrador refinado. Pide feedback específicamente sobre:
- Especificidad de mis RQs
- Razonabilidad del alcance
- Alguna brecha que no haya considerado

### Para pensar

> *¿Cómo te sentirías si después de 6 meses de investigación, descubrieras que tu pregunta de investigación era ambigua o demasiado amplia? ¿Cómo puedes usar el trabajo de esta semana para prevenir eso?*

---

## Próximos pasos

Una vez que tienes un planteamiento aprobado, necesitarás construir un **marco teórico sólido** que establezca los conceptos fundamentales que tu lector necesita entender. Verás cómo tu planteamiento de problema guía qué conceptos son más importantes para explicar.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Booth, W., Colomb, G., & Williams, J. (2016). The Craft of Research (4th ed.). University of Chicago Press.
- Creswell, J. (2014). Research Design: Qualitative, Quantitative, and Mixed Methods Approaches (4th ed.). SAGE.
