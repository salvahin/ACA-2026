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

# Conclusiones y Abstract: Cerrar tu Investigación con Impacto

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/08-conclusiones-y-abstract.ipynb)
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
> **Semana:** 8
> **Tiempo de lectura:** ~26 minutos

---

## Introducción

Aquí está la verdad incómoda: muchas personas que leen tu proyecto de investigación solo leerán dos cosas: el Abstract (al principio) y las Conclusiones (al final). Algunos quizás saltarán directo al Abstract y luego a las Conclusiones, completamente. Esto significa que estas dos secciones necesitan ser extraordinarias.

Tu Abstract es tu "elevator pitch" de 250 palabras. Tu Conclusión es donde cierras el círculo: regresas a tus preguntas de investigación, explicas qué aprendiste, y sugières hacia dónde va el campo. Ambas son críticas para comunicar el impacto de tu trabajo.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Escribir un Abstract que es preciso, conciso, y compelling
2. Estructurar una Sección de Conclusiones que vuelve a tus RQs
3. Articular contribuciones claras de tu investigación
4. Discutir limitaciones honestamente sin depreciar tu trabajo
5. Proponer trabajo futuro que surge naturalmente de tu investigación

---

## El Abstract: Tu Ventana de 250 Palabras al Mundo

El Abstract es el primero (a veces el único) contacto que la mayoría de personas tiene con tu investigación. Tiene que contar tu historia completa brevemente.

### Estructura Estándar del Abstract

**1. Contexto / Motivación (1-2 oraciones)**

"Los modelos de lenguaje son herramientas poderosas para generar código, pero frecuentemente producen sintaxis inválida. Garantizar la corrección sintáctica es crítico para aplicaciones prácticas."

Establece por qué alguien debería importarle.

**2. Brecha / Problema (1-2 oraciones)**

"Mientras trabajos previos han propuesto restricciones gramaticales para mejorar corrección, su costo computacional y aplicabilidad a código de bajo nivel como CUDA kernels permanecen poco comprendidos."

¿Qué falta en la literatura existente?

**3. Tu Enfoque / Pregunta (1-2 oraciones)**

"Esta proyecto de investigación investiga cómo restricciones gramaticales formales afectan tanto la calidad como la velocidad de generación de kernels GPU, evaluando tres sistemas principales: regex, gramáticas libres de contexto, y gramáticas EBNF."

¿Qué exactamente hiciste?

**4. Metodología Brevísima (1-2 oraciones)**

"Condujimos un estudio controlado comparando estos tres enfoques en 1,500 kernels de prueba de código abierto, midiendo validez sintáctica, velocidad de generación, y seguridad funcional."

¿Cómo lo hiciste?

**5. Resultados Clave (1-3 oraciones)**

"El enfoque EBNF alcanzó 94.3% de validez sintáctica (vs. 78% sin restricciones), con overhead computacional de 15%. Mientras la exactitud mejoró significativamente, el overhead sugiere que optimizaciones adicionales son necesarias para aplicaciones latency-sensitive."

¿Qué encontraste?

**6. Implicaciones / Conclusión (1 oración)**

"Estos hallazgos indican que restricciones gramaticales formales son prácticas para mejorar corrección de código generado, aunque es necesario optimización de compilación de gramática para aplicaciones que requieren generación rápida."

¿Por qué importa?

### Ejemplo Completo: Un Abstract de Muestra

"Generative models for code, while powerful, frequently produce syntactically invalid output. Grammar-constrained generation—limiting the model's output to a formal grammar—is a promising approach to improve validity, but its computational cost remains unclear. This thesis systematically investigates the tradeoffs of grammar-constrained generation for CUDA kernel generation, comparing three approaches: regular expressions, context-free grammars, and EBNF specifications. We evaluate these approaches on 1,500 kernels from open-source repositories, measuring syntactic validity, generation latency, and functional safety. Results show that EBNF-constrained generation achieves 94.3% syntactic validity (vs. 78% unconstrained), but incurs 15% computational overhead. While validity improvements are substantial and practically significant, the overhead suggests additional optimization is necessary for latency-sensitive applications. These findings advance our understanding of grammar-constrained generation and provide guidance for practitioners choosing between correctness and speed."

(Conteo de palabras: ~150 palabras)

:::{figure} diagrams/abstract_structure.png
:name: fig-abstract-structure
:alt: Diagrama que muestra la estructura de un abstract académico con sus seis componentes: contexto/motivación, brecha/problema, enfoque, metodología, resultados clave e implicaciones
:align: center
:width: 90%

**Figura 1:** Estructura estándar de un Abstract académico, mostrando los seis componentes esenciales que comunican la investigación completa de manera concisa en ~250 palabras.
:::

### Longitud y Contenido

- **Objetivo:** ~250 palabras (permitido rango 200-300 típicamente)
- **Verbos:** Usa pasado para lo que hiciste, presente para implicaciones
- **Cuidado:** No uses referencias. Ej: "Como Smith et al. muestran" — el Abstract debe ser auto-contenido

### Revisión del Abstract

El Abstract es frecuentemente lo último que escribes. Escribe cuando sabes exactamente qué encontraste.

**Ciclo de revisión típico:**

Borrador 1: ~350 palabras, demasiado detalle, lenguaje impreciso
↓
Feedback: "Aún demasiado largo. ¿Cuáles son los 3 números clave?"
↓
Borrador 2: ~280 palabras, más enfocado, claridad mejorada
↓
Feedback: "Mejor. Pero la parte sobre 'future work' no pertenece aquí. Termina con implicaciones."
↓
Borrador 3: ~248 palabras, listo

> 💡 **Consejo práctico:** Escribe tu Abstract último. Conocerás exactamente qué reportar, evitando redacción cuando no sabes tus resultados finales.

---

## Estructura de la Sección de Conclusiones

Tu conclusión es donde cierras tu argumento y la das al lector un sentido de cierre y dirección futura.

### Estructura Recomendada

**1. Regresar a tu Pregunta de Investigación Primaria (RQ1) (1 página)**

"En esta proyecto de investigación, investigamos la pregunta: ¿Pueden restricciones gramaticales compiladas a GPU mejorar la calidad de kernels generados mientras mantienen overhead bajo? Nuestros resultados indicaron que sí—restricciones EBNF mejoraron validez sintáctica de 78% a 94%—pero con tradeoff computacional que requiere consideración."

Explícitamente regresas a tu pregunta y resumidamente responde.

**2. Preguntas Subsidiarias (RQ2, RQ3) (0.5 páginas)**

"Nuestras preguntas subsidiarias sobre overhead (RQ2) revelaron que diferentes restricciones tienen costos diferentes: regex ~8% overhead, EBNF ~15%. Nuestra pregunta sobre seguridad funcional (RQ3) encontró que restricciones sintácticas no garantizan corrección funcional—un importante distinction."

Brevemente aborda cada una.

**3. Contribuciones Principales (1 página)**

"Las contribuciones clave de esta investigación son:

1. Evaluación sistemática del costo-beneficio de restricciones gramaticales formales en generación de kernels GPU
2. Análisis comparativo de tres aproximaciones (regex, CFG, EBNF) en múltiples dimensiones
3. Identificación de que restricciones EBNF son prácticas para mejorar corrección, pero optimización es necesaria
4. Resultados que informan decisiones de ingeniería sobre qué restricciones usar en práctica"

Sé explícito. ¿Qué contribuyó realmente que no fue conocido antes?

**4. Limitaciones (0.5 páginas)**

"Nuestra investigación tiene limitaciones que deben reconocerse:

- Evaluamos solo en kernels de GPU; resultados pueden no generalizar a código de otros dominios
- Dataset fue filtrado a kernels de 50-500 líneas; kernels más grandes pueden comportarse diferentemente
- Evaluamos 'correctness' usando test suites simples; evaluación contra especificaciones formales sería más rigurosa
- Estudio fue realizado en una arquitectura GPU (A100); resultados en hardware más nuevo podrían variar"

Honestamente reporta dónde tu estudio fue limitado. Esto no debilita tu contribución; fortalece credibilidad. Algunos investigadores futuros usarán exactamente estas limitaciones como punto de partida.

**5. Direcciones Futuras (0.5 páginas)**

"Investigación futura podría:

1. Investigar optimizaciones de compilación de gramática para reducir overhead bajo 5%
2. Extender evaluación a kernels más complejos y diversos dominios de código
3. Explorar restricciones de seguridad funcional, no solo sintáctica
4. Investigar cómo restricciones interactúan con diferentes arquitecturas de modelo (GPT-3.5 vs. Llama vs. otros)"

Sugiere trabajo que surge naturalmente de lo que descubriste.

**6. Reflexión Final (1-2 párrafos)**

"Esta investigación contribuye a nuestra comprensión de cómo generar código confiable automáticamente. A medida que modelos de lenguaje se vuelven más prevalentes en ingeniería de software, garantizar que el código generado es correcto se vuelve cada vez más importante. Nuestros hallazgos sugieren un camino hacia sistemas de generación más seguros y prácticos."

Termina con una nota sobre significancia más amplia.

---

## Contribuciones: Sé Explícito

Un error común: Tu contribución no es clara. El lector termina pensando "¿Qué exactamente fue nuevo aquí?"

Sé explícito. Escribe una lista de "Contribuciones Principales":

**❌ Vago:**
"Nuestro trabajo avanza el campo de generación de código con nuevos insights."

(¿Cuáles insights?)

**✅ Explícito:**
"Nuestras contribuciones son:
1. Primera evaluación sistemática del costo-beneficio de EBNF vs. regex vs. CFG en generación de kernels GPU
2. Demostración de que restricciones EBNF alcanzan 94% de validez, mejora de 16pp sobre baseline
3. Análisis de la relación entre complejidad de restricción y overhead computacional
4. Reconocimiento de que corrección sintáctica no implica corrección funcional (implicación importante)"

Ahora está claro. Esto es contribución.

---

## Limitaciones: Honestidad como Fortaleza

Reconocer limitaciones es un signo de rigor, no debilidad.

### Estructura de Limitaciones

**1. Alcance**
"Evaluamos kernels GPU específicamente; la generalización a otros dominios es incierta."

**2. Tamaño/Escala**
"Nuestro dataset de 1,500 kernels es modesto; evaluación a escala más grande sería valiosa."

**3. Métrica**
"Evaluamos 'correctness' funcional usando test suites simples; especificaciones formales sería más riguroso."

**4. Contexto**
"Nuestro estudio fue realizado en GPUs A100 en 2024; resultados en hardware futuro pueden variar."

**5. Metodología**
"Usamos modelos base sin fine-tuning adicional específico a kernel generation; modelos especializados podrían rendimiento diferente."

### Cómo Presentar Limitaciones

**No debilites tu trabajo:**
"Nuestra investigación fue muy limitada y probablemente poco válida..."

(Esto es auto-sabotaje)

**Sé honesto y equilibrado:**
"Mientras nuestro dataset fue modesto (1,500 kernels), es representativo de kernels de código abierto de tamaño típico. Evaluación a escala más grande sería valiosa pero probablemente no cambiaría hallazgos principales."

Reconoce, pero proporciona contexto de por qué aún es válido.

---

## Trabajo Futuro: Emergente vs. Independiente

Trabajo futuro debería surgir de tu investigación, no ser una lista aleatoria.

**❌ Independiente (no conectado):**
"Trabajo futuro podría: 1) Generación de kernels, 2) Optimización de GPU, 3) Machine Learning Interpretability, 4) ..."

(¿Por qué estos? Parecen no conectados a tu investigación.)

**✅ Emergente (conectado):**
"Nuestros hallazgos sugieren tres direcciones futuras:

1. **Optimización de compilación:** Nuestro análisis de profiling sugiere que la compilación de gramática causa 60% del overhead. Investigación en compilación más eficiente podría recuperar la mayoría del overhead.

2. **Restricciones funcionales:** Esta investigación evaluó solo correctness sintáctica. Restricciones que también garantizan propiedades funcionales (sin race conditions, etc.) sería más poderoso.

3. **Generalización:** ¿Funcionan restricciones EBNF tan bien para otros dominios de código (sistemas, aplicaciones web)? ¿Los resultados generalizan?"

Cada uno surge naturalmente de tu investigación.

---

## Checklist para Conclusiones Fuertes

- [ ] Regreso explícito a RQ primaria: respondo directamente
- [ ] Preguntas subsidiarias brevemente direccionadas
- [ ] Contribuciones principales claramente listadas
- [ ] Limitaciones honestamente reconocidas (sin auto-sabotaje)
- [ ] Trabajo futuro surge naturalmente de limitaciones + hallazgos
- [ ] Reflexión final sobre significancia más amplia
- [ ] Tono: confiado pero no arrogante
- [ ] Longitud: 3-5 páginas típicamente (varía según campo)

---

## Resumen

En esta lectura exploramos:

- **Abstract:** 250 palabras, 6 secciones (contexto, brecha, enfoque, método, resultados, implicaciones)
- **Conclusiones:** Retorno a RQs, contribuciones explícitas, limitaciones honestas, trabajo futuro
- **Contribuciones:** Sé específico; "nuevo insights" no es suficiente
- **Limitaciones:** Honestidad que fortalece, no debilita
- **Trabajo futuro:** Debe surgir de tu investigación, no ser aleatorio

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Cuáles son los 6 componentes de un Abstract bien estructurado?

2. ¿Por qué es importante ser explícito sobre contribuciones en lugar de vago?

3. ¿Cómo debería un investigador presentar limitaciones de manera que fortalezca en lugar de debilitar su trabajo?

### Ejercicio práctico

**Tarea 1: Escribir un Abstract (60 minutos)**

Escribe un Abstract para tu investigación siguiendo la estructura de 6 partes:
1. Contexto/Motivación (1-2 oraciones)
2. Brecha/Problema (1-2 oraciones)
3. Tu enfoque (1-2 oraciones)
4. Metodología brevísima (1-2 oraciones)
5. Resultados clave (1-3 oraciones)
6. Implicaciones (1 oración)

Objetivo: ~250 palabras. Después de escribir, contar palabras y ajustar si es necesario.

**Tarea 2: Contribuciones Explícitas (30 minutos)**

Escribe una lista de 3-5 contribuciones principales de tu investigación. Para cada una:
- Sé específico (no "insights", sino "qué insight")
- Explica por qué es nueva
- Explica por qué importa

**Tarea 3: Limitaciones Honestas (25 minutos)**

Escribe 3-4 limitaciones significativas de tu investigación. Para cada una:
- Describe la limitación claramente
- Explica por qué existe
- Discute cómo afecta conclusiones (¿invalidalas? ¿moderalas?)
- Sugiere investigación futura que la aborda

**Tarea 4: Estructura de Conclusiones (40 minutos)**

Escribe un outline para tu sección de Conclusiones:
- Párrafo sobre RQ1 (1-2 párrafos)
- Párrafo sobre RQ2-3 (0.5 página)
- Contribuciones principales (listadas)
- Limitaciones (listadas)
- Trabajo futuro (3-4 direcciones)
- Reflexión final (1-2 párrafos)

### Para pensar

> *Si solo dos secciones de tu proyecto de investigación serán leídas por la mayoría de personas (Abstract y Conclusiones), ¿cómo cambia cómo escribes estas secciones? ¿Qué es absolutamente crítico que comuniques?*

---

## Próximos pasos

Con Abstract y Conclusiones escritas, estás casi al final. La próxima lectura enfatiza **verificación de consistencia e integridad** de tu proyecto de investigación entera: revisando que tus preguntas de investigación se responden en resultados, que tus conclusiones son soportadas por evidencia, y que no tienes contradicciones internas. Este es el "quality control" final antes de presentar.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Swales, J. & Feak, C. (2009). Abstracts and the Writing of Abstracts. University of Michigan Press.
- Day, R. & Gastel, B. (2016). How to Write and Publish a Scientific Paper (8th ed.). Cambridge University Press.
