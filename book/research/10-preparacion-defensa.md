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

# Preparación para Defensa: Del Documento Escrito a la Presentación Oral

> **Módulo:** Research
> **Semana:** 10
> **Tiempo de lectura:** ~28 minutos

---

## Introducción

Has escrito un proyecto de investigación excelente. Verificaste su estructura. Ahora viene lo aterrador: hablar sobre ella frente a un panel de expertos. La defensa oral es donde demuestras no solo que escribiste un documento bueno, sino que entiendes profundamente tu investigación, puedes responder preguntas difíciles, y puedes defender tus decisiones.

Aquí está lo bueno: si escribiste tu proyecto de investigación cuidadosamente (que has hecho), ya tienes 80% del material listo. Solo necesitas traducirlo de "documento escrito" a "presentación oral" y prepararte para preguntas difíciles.

Esta lectura final de la serie te enseña a estructurar una presentación compelling, anticipar preguntas, manejar objeciones, y demostrar maestría en tu área.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Estructurar una presentación de defensa que es clara, concisa, y convincente
2. Crear slides efectivos que apoyan (no distraen) tu narrativa
3. Anticipar preguntas difíciles y preparar respuestas
4. Manejar crítica y objeciones profesionalmente
5. Demostrar dominio y confianza durante la defensa

---

## Estructura de una Presentación de Defensa: El Arco Narrativo

Tu presentación debe contar la misma historia que tu proyecto de investigación, pero de forma comprimida y más dinámica.

### Duración Típica

- **Presentación:** 20-30 minutos (típicamente 25 para un proyecto de investigación de maestría)
- **Preguntas:** 20-40 minutos

Esto significa debes comprimir meses de trabajo en 25 minutos. Eficiencia es crítica.

### Estructura de 7 Partes (Para 25 minutos)

**1. Título y Contexto (2-3 minutos)**

Slide 1: Título, tu nombre, fecha, universidad
Slide 2: Contexto general—por qué este problema importa

"Hoy hablaré sobre generación automática de código GPU. Como modelos de lenguaje se vuelven más poderosos, la pregunta no es 'podemos generar código' sino 'podemos generar código correcto'. Esto es donde restricciones gramaticales entran."

**2. Problema y Preguntas (3-4 minutos)**

Slides 3-4: El problema específico, preguntas de investigación

"Cuando usamos modelos de lenguaje para generar código, frecuentemente producen sintaxis inválida. Mientras investigaciones previas propusieron restricciones gramaticales, nadie había evaluado sistemáticamente su costo versus beneficio. Nuestra pregunta primaria: ¿Pueden restricciones gramaticales formales mejorar calidad mientras mantienen velocidad practicable?"

**3. Trabajo Anterior (3-4 minutos)**

Slide 5: Tabla comparativa de enfoques anteriores

"Antes de nuestro trabajo, había tres enfoques principales: (1) generación sin restricciones, rápida pero a menudo incorrecta; (2) regex-based constraints, balance moderado; (3) formal grammars, muy restrictivas pero no bien evaluadas. Nuestro trabajo cierra esta brecha."

**4. Metodología (4-5 minutos)**

Slides 6-7: Cómo lo hiciste

"Evaluamos tres sistemas en 1,500 kernels de código abierto. Medimos tres dimensiones: validez sintáctica (¿compila?), velocidad (latencia de generación), y seguridad funcional (¿produce resultados correctos?). Esto nos permitió ver tradeoffs."

Muestra un diagrama simple de tu setup experimental.

**5. Resultados (5-6 minutos)**

Slides 8-10: Los hallazgos principales

Este es tu "moment". Presenta resultados con impacto:

"EBNF restrictions achieved 94% syntactic validity—a 16 percentage point improvement over the unconstrained baseline. This is statistically significant (p < 0.001) and practically meaningful. However, this comes with 15% computational overhead. Here's the tradeoff landscape..."

Usa gráficos claros. Enfatiza números clave.

**6. Implicaciones (2-3 minutos)**

Slide 11: Qué significa

"Estos hallazgos sugieren que gramáticas EBNF formales son prácticas para aplicaciones donde corrección importa más que latencia ultra-baja. Para aplicaciones latency-sensitive, el overhead requiere optimización. Esto guía ingeniería práctica de generadores de código."

**7. Conclusión y Preguntas (1-2 minutos)**

Slide 12: Conclusión, trabajo futuro

"En resumen, investigamos un problema importante con metodología rigurosa y encontramos hallazgos prácticos. Trabajo futuro podría optimizar overhead o extender a otros dominios. Ahora estoy listo para preguntas."

### Notas sobre Timing

Practica en voz alta con un timer. Errores comunes:
- Gastar 10 minutos en antecedentes (aburrido)
- Gastar solo 3 minutos en resultados (no es suficiente enfoque)
- Conclusión precipitada (parece apresurado)

Propuesta: 20% antecedentes, 30% metodología, 40% resultados, 10% conclusión.

:::{figure} diagrams/defense_timeline.png
:name: fig-defense-timeline
:alt: Diagrama de cronograma de defensa mostrando la distribución de tiempo para una presentación de 25 minutos: contexto (2-3 min), problema (3-4 min), trabajo anterior (3-4 min), metodología (4-5 min), resultados (5-6 min), implicaciones (2-3 min), conclusión (1-2 min)
:align: center
:width: 90%

**Figura 1:** Distribución recomendada de tiempo para una defensa de proyecto de investigación de maestría de 25 minutos, optimizando para máximo impacto en la presentación de resultados.
:::

> 💡 **Concepto clave:** Cada slide debería tomar 1-2 minutos de presentación. Si toma más, quizás tiene demasiado contenido.

---

## Diseño de Slides: Haz que tu Investigación Brille

Slides malas oscurecen incluso investigación brillante.

### Principios de Buen Diseño

**1. Una idea por slide**

**❌ Malo:**
[Slide titulado "Methodology and Results"]
[Párrafo completo de metodología + tabla de resultados + 3 figuras]

Demasiado. El lector no sabe dónde enfocarse.

**✅ Bueno:**
[Slide 1 titulado "Experimental Setup": diagrama simple del setup]
[Slide 2 titulado "Syntactic Validity Results": un gráfico claro]
[Slide 3 titulado "Performance Overhead": otro gráfico]

Cada slide, una idea central.

**2. Texto mínimo**

Usa bullets, no párrafos:

**❌:**
"The methodology involves evaluating three different approaches across a dataset of 1,500 kernels. We measured multiple metrics including syntactic validity, generation speed, and functional correctness..."

**✅:**
- 3 grammar constraint approaches
- Dataset: 1,500 kernels
- Metrics: Validity, Speed, Correctness

Tú proporciona los detalles oralmente. Las slides son notas visuales.

**3. Figuras claras**

Tus gráficos en el proyecto de investigación son buenos. Para slides, simplifica:
- Quita colores no-esenciales
- Aumenta tamaño de fuente (debe ser legible desde 5m de distancia)
- Quita gridlines si no ayudan
- Leyendas claras y grandes

**4. Consistencia visual**

Usa el mismo color scheme en todas las slides. Tipografía consistente. No mezules fonts.

### Estructura de Slide Estándar

```
┌─────────────────────────────────────────┐
│ Título Claro (24pt, bold)               │
├─────────────────────────────────────────┤
│                                         │
│  • Bullet 1 (18pt)                      │
│  • Bullet 2                             │
│    - Sub-bullet (14pt)                  │
│                                         │
│  [Figura o tabla si es necesario]       │
│                                         │
└─────────────────────────────────────────┘
   Página # (8pt, abajo a la derecha)
```

### Herramientas Recomendadas

- **PowerPoint/Google Slides:** Fácil, familiar, buen soporte
- **Beamer (LaTeX):** Profesional, matemáticas hermosas (si necesario)
- **Reveal.js:** HTML-based, moderno, pero requiere tech-comfort

Cualquiera funciona. Importante es el contenido, no la herramienta.

---

## Anticipación de Preguntas: Preparate para lo Inesperado

El panel puede hacer preguntas profundas, capciosas, o completamente fuera de campo. Aquí está cómo prepararse.

### Categorías de Preguntas

**Categoría 1: Clarificación**

"Cuando dices 'EBNF', ¿te refieres a Extended Backus-Naur Form específicamente, o a formalismos relacionados?"

**Respuesta:** Claro, conciso, refiere a slide si es necesario.

**Categoría 2: Profundidad técnica**

"¿Cómo exactamente manejabas [detalles técnicos]? ¿Consideraste [complicación alternativa]?"

**Respuesta:** Aquí es donde tu profundo conocimiento de tu investigación brilla. Puedes ir profundo.

**Categoría 3: Límites de validez**

"¿Estos resultados generalizarían a kernels más grandes? ¿Qué pasa si cambias el modelo base?"

**Respuesta:** Honesta. "Evaluamos kernels de 50-500 líneas. Kernels más grandes podría diferir. Eso es trabajo futuro interesante."

**Categoría 4: Crítica metodológica**

"¿Por qué no comparaste contra [otro método]? ¿Por qué no mediste [otra métrica]?"

**Respuesta:** "Nos enfocamos en [razón]. [Otro método] sería interesante pero está fuera del alcance de esta proyecto de investigación. Eso podría ser trabajo futuro."

**Categoría 5: Relevancia / Impacto**

"¿Quién realmente usaría esto? ¿Cuál es el impacto práctico?"

**Respuesta:** "Ingenieros que generan código automáticamente se enfrentan exactamente a este tradeoff. Nuestros resultados guían qué restricciones usar..."

**Categoría 6: Lo Completamente Inesperado**

"¿Cómo se conecta esto a [tópico completamente diferente]?"

**Respuesta:** "Excelente pregunta. Mientras no es directo, yo diría..."

Intenta hacer una conexión razonable o admite que es fuera de alcance.

### Preparación Sistemática

**Paso 1: Anticipa las 10 Preguntas Más Probables**

Escribe 10 preguntas que crees que el panel preguntará:

1. "¿Por qué EBNF específicamente?"
2. "¿Cómo se compara contra [método alternativo]?"
3. "¿Evaluaste en modelos reales?"
4. "¿Qué pasa si el kernel es muy complejo?"
5. etc.

**Paso 2: Escribe Respuestas**

Para cada pregunta, escribe 1-2 párrafos de respuesta. Practica en voz alta.

**Paso 3: Anticipa Objeciones Sutiles**

Algunos panelistas pueden ser provocadores:
- "Esto no es realmente nuevo, ¿verdad?"
- "Tus resultados son modestos."
- "¿Por qué debería importarle a alguien?"

Prepara respuestas que defienden tu trabajo sin ser defensivo.

**Paso 4: Practica con Colegas**

Pide a amigos/colegas que hagan el papel de panelistas y te hagan preguntas. Esto es invaluable para acostumbrarte al formato.

---

## Manejo de Crítica: Ganar Incluso Cuando te Desafían

Durante la defensa, alguien puede objetar tu trabajo o cuestionar un resultado.

### Cómo Responder

**No defensivo:**
"Buena pregunta. Esperábamos ese resultado pero nuestros datos muestran lo opuesto. Aquí está mi interpretación..."

**Defensivo (evitar):**
"Eso es incorrecto. Claramente no entiendes mi investigación."

(Esto te hace parecer inseguro e inmaduro.)

### Estructura de Respuesta a Crítica

1. **Agradece la pregunta/objeción**

"Gracias por plantear eso."

2. **Muestra que entiendes la preocupación**

"Entiendo tu punto—si [preocupación], entonces [implicación]. Aquí está cómo lo abordé..."

3. **Proporciona tu respuesta**

"Mi interpretación es [reasoning]. Los datos que soportan esto están en [slide/sección]."

4. **Ofrece extensión si es aplicable**

"Eso sería interesante investigar más a fondo. Mi proyecto de investigación se enfocó en [scope], pero definitivamente es un trabajo futuro valioso."

### Ejemplo

**Panel:** "Tus números de overhead (15%) parecen altos. ¿Están seguros?"

**Respuesta:** "Excelente pregunta. El overhead que reporté es específicamente para la compilación de gramática. Incluí desglose en el apéndice (Appendix A, Tabla A.3) mostrando: 8% en parsing, 5% en matching, 2% en otras operaciones. He verificado estas números múltiples veces. Ahora, si optimizamos la compilación, podríamos probablemente reducir esto. De hecho, ese es uno de mis items de trabajo futuro."

Esto es confiado, honesto, y muestra maestría.

---

## La Presentación En Vivo: Tips de Ejecución

### Antes de Comenzar

- Llega temprano, chequea la tecnología (projector, mouse, etc.)
- Respira profundo
- Deja tus notas / speaker notes donde puedas verlas sin leer de ellas directamente
- Bebe agua (tu garganta se secará)

### Durante la Presentación

**Velocidad:** Lento es mejor que rápido. Muchos estudiantes apresuran.

**Volumen:** Asegúrate de que la gente al fondo puede oírte.

**Eye contact:** Mira a diferentes miembros del panel, no solo al que hace preguntas.

**Gestos:** Úsalos. No estés rígido. Pero evita pacing nervioso.

**Umms/Ahhs:** Practica para evitar muletillas verbales.

**Slides:** Apunta a la slide con tus palabras: "Como ves en la Figura 3..." Pero no leas la slide directamente.

### Manejo de Nervios

Es normal estar nervioso. El panel lo sabe. Ellos ven esto todo el tiempo. Aquí está cómo manejarlo:

- **Practica mucho.** Nervios disminuyen con familiaridad.
- **Recuerda que eres el experto.** Sabes tu investigación mejor que cualquiera en la sala.
- **Respira lentamente.** Oxígeno calma el sistema nervioso.
- **Falla gracefully.** Si tropiezas con palabras o pierdes tu lugar, simplemente continúa. No te disculpes obsesivamente.

> 💡 **Concepto clave:** La defensa no es un examen que puedas fallar. Es una conversación donde demuestras que entiendes tu trabajo. El panel quiere que tengas éxito.

---

## Las Preguntas Finales: Lo que el Panel Realmente Pregunta

Después de tu presentación, el panel hace preguntas de seguimiento. Aquí están los tipos:

**1. Clarificaciones técnicas**

"En tu metodología, ¿cómo exactamente manejaste [detalles]?"

Responde con precisión. Usa slides o dibuja en pizarra si es necesario.

**2. Implicaciones más amplias**

"¿Cómo piensas que tus hallazgos impactan el campo?"

"¿Debería esto cambiar cómo los investigadores piensan sobre este problema?"

**Respuesta:** Serio pero no arrogante. Relaciona a tu trabajo.

**3. Trabajo futuro**

"¿Cuál es el siguiente paso natural?"

**Respuesta:** You ya tienes esto de tu proyecto de investigación. Solo amplía.

**4. Preguntas de perspectiva**

"Si fueras a hacer esto nuevamente, ¿qué cambiarías?"

**Respuesta:** Reflexiva. Muestra crecimiento.

---

## Práctica y Timing

**2 semanas antes:**
- Slides finalizadas
- Practica completa una vez (nota timing)
- Pide feedback de supervisor o colegas

**1 semana antes:**
- Practica 2-3 veces
- Memoriza apertura y cierre
- Prepara respuestas a preguntas anticipadas

**3 días antes:**
- Practica una última vez, en voz alta, con timer
- Durmete bien
- No memorices todo (suena robótico)

**Día anterior:**
- No practiques (cerebro necesita descanso)
- Revisa slides casualmente
- Prepara lo que usarás (ropa, notas, agua)

**Día de la defensa:**
- Come bien (no con los nervios)
- Llega 30 minutos temprano
- Camina un poco para calmar nervios

---

## Resumen

En esta lectura exploramos:

- **Estructura de 25 minutos:** 20% contexto, 30% metodología, 40% resultados, 10% conclusión
- **Diseño de slides:** Una idea por slide, texto mínimo, figuras claras
- **Anticipación de preguntas:** 10 preguntas probables, preparar respuestas
- **Manejo de crítica:** Agradece, muestra entendimiento, responde, ofrece extensión
- **Ejecución:** Lento, volumen, eye contact, practica mucho

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Cuál es la diferencia entre una defensa "presentación de hechos" vs. "demostración de maestría"?

2. ¿Por qué es importante anticipar preguntas antes de la defensa?

3. ¿Cómo deberías responder a una crítica que te desafía?

### Ejercicio práctico

**Tarea 1: Creación de Slides (3 horas)**

Crea una presentación de 12-15 slides para tu defensa:
1. Título (nombre, date, universidad)
2. Contexto (por qué importa)
3-4. Problema + Preguntas de investigación
5. Trabajo anterior
6-7. Metodología
8-10. Resultados principales (3 slides)
11. Implicaciones
12. Conclusión

Usa principios de diseño: una idea por slide, texto mínimo, figuras claras.

**Tarea 2: Preguntas Anticipadas (45 minutos)**

Escribe 10 preguntas que crees que el panel preguntará. Para cada una, escribe 2-3 oraciones de respuesta.

**Tarea 3: Respuesta a Crítica (30 minutos)**

Imagina una crítica a tu trabajo:
- Escribe la crítica
- Escribe cómo responderías
- Asegúrate de que es no-defensiva, muestra entendimiento, proporciona evidence

**Tarea 4: Práctica En Voz Alta (60 minutos)**

Practica tu presentación completa en voz alta con timer:
- Tiempo total (debería ser ~25 minutos)
- Nota dónde te atropellas o pierdes tu lugar
- Mejora para siguiente intento

Idealmente, practica múltiples veces (mínimo 3) hasta que se sienta fluido.

### Para pensar

> *¿Cuál es la diferencia entre "presentar resultados" y "defender una investigación"? ¿Qué habilidades adicionales necesitas para el segundo?*

---

## Epílogo: La Jornada Completa

Comenzaste esta serie de lecturas sin proyecto de investigación, sin estructura, sin dirección. Ahora tienes:

✓ Una pregunta de investigación clara y respondible
✓ Una revisión bibliográfica que sintetiza (no solo resume) tu campo
✓ Un marco teórico que establece conceptos fundamentales
✓ Una metodología tan detallada que otros podrían reproducirla
✓ Resultados presentados con rigor estadístico
✓ Conclusiones que retornan a tus preguntas originales y sugieren trabajo futuro
✓ Una proyecto de investigación que ha pasado auditoría de integridad interna
✓ Una presentación convincente y práctica para defenderte

Lo más importante: Has aprendido el proceso de investigación rigurosa. Este proceso no es específico a proyecto de investigación de maestría. Es el proceso que los investigadores de élite en universidades de élite usan. Dominar esto es una habilidad que te llevará lejos en tu carrera.

Tu proyecto de investigación es solo el comienzo.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Recursos Adicionales Recomendados

Para profundizar en áreas específicas después de completar esta serie:

- **Sobre escritura académica:** "The Sense of Structure" (George Gopen), "Craft of Research" (Wayne Booth)
- **Sobre presentaciones:** "Presentation Zen" (Garr Reynolds), "Talk Like TED" (Carmine Gallo)
- **Sobre investigación empírica:** "Statistical Rethinking" (Richard McElreath), "Designing Experiments" (Douglas Montgomery)
- **Sobre GPU y compiladores:** NVIDIA CUDA Programming Guide, LLVM Compiler Infrastructure documentation
- **Sobre LLMs y restricciones:** arXiv papers en "Constrained Generation", "Grammar-Constrained Decoding"

---

*Felicidades por completar esta serie. Ahora ve y construye un proyecto de investigación excelente.*
