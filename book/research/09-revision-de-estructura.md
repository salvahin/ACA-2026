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

# Revisión de Estructura: Verificar la Integridad Interna de tu Proyecto de Investigación

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/09-revision-de-estructura.ipynb)
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
> **Semana:** 9
> **Tiempo de lectura:** ~25 minutos

---

## Introducción

Imagina que construyes una casa y terminas todas las habitaciones. Las paredes se ven bien, el techo está en su lugar, pero olvidaste verificar si las tuberías conectan correctamente o si las puertas se alinean. Cuando alguien intenta vivir en la casa, todo colapsa.

Una proyecto de investigación tiene el mismo riesgo. Puedes escribir un excelente Abstract, un marco teórico sólido, resultados rigurosos, y conclusiones convincentes. Pero si estos no se alinean internamente—si tus resultados no responden tus preguntas de investigación, si tus conclusiones van más allá de lo que tu evidencia soporta—toda la estructura falla.

Esta lectura te enseña a hacer una "auditoría de integridad" de tu proyecto de investigación entera, asegurandote que cada parte conecta lógicamente con otras.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Realizar una auditoría de integridad de tu proyecto de investigación entera
2. Verificar que tus preguntas de investigación se responden en tu sección de resultados
3. Asegurar que tus conclusiones están soportadas por evidencia
4. Identificar y resolver contradicciones internas
5. Checkear coherencia narrativa de principio a fin

---

## La Auditoría de Integridad: Verificación Sistemática

Una auditoría de integridad responde estas preguntas:

1. ¿Están tus resultados alineados con tus preguntas de investigación?
2. ¿Tus conclusiones son soportadas por tus resultados?
3. ¿Tu metodología fue suficiente para responder tus preguntas?
4. ¿Hay contradicciones entre secciones?
5. ¿La narrativa fluye de forma lógica?

Mejor aún, esta auditoría es algo que *puedes hacer tú mismo* antes de entregar a tu supervisor.

---

## Auditoría 1: RQs ↔ Resultados ↔ Conclusiones

Este es el "hilo dorado" de tu proyecto de investigación. Cada pregunta de investigación debe tener una correspondencia clara en resultados y conclusiones.

### Creando el Mapa

En una hoja de papel o documento, crea una tabla:

```
RQ | Sección Metodología | Sección Resultados | Sección Conclusiones
---|-------|-------------|----------|
RQ1: ¿Reducen restricciones... | Secc 5.2 | Figura 2, Table 2 | Párrafo 1-2
RQ2: ¿Cuál es el overhead... | Secc 5.3 | Figura 3 | Párrafo 3
RQ3: ¿Garantizan seguridad... | Secc 5.4 | Table 3 | Párrafo 4
```

Ahora tienes un mapa visual.

### Verificación Punto a Punto

Para cada RQ:

**¿Mi metodología fue diseñada para responder esta pregunta?**

RQ: "¿Reducen restricciones la tasa de kernels sinténticamente inválidos?"

Metodología: "Evaluamos tres modelos (sin restricciones, regex, EBNF) en 1,500 kernels de prueba y medimos validez sintáctica."

✅ Sí, la metodología responde directamente esta pregunta.

**¿Reporte resultados que responden esta pregunta?**

Resultados: "EBNF alcanzó 94.3% de validez (vs. 78% para sin restricciones)."

✅ Sí, reporté exactamente lo que necesitaba para responder RQ1.

**¿Direcciono esta pregunta explícitamente en mis conclusiones?**

Conclusiones: "Respondiendo RQ1, nuestros resultados demuestran que restricciones EBNF reducen significativamente la tasa de kernels inválidos..."

✅ Sí, explícitamente direcciono RQ1 en conclusiones.

Si todas las respuestas son "sí", tienes alineación. Si es "no" en cualquiera, tienes un problema a arreglar.

### Problemas Comunes

**Problema: Metodología diseñada para RQ3, pero no se reporta**

RQ3: "¿Garantizan seguridad funcional las restricciones?"

Metodología: "Evaluamos usando test suite de 50 tests..."

Resultados: (Ninguna mención de seguridad funcional—solo sintaxis)

Conclusiones: "Por lo tanto, las restricciones garantizan seguridad funcional."

(Esto es mentira. No evaluaste seguridad funcional.)

Solución: O 1) reporta los resultados de seguridad funcional que mediste, o 2) cambia RQ3 a lo que realmente mediste.

**Problema: Conclusión va más allá de evidencia**

Resultados: "EBNF fue más lento (312ms vs. 425ms para baseline)."

Conclusiones: "Por lo tanto, EBNF es impracticable para todas las aplicaciones en tiempo real."

(Mediste solo overhead, no testeaste aplicaciones reales en tiempo real.)

Solución: Modifica conclusión para lo que realmente mostraste: "EBNF tiene 15% overhead, lo que puede ser problemático para aplicaciones muy latency-sensitive, aunque para muchas aplicaciones esto es tolerable."

> 💡 **Concepto clave:** Tu conclusión no puede hacer afirmaciones que van más allá de tu evidencia. Si no mediste algo, no puedes concluir sobre ello.

---

## Auditoría 2: Coherencia de Métricas

Asegúrate de que cómo defines y reportas una métrica es consistente en toda el proyecto de investigación.

### Ejemplo: Definición de "Validez Sintáctica"

**En Metodología, defines:**
"Validez sintáctica se define como: el kernel compila exitosamente con nvcc sin errores de compilación."

**En Resultados, reportas:**
"El modelo alcanzó 94% de validez sintáctica" — OK, consistente

**Pero luego añades:**
"También, el 85% de kernels produjeron output correcto." — Aquí cambias a "funcional", no sintáctica

**En Conclusiones, afirmas:**
"Las restricciones mejoran validez sintáctica" — Este es sobre sintaxis, correcto

Pero si confundiste sintáctica y funcional en Resultados, esto es inconsistente.

### Cómo Verificar

En tu proyecto de investigación, busca cada métrica importante (Ctrl+F "syntactic validity"):

- Primera ocurrencia: ¿Se define claramente?
- Ocurrencias siguientes: ¿Se usa la definición consistentemente?
- ¿Hay algún lugar donde cambio la definición sin advertir al lector?

Si encuentras inconsistencias, arréglalo:
1. Standardiza la definición
2. Actualiza todas las ocurrencias
3. Explícitamente nota cualquier cambio deliberado

---

## Auditoría 3: Verificación de Hechos

Errores fácticos son desastres de credibilidad. Un número citado incorrectamente, un nombre de autor mal escrito—esto socava la confianza.

### Qué Verificar

**1. Números reportados consistently**

¿Reportaste 1,500 kernels en Metodología, pero 1,501 en Resultados?

Herramienta: Busca cada número importante (1500, 94%, etc.) en toda el proyecto de investigación. Asegúrate de que es consistente.

**2. Citas de trabajos anteriores**

¿Dijiste que "Smith et al. muestran validez de 85%"? ¿Realmente encontraste eso en el papel?

Herramienta: Vuelve y verifica cada citación numérica. Puedes estar recordando mal.

**3. Nombres, años, títulos**

¿Es "XGrammar" o "X-Grammar"? ¿2023 o 2024?

Herramienta: Usa tu gestor de referencias (Zotero, Mendeley) para buscar información canónica.

**4. Configuraciones reportadas**

¿Dijiste en Metodología que usaste "batch size 16" pero en Resultados parece haber usado algo diferente basado en los números?

Herramienta: Verifica que resultados son consistentes con configuraciones reportadas.

### Cómo Ejecutar Esta Auditoría

Crea un documento "Fact Check":

```
Número/Hecho | Dónde se reporta | Verificación | Estado
---|---|---|---
1,500 kernels | Método 3.2, Res 5.1 | Ambos dicen 1,500 ✓ | OK
94% validez | Res 5.2, Conc 6.1 | Ambos dicen 94.3% ✓ | OK
Smith 2023 | Intro 1.2, Rel Work 2.1 | Paper es de 2024 ✗ | ARREGLAR
Batch size 16 | Método 4.3, (implícito en Res) | Números consistentes ✓ | OK
```

---

## Auditoría 4: Contradicciones Lógicas

A veces, afirmas dos cosas que no pueden ambas ser verdaderas.

### Ejemplo 1

Introducción: "Las restricciones gramaticales son ampliamente adoptadas en la industria."

Conclusiones: "Investigación futura debería explorar cómo hacer restricciones prácticas para industria."

(Si ya son ampliamente adoptadas, ¿por qué necesitan ser más prácticas?)

### Ejemplo 2

Metodología: "Usamos modelos base sin fine-tuning adicional para aislar el efecto de restricciones."

Discusión: "Nuestros modelos fueron fine-tuned en datos sintéticos para mejorar performance."

(¿Fine-tuned o no? Esto es contradictorio.)

### Cómo Encontrarlas

Lee tu proyecto de investigación de principio a fin, buscando afirmaciones que se contradigan. Haz preguntas como:

- ¿Afirmo que X es cierto pero luego que X es falso?
- ¿Digo que hice A pero luego que no hice A?
- ¿Afirmo que X es importante pero luego lo ignoro?

Cuando encuentras una contradicción, resuelvela:
1. Determina cuál afirmación es correcta
2. Cambia la otra para ser consistente
3. Considera si hay una explicación legítima (ej: "Inicialmente creíamos X, pero nuestro análisis reveló Y")

---

## Auditoría 5: Coherencia Narrativa

Tu proyecto de investigación debería leer como una narrativa coherente, no como un conjunto de secciones desconectadas.

### La Narrativa Ideal

**Acto 1 (Introducción + Marco):** "Aquí está el problema, aquí está lo que se sabe, aquí está qué no sabemos (mi pregunta)."

**Acto 2 (Metodología):** "Aquí está exactamente cómo respondí esa pregunta."

**Acto 3 (Resultados):** "Aquí está lo que encontré."

**Acto 4 (Discusión + Conclusiones):** "Aquí está lo que significa, cómo se conecta a lo que otros han hecho, y qué viene después."

### Verificación de Flujo

Lee el primer párrafo de cada sección principal:

- Intro: ¿Estableció el problema?
- Marco: ¿Explicó conceptos necesarios para entender el problema?
- Metodología: ¿Explicó cómo estudiarías el problema?
- Resultados: ¿Presentó hallazgos directamente relacionados al problema?
- Conclusiones: ¿Revisited el problema original y explicó qué aprendimos?

Si sí a todas, tienes una buena narrativa. Si no, hay secciones que no conectan.

### Transiciones

Revisa las transiciones entre secciones:

**Final de Introducción → Inicio de Marco:**
"Para entender este problema, primero debemos comprender los conceptos fundamentales..."

¿Flye naturalmente? ¿O hay un salto sorpresa?

**Final de Marco → Inicio de Metodología:**
"Con esta comprensión, podemos ahora diseñar una investigación para responder nuestra pregunta..."

¿Flye naturalmente?

Si hay saltos abruptos, agrega conectores.

---

## Auditoría 6: Afirmaciones sin Soporte

A veces haces afirmaciones pero no las soportas con evidencia o reasoning.

### Búsqueda de Afirmaciones

Busca afirmaciones "fuertes" (palabras como "clearly", "obviously", "must", "always"):

- ❌ "Obviously, EBNF is better than regex."
  - Assertion, pero ¿dónde es obvio? ¿Todas las dimensiones? ¿Siempre?

- ✅ "On the primary metric of syntactic validity, EBNF outperforms regex (94% vs. 85%), although regex is faster (8% vs. 15% overhead)."
  - Assertion con soporte. Y honesta sobre tradeoffs.

### Cómo Verificar

Para cada afirmación fuerte:
1. ¿Es esta afirmación soportada por datos o argumentación?
2. ¿Es demasiado fuerte dadas las evidencias?
3. ¿Debería ser hedged (más cautela)?

Ejemplo de arreglo:

**Original:** "Restricciones gramaticales son la solución a problemas de corrección de código."

**Revisado:** "Nuestros resultados sugieren que restricciones gramaticales pueden mejorar significativamente la corrección sintáctica, aunque restricciones adicionales sería necesario para garantizar corrección funcional."

---

## Auditoría 7: Lenguaje Consistente

Pequeñas inconsistencias de lenguaje pueden confundir al lector.

### Ejemplos

**¿Es "kernel" o "CUDA kernel" o "GPU kernel"?**

Usa "kernel" después de haber definido claramente qué significa. Consistencia.

**¿Es "baseline" o "baseline model" o "unconstrained model"?**

Elige uno y úsalo consistentemente.

**¿Es "EBNF-constrained" o "EBNF constraints" o "EBNF approach"?**

Nuevamente, elige uno.

### Cómo Verificar

Usa Find & Replace en tu editor:
- Busca variaciones (ej: "EBNF", "ebnf", "EBNF-constrained")
- Identifica cuál es más consistente
- Usa Find & Replace para standardizar

---

## Checklist de Auditoría Completa

- [ ] Cada RQ tiene correspondencia clara en Resultados
- [ ] Cada Resultado importante se menciona en Conclusiones
- [ ] Conclusiones no afirman nada que no fue medido
- [ ] Todas las métricas se definen consistentemente
- [ ] Números reportados son idénticos en todas partes
- [ ] Citas numéricas verificadas contra fuentes originales
- [ ] Ninguna contradicción lógica (X es verdadero vs. X es falso)
- [ ] Narrativa fluye logicamente de Intro → Conclusiones
- [ ] Transiciones entre secciones son suaves
- [ ] Afirmaciones fuertes son soportadas o hedged
- [ ] Lenguaje y terminología son consistentes

---

## Cómo Conducir Auditorías Efectivamente

**Paso 1: Distancia (1-2 semanas)**

Después de escribir, toma 1-2 semanas de distancia. Necesitas objetividad.

**Paso 2: Leer de Corrido**

Lee tu proyecto de investigación de principio a fin de una sola vez (o en 2-3 sesiones). Nota donde te pierdes o donde hay confusión.

**Paso 3: Auditorías Sistemáticas**

Conduce las 7 auditorías arriba en orden.

**Paso 4: Documentar Problemas**

Escribe lista de "Issues Found":
- Locación (página, párrafo)
- Problema (ej: "RQ2 no se aborda en Conclusiones")
- Solución propuesta

**Paso 5: Arreglar Sistemáticamente**

No arregles todo a la vez. Arregla por tipo (primero inconsistencias numéricas, luego RQ alignment, etc.)

**Paso 6: Re-leer**

Después de arreglos significativos, re-lee al menos una vez.

---

## Resumen

En esta lectura exploramos:

- **Auditoría 1:** RQs alineadas con Resultados y Conclusiones
- **Auditoría 2:** Coherencia de métricas y definiciones
- **Auditoría 3:** Verificación de hechos (números, citas, configuraciones)
- **Auditoría 4:** Contradicciones lógicas
- **Auditoría 5:** Coherencia narrativa y flujo
- **Auditoría 6:** Afirmaciones sin soporte
- **Auditoría 7:** Consistencia de lenguaje y terminología

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Por qué es importante alinear RQs, Resultados, y Conclusiones?

2. ¿Cuál es la diferencia entre una métrica "definida consistentemente" vs. "inconsistentemente"?

3. ¿Cómo puedes detectar afirmaciones sin soporte en tu propia escritura?

### Ejercicio práctico

**Tarea 1: Mapa RQ-Resultados-Conclusiones (30 minutos)**

Crea una tabla que mapea:
- Cada pregunta de investigación
- Dónde en Metodología se aborda
- Qué Resultados la responden
- Dónde en Conclusiones se discute

**Tarea 2: Fact Check (40 minutos)**

Selecciona 5-10 "hechos" importantes de tu proyecto de investigación (números, años, nombres de autores, métricas). Para cada uno:
- Verifica que aparece consistentemente
- Si es cita de otro trabajo, verifica contra la fuente original
- Documentar en hoja de cálculo

**Tarea 3: Búsqueda de Contradicciones (30 minutos)**

Lee tu proyecto de investigación rápidamente y anota 3-4 lugar donde podría haber contradicciones o inconsistencias. Para cada una:
- Describe la contradicción
- Determina cuál es correcta
- Propone como arreglarlo

**Tarea 4: Coherencia Narrativa (25 minutos)**

Lee el primer párrafo de: Intro, Marco, Metodología, Resultados, Conclusiones.

¿Flye la narrativa? ¿O hay saltos sorpresa? Documenta y propone transiciones adicionales donde sea necesario.

### Para pensar

> *¿Cuánto de tu trabajo se pierde si tienes un proyecto de investigación con resultados brillantes pero estructura incoherente? ¿Cómo afecta la integridad estructural la credibilidad?*

---

## Próximos pasos

Una vez que has verificado la integridad interna de tu proyecto de investigación, tienes un documento sólido. El paso final es **preparar para tu defensa oral**: cómo presentar tu trabajo hablado, anticipar preguntas difíciles, y demostrar que entienden profundamente tu investigación.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Booth, W., Colomb, G., & Williams, J. (2016). The Craft of Research (4th ed.). University of Chicago Press.
- Day, R. & Gastel, B. (2016). How to Write and Publish a Scientific Paper (8th ed.). Cambridge University Press.
