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

# Qué es un Proyecto de Investigación: Estructura Fundamental

> **Módulo:** Research
> **Semana:** 1
> **Tiempo de lectura:** ~25 minutos

---

## Introducción

Imagina que eres un detective en una película. No empiezas simplemente buscando pistas al azar. Primero, planteas una pregunta clara: "¿Quién cometió el crimen?" Luego, reúnes evidencia sistemáticamente, la analizas cuidadosamente, y finalmente presentas tu caso de manera convincente. Un proyecto de investigación académico funciona exactamente así.

Un proyecto de investigación no es simplemente "un documento que escribo al final de mi carrera". Es una **declaración argumentada sobre una pregunta de investigación**, respaldada por evidencia y análisis riguroso. En nuestro caso particular—investigación sobre generación de kernels GPU con restricciones gramaticales—estás participando en la construcción del conocimiento científico en inteligencia artificial y compiladores.

¿Por qué esto importa? Porque tu proyecto no es solo para ti. Es una contribución que otros investigadores, ingenieros y estudiantes futuros usarán como punto de partida para sus propios proyectos. Establecer los cimientos correctamente desde el inicio determina toda la estructura que construirás después.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Entender la diferencia fundamental entre un planteamiento de problema y una hipótesis
2. Identificar los componentes esenciales de un proyecto de investigación académico
3. Seleccionar y utilizar herramientas de gestión de referencias (Zotero, Mendeley)
4. Organizar tu investigación desde el inicio con las prácticas recomendadas

---

## La Anatomía de un Proyecto: Más que un Documento

Un proyecto de investigación académico está estructurado como un argumento retórico bien construido. Piensa en él como un edificio: necesitas cimientos sólidos (revisión bibliográfica), columnas robustas (marco teórico), una estructura clara (metodología) y un propósito definido (tus conclusiones).

Los componentes principales son:

**1. Portada y materia administrativa** - El "embalaje" de tu investigación.

**2. Resumen ejecutivo (Abstract)** - Una ventana de 250 palabras hacia tu investigación completa. Muchas personas solo leerán esto, así que debe ser absolutamente clara y convincente.

**3. Introducción** - Donde estableces el contexto general, explicas por qué el problema importa, y guías al lector hacia tu pregunta específica.

**4. Revisión bibliográfica** - Tu conversación con investigadores anteriores. No es simplemente una lista de papers; es un análisis crítico de qué se sabe, qué sigue siendo incierto, y dónde encaja tu contribución.

**5. Marco teórico** - Los conceptos fundamentales que necesita tu lector para entender tu investigación. En nuestro caso: restricciones gramaticales, generación de código, arquitectura de GPUs.

**6. Metodología** - La "receta exacta" de cómo hiciste tu investigación. Tan específica que alguien debería poder reproducir tu trabajo.

**7. Resultados** - Lo que encontraste, presentado con claridad y rigor estadístico cuando sea aplicable.

**8. Conclusiones** - Donde regresas a tus preguntas originales, explicas qué aprendiste, y sugieres hacia dónde debería avanzar la investigación siguiente.

> 💡 **Concepto clave:** Un proyecto de investigación no es un reporte de "lo que hice". Es un argumento cuidadosamente construido donde cada sección respalda y construye sobre la anterior.

:::{figure} diagrams/thesis_structure.png
:name: fig-thesis-structure
:alt: Diagrama mostrando la estructura de un proyecto de investigación con sus componentes principales: portada, abstract, introducción, revisión bibliográfica, marco teórico, metodología, resultados y conclusiones
:align: center
:width: 90%

**Figura 1:** Estructura fundamental de un proyecto de investigación académico, mostrando cómo cada sección construye sobre la anterior para formar un argumento coherente.
:::

---

## Planteamiento del Problema vs. Hipótesis: Una Distinción Crucial

Aquí es donde muchos estudiantes se confunden. Veamos la diferencia con un ejemplo práctico.

### El Planteamiento del Problema

Es tu pregunta de investigación expresada como una brecha en el conocimiento. Es **observable, importante, y sin respuesta clara**.

**Ejemplo débil:** "Los kernels GPU son importantes"
- Esto es obvio, no es un problema a resolver

**Ejemplo fuerte:** "Los métodos actuales para generar kernels GPU optimizados requieren expertise manual en CUDA, lo que limita su adopción. ¿Podemos automatizar la generación de kernels manteniendo restricciones gramaticales que garanticen seguridad?"
- Esto identifica un problema específico, por qué importa, y sugiere una dirección de investigación

### La Hipótesis

Es tu **predicción específica y falsable** sobre cómo responder al problema. Nota la palabra clave: falsable. Significa que puede ser comprobada como falsa.

**Hipótesis de nuestra investigación podría ser:**

"Un modelo de lenguaje fine-tuned restringido gramaticalmente puede generar kernels GPU más seguros que métodos no-restringidos, con pérdida mínima en velocidad de generación (menos del 5%)."

Observa que es:
- **Específica:** Define exactamente qué medimos (kernels "más seguros", "pérdida menor al 5%")
- **Falsable:** Podemos realmente probar si es falsa (ejecutamos la métrica, comparamos resultados)
- **Basada en lógica:** Tiene una razón de ser (restricciones gramaticales → seguridad)

> 💡 **Concepto clave:** Tu planteamiento de problema es la pregunta; tu hipótesis es tu predicción de respuesta.

### Preguntas de Investigación vs. Hipótesis

A veces usaremos "Preguntas de Investigación" (RQ1, RQ2, RQ3) en lugar de hipótesis formales. Estos son intermedios: más específicos que un problema general, pero no tan formales como una hipótesis estadística.

Ejemplo:
- **RQ1:** ¿Reducen las restricciones gramaticales la tasa de kernels sintácticamente inválidos?
- **RQ2:** ¿Cuál es el overhead computacional de enforcing restricciones gramaticales durante generación?
- **RQ3:** ¿Qué patrones de kernels son más afectados por restricciones gramaticales?

---

## Gestión de Referencias: Tu Sistema de Organización

Conforme lees papers relacionados a tu investigación, necesitarás organizarlos, anotarlos, y recuperarlos fácilmente. Aquí es donde herramientas de gestión de referencias brillan.

### Zotero vs. Mendeley

Ambas son excelentes; la elección es personal.

**Zotero:**
- Código abierto y gratuito
- Integración fluida con navegador (captura referencias automáticamente)
- Almacenamiento local o en la nube
- Ideal si: Valoras la privacidad y quieres control total

**Mendeley:**
- Interfaz pulida y moderna
- Capacidades sociales (ver qué leen otros investigadores)
- Planes de pago para almacenamiento en la nube
- Ideal si: Prefieres una interfaz visual limpia y características modernas

### Workflow Recomendado

1. **Crea una colección/carpeta** para tu proyecto (ej: "GPU-Kernels-Research")

2. **Mientras busques papers**, agrega con una anotación rápida:
   - Relevancia para tu trabajo (alta/media/baja)
   - Palabras clave que encontraste útiles
   - Una frase sobre por qué importa

3. **Organiza por tema:**
   - Kernel generation techniques
   - Grammar-constrained generation (XGrammar, Outlines, Guidance)
   - GPU optimization
   - LLM fine-tuning

4. **Anota mientras lees:**
   - Resalta hallazgos clave
   - Escribe notas sobre cómo conecta con tu investigación
   - Marca limitaciones o preguntas abiertas

5. **Exporta según necesites:**
   - Generación automática de bibliografía en APA/IEEE
   - Sincronización con tu editor de documentos

> 💡 **Consejo práctico:** Invierte 30 minutos en configurar tu sistema de referencias ahora. Ahorrarás horas después cuando necesites citarlas.

---

## La Importancia de Comenzar Bien

Un grave error común: comenzar a escribir sin tener claro tu planteamiento de problema y tus preguntas de investigación. Es como construir una casa sin planos—terminas con paredes que no se alinean y habitaciones que no tienen sentido.

Tu supervisor debería aprobar tu problema y preguntas de investigación **antes de** que empieces a profundizar en metodología. Esta es una inversión de tiempo que se pagará exponencialmente.

---

## Resumen

En esta lectura exploramos:

- **Estructura de un proyecto:** Desde Abstract hasta Conclusiones, cada sección tiene un propósito específico
- **Problema vs. Hipótesis:** El problema es tu pregunta; la hipótesis es tu predicción falsable
- **Preguntas de Investigación:** Una forma estructurada de articular qué exactamente investigarás
- **Gestión de referencias:** Herramientas como Zotero y Mendeley te ayudan a organizarte desde el inicio

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Cuál es la diferencia fundamental entre un planteamiento de problema y una hipótesis? Proporciona un ejemplo de tu propia investigación.

2. ¿Por qué es importante que una hipótesis sea falsable? ¿Qué sucedería si escribieras una hipótesis que no pudiera ser falsada?

3. Nombre los cuatro componentes críticos de un proyecto según la analogía del "edificio" presentada en esta lectura.

### Ejercicio práctico

**Fase 1 - Identifica el problema (15 minutos)**

Escribe 2-3 párrafos respondiendo: "¿Cuál es el problema específico que mi investigación abordará?" Asegúrate de que sea observable, importante, y que actualmente no tenga una respuesta clara.

**Fase 2 - Formula preguntas de investigación (15 minutos)**

Basado en tu planteamiento de problema, escribe tres preguntas de investigación (RQ1, RQ2, RQ3) que sean:
- Específicas (no vagas)
- Respondibles con tu metodología
- Ordenadas de lo general a lo específico

**Fase 3 - Abre tu herramienta de referencias (15 minutos)**

Si nunca has usado Zotero o Mendeley, crea una cuenta gratuita, instala la extensión del navegador, y agrega 3 papers relevantes a tu tema como prueba. Anota por qué cada uno es relevante.

### Para pensar

> *¿Cuál es la diferencia entre escribir un proyecto sobre "lo que aprendí haciendo un proyecto" versus escribir un proyecto que responde una pregunta de investigación original? ¿Por qué crees que muchas universidades insisten en esta distinción?*

---

## Próximos pasos

En la siguiente semana, aprenderás a **leer papers académicos estratégicamente** (cómo extraer valor sin leerlo todo palabra por palabra) y a **manejar la abrumadora cantidad de literatura** existente en tu área. Verás cómo los papers que recopilas ahora se convertirán en el fundamento de tu revisión bibliográfica.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*
