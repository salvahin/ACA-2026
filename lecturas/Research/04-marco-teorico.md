# Marco Teórico: Construir los Cimientos Intelectuales de tu Investigación

> **Módulo:** Research
> **Semana:** 4
> **Tiempo de lectura:** ~27 minutos

---

## Introducción

Imagina que invitas a alguien a una cena, pero esa persona no habla tu idioma. Antes de conversar, necesitas un idioma común. De manera similar, tu tesis necesita construir un "idioma común" con tu lector antes de poder presentar tu investigación compleja.

Eso es lo que el marco teórico hace. No es simplemente una lista de definiciones. Es una construcción cuidadosa de los conceptos, teorías, y frameworks que tu lector **debe entender** para comprender por qué tu investigación importa y cómo abordas tu pregunta.

En nuestro contexto—generación de kernels GPU con restricciones gramaticales—tu lector necesita entender qué son kernels GPU, cómo funcionan modelos de lenguaje, qué significa "restricción gramatical", y cómo estos conceptos se conectan. Tu marco teórico es el puente que construye esas conexiones.

---

## Objetivos de Aprendizaje

Al finalizar esta lectura, serás capaz de:

1. Estructurar un marco teórico que sea progresivo (simple a complejo)
2. Conectar conceptos fundamentales a tu pregunta de investigación específica
3. Identificar qué es "marco teórico" vs. "revisión bibliográfica"
4. Integrar consideraciones éticas en tu marcos teórico
5. Escribir con precisión sobre conceptos técnicos sin sobrecargar al lector

---

## Marco Teórico vs. Revisión Bibliográfica: Dos Bestias Diferentes

Muchos estudiantes confunden estos. Son relacionados pero distintos.

### Revisión Bibliográfica

**Qué es:** Un análisis crítico de trabajos **anteriores** y cómo se relacionan entre sí.

**Propósito:** Mostrar dónde tu investigación se ubica en el panorama de conocimiento existente.

**Estructura:** "Smith hizo A, Johnson hizo B, Lee hizo C. Aquí está cómo se conectan."

**Ejemplo:** "Trabajos recientes en restricciones gramaticales (Smith 2023) y generación de código (Johnson 2024) han explorado cómo limitar el espacio de salida del modelo. Sin embargo, estos trabajos se enfocaron en tareas de NLP estándar, no en generación de código de bajo nivel como CUDA kernels."

### Marco Teórico

**Qué es:** Una explicación de los **conceptos y teorías fundamentales** que sustentan tu investigación.

**Propósito:** Asegurar que tu lector tiene el conocimiento mínimo necesario para entender tu investigación.

**Estructura:** "Un kernel GPU es..., restricciones gramaticales significan..., modelos de lenguaje funcionan mediante..."

**Ejemplo:** "Un kernel GPU es una función ejecutada en paralelo en múltiples threads del procesador. Los kernels deben respetar restricciones de memoria (coalesced access) y sincronización (barriers). La sintaxis es estricta: un error de tipado causa compilación a fallar."

### En la Práctica

Tu tesis probablemente tendrá ambas secciones:

```
Capítulo 2: Marco Teórico
   - Qué son kernels GPU (definición, restricciones)
   - Cómo funcionan modelos de lenguaje
   - Restricciones gramaticales (definición formal, ejemplos)

Capítulo 3: Revisión Bibliográfica / Trabajos Relacionados
   - Smith et al. 2023 en restricciones con LLMs
   - Johnson et al. 2024 en generación de código
   - Lee et al. 2023 en optimización de GPU
   - Cómo tu trabajo diferencia de estos
```

> 💡 **Concepto clave:** Marco teórico = "qué necesitas saber"; Revisión bibliográfica = "quién hizo qué antes".

---

## Estructura de un Marco Teórico Efectivo: De lo General a lo Específico

Un error común es organizar temáticamente sin una progresión lógica. Tu lector se pierde en los detalles sin entender el panorama.

### Estructura Recomendada: El "Funneling" Teórico

```
Nivel 1: Conceptos amplios y bien conocidos
   └─ Nivel 2: Conceptos más específicos
      └─ Nivel 3: Conceptos directamente relevantes a tu investigación
         └─ Nivel 4: Tu formulación específica del problema
```

![Marco Teórico: Estructura de Funneling](./diagrams/marco_teorico_funneling.png)

> **Marco Teórico: Del Concepto Amplio a la Formulación Específica del Problema**
>
> Jerarquía de 4 niveles con flechas de estrechamiento (de Computación Paralela → Kernels GPU → Generación Automática → tu RQ). Panel derecho: distinción entre Revisión Bibliográfica (qué existe en la literatura) y Marco Teórico (qué necesita el lector para entender tu trabajo). Panel inferior: plantilla de definición operacional con estructura: término + definición formal + cómo se mide en este proyecto.

### Ejemplo Concreto: Nuestro Dominio

**Nivel 1: Computación Paralela (asume conocimiento mínimo)**

"Computación paralela ejecuta múltiples instrucciones simultáneamente. Las GPUs, a diferencia de CPUs, contienen miles de pequeños cores capaces de ejecutar el mismo código en diferentes datos. Este modelo se llama 'SIMT' (Single Instruction, Multiple Threads)."

*¿Por qué esto?* Estableces por qué las GPUs son especiales.

**Nivel 2: Kernels GPU Específicamente**

"Un kernel GPU es una función C/CUDA que se ejecuta en todos estos threads. Cada thread accede a memoria global compartida, por lo cual debe ser cauteloso (coalesced access), y debe sincronizarse con otros threads (barriers). Estas restricciones hacen que escribir kernels eficientes sea arduo."

*¿Por qué esto?* Estableces el problema de dominio que tu investigación aborda.

**Nivel 3: Generación Automática de Código**

"Dados estos desafíos, investigadores han explorado usar modelos de lenguaje para generar kernels automáticamente. Un modelo de lenguaje es una red neuronal entrenada para predecir el siguiente token en una secuencia. Generamos kernels token-por-token hasta que el modelo predice un token de 'fin'."

*¿Por qué esto?* Estableces tu enfoque de solución.

**Nivel 4: Restricciones Gramaticales (Tu Contexto Específico)**

"Sin embargo, modelos sin restricciones frecuentemente generan sintaxis inválida. Una restricción gramatical limita los tokens válidos en cada paso basado en una gramática formal. Por ejemplo, si la gramática dice 'un for loop debe tener un contador inicializado', el modelo no puede generar un for loop sin antes generar la inicialización."

*¿Por qué esto?* Estableces tu específica contribución.

Observa cómo cada nivel se construye sobre el anterior. Un lector sin conocimiento puede comenzar desde Nivel 1 y gradualmente entender el contexto.

---

## Defina Conceptos Críticos con Precisión

Diferentes campos definen "restricción gramatical" de maneras ligeramente diferentes. Establece qué significa en **tu investigación**.

### Estructura para Definiciones

1. **Definición de diccionario / General**
   "Una gramática es un conjunto de reglas que define un lenguaje formal."

2. **Definición técnica específica**
   "En nuestro contexto, una gramática es una especificación EBNF (Extended Backus-Naur Form) que define qué secuencias de tokens forman un CUDA kernel válido."

3. **Ejemplo concreto**
   ```
   kernel ::= "void" identifier "(" parameters ")" "{" statements "}"
   parameters ::= parameter ("," parameter)*
   statements ::= statement (";" statement)*
   ```

4. **Implicación para tu investigación**
   "Esto significa que nuestro sistema no puede generar un kernel sin la palabra clave 'void' al inicio, asegurando conformidad sintáctica desde el primer token."

### Evita Ambigüedad

- ❌ Débil: "Usamos restricciones para asegurar seguridad."
  - ¿Qué significa "seguridad"? (Sintáctica, funcional, memoria?)

- ✅ Fuerte: "Usamos restricciones gramaticales para asegurar que kernels generados compilarán exitosamente, eliminando la clase de errores de sintaxis (aunque no errores funcionales como race conditions)."
  - Claro qué tipo de seguridad y qué no cubre.

> 💡 **Concepto clave:** Las definiciones precisas previenen confusión después. Un lector que entienda exactamente qué significa cada término puede seguir tu lógica.

---

## Conectar el Marco Teórico a tu Pregunta de Investigación

El marco teórico no debe flotar en el vacío. Debe conectar gradualmente a tu pregunta específica.

### Estrategia: El "Narrowing Statement"

En la transición entre cada nivel de tu marco, escribe una oración de conexión:

**Después de explicar computación paralela:**
"Estas capacidades de paralelismo masivo hacen las GPUs atractivas para científico computacional, pero también introducen desafíos únicos..."

**Después de explicar kernels GPU:**
"Estos desafíos motivan el desarrollo de herramientas automáticas. Una promisora dirección es usar modelos de lenguaje..."

**Después de explicar modelos de lenguaje:**
"Sin embargo, los modelos sin restricciones frequentemente generan código inválido. Para abordar esto, investigadores han propuesto restricciones gramaticales..."

**Después de explicar restricciones gramaticales:**
"A pesar de su promesa, preguntas críticas permanecen sin respuesta: ¿cuál es el costo computacional exacto? ¿Cómo variamos según tipo de restricción? ¿Realmente mejoramos la seguridad funcional? Estas preguntas constituyen el enfoque de nuestra investigación."

Observa cómo cada oración de transición lleva gradualmente al lector de conceptos generales a tu pregunta específica.

---

## Consideraciones Éticas en tu Marco Teórico

Tu marco teórico también debería mencionar consideraciones éticas relevantes. Esto no es un "bonus"; es esperado en investigación moderna.

### Qué Incluir

Para investigación sobre generación de código y AI:

**1. Sesgo en datos de entrenamiento**
"Los modelos de lenguaje se entrenan en corpus de código públicos como GitHub. Estos corpus reflejan los sesgos de la comunidad de desarrolladores—sobre-representación de ciertos lenguajes, estilos de codificación, y patrones de seguridad. Los kernels generados probablemente reflejarán estos sesgos."

**2. Privacidad**
"El código de entrenamiento puede contener lógica propietaria o patrones específicos de empresas. Cuando generamos nuevos kernels, ¿generamos código similar al material privado de entrenamiento?"

**3. Responsabilidad**
"Si alguien usa un kernel generado en producción sin verificación rigurosa, y ello causa un crash o pérdida de datos, ¿quién es responsable? ¿El investigador que propuso el método? ¿El usuario?"

No necesitas resolver estas (son preguntas abiertas en el campo), pero deberías reconocerlas.

---

## Marco Teórico Iterativo: Escribir, Feedback, Revisar

A diferencia de la revisión bibliográfica (que puedes escribir relativamente pasivamente), el marco teórico requiere iteración activa.

### Ciclo Típico

**Primer borrador:** Escribes definiciones y conceptos como los entiendes.

**Feedback de supervisor:** "Esto es muy técnico. Asume que el lector no está familiarizado con CUDA. Simplifica. También, ¿cómo conecta esta sección a tu pregunta? Agregue una oración de transición."

**Revisión:** Simplificas, agregas contexto, escribes oraciones de transición.

**Feedback siguiente:** "Mejor, pero ahora siento que te quedaste fuera de detalles técnicos que son críticos. Explica más específicamente cómo funcionan las restricciones gramaticales."

**Revisión nuevamente:** Profundizas ciertos temas, mantienes otros simplificados.

Este ciclo típicamente toma 3-5 iteraciones para un marco sólido. **Es normal.** El marco teórico es donde enseñas; enseñanza requiere refinamiento.

---

## Longitud: Cuánto es Suficiente

Una pregunta común: "¿Cuánto debería ser mi marco teórico?"

**Pauta:** El marco teórico debería ser lo suficientemente largo para que un lector inteligente pero sin especialidad entienda tu investigación, pero no tan largo que sea tedioso.

- **Demasiado corto** (2-3 páginas): Probablemente omites conceptos críticos. El lector se pierde en tu metodología.
- **Correcto** (8-12 páginas): Suficiente para explicar conceptos progresivamente sin sobre-detalle.
- **Demasiado largo** (20+ páginas): Probablemente includes material que pertenece a "revisión bibliográfica" o está sobre-detallado.

### Cómo Saber si es Suficiente

Pasa tu marco a un colega (o amigo) que **no** es especialista. Pídele que lea y responda:
- ¿Entiendes qué es un kernel GPU después de leer esto?
- ¿Entiendes qué hace una restricción gramatical?
- ¿Entiende la conexión entre estos conceptos?

Si responde "sí" a todas, tu marco es probablemente suficiente. Si responde "más o menos" a algo, tienes trabajo que hacer.

---

## Resumen

En esta lectura exploramos:

- **Marco teórico vs. revisión bibliográfica:** El primero explica conceptos; el segundo critica trabajos anteriores
- **Estructura progresiva:** De conceptos generales a específicos, funneling gradualmente
- **Definiciones precisas:** Especialmente importante para términos que el campo usa vagamente
- **Narrowing statements:** Transiciones que conectan cada sección a tu pregunta
- **Consideraciones éticas:** Reconoce y discute implicaciones éticas de tu investigación

---

## Ejercicios y Reflexión

### Preguntas de comprensión

1. ¿Cuál es la diferencia entre marco teórico y revisión bibliográfica? ¿Cuál necesita ser más crítico?

2. ¿Por qué es importante que el marco teórico sea "progresivo" de conceptos generales a específicos?

3. Proporciona un ejemplo de cómo podrías escribir un "narrowing statement" para conectar dos secciones de tu marco.

### Ejercicio práctico

**Tarea 1: Outline de Marco Teórico (30 minutos)**

Basado en tu pregunta de investigación, escribe un outline para tu marco teórico. Debería ser:
- Al menos 4 niveles de profundidad (general a específico)
- Cada nivel con 1-2 oraciones de descripción
- Oraciones de transición conectando niveles

Ejemplo:
```
Nivel 1: Computación paralela
   - Definición, por qué GPU, diferencia con CPU
   → Transición: "Estas capacidades motivan aplicaciones en..."

Nivel 2: Kernels GPU específicamente
   - Qué es kernel, cómo se ejecuta, restricciones
   → Transición: "Escribir kernels eficientes es desafiante, por lo cual..."
```

**Tarea 2: Escribir Definición Precisa (20 minutos)**

Selecciona un concepto crítico para tu investigación. Escribe una definición que incluya:
1. Definición general/diccionario
2. Definición técnica específica a tu contexto
3. Un ejemplo concreto
4. Implicación para tu investigación

**Tarea 3: Primera Sección de Marco (45 minutos)**

Escribe la primera sección (Nivel 1) de tu marco teórico (~500-800 palabras). Debería:
- Ser accesible a un lector sin especialidad
- Introducir conceptos clave
- Terminar con una oración de transición hacia la siguiente sección

### Para pensar

> *¿A quién es "ideal" lector de tu marco teórico? ¿Un profesor de IA? ¿Un ingeniero de hardware? ¿Un estudiante de grado? Mantén este lector ideal en mente mientras escribes.*

---

## Próximos pasos

Con tu marco teórico en lugar, estás listo para la sección más crítica de tu tesis: **Metodología**. Aquí describirás exactamente cómo realizaste tu investigación, permitiendo a otros reproducir tu trabajo. Verás cómo el marco teórico que construiste sienta la base para explicar métodos complejos con claridad.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - TC3002B*
