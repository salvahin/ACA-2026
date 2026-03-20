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

# Limitaciones Context-Sensitive: Más Allá de CFG

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    # Dependencias ya incluidas en Colab
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```


```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Identificar restricciones que las CFGs NO pueden expresar (declaración antes de uso, tipos, scope)
- Aplicar el Pumping Lemma para demostrar que un lenguaje no es context-free
- Diseñar estrategias de análisis multi-pasada para validar restricciones semánticas
- Implementar tabla de símbolos con gestión de scope para verificar visibilidad de variables
- Explicar por qué XGrammar combina CFG con análisis semántico post-parsing
```

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/09-limitaciones-context-sensitive.ipynb)
```


## El Muro: Lo que CFG No Puede Expresar

Hemos construido máquinas poderosas. Los DFAs reconocen lenguajes regulares. Los PDAs y CFGs reconocen lenguajes libres de contexto. Pero hay un mundo de restricciones que **no pueden** expresarse con una sola CFG.

Ejemplos en programación real:

```
1. Variables deben estar declaradas
   { int x; y = x + 1; }  ← válido
   { y = x + 1; }         ← ERROR: x no declarada

2. Tipos deben coincidir
   int x = "hello";       ← ERROR: incompatible
   int x = 42;            ← OK

3. Scope
   { { int x; } y = x; }  ← ERROR: x fuera de alcance
   { int x; { y = x; } }  ← OK

4. Indentación coherente (Python)
   if x > 0:
       y = 1
      z = 2              ← ERROR: inconsistente

5. Balanceo de recursos
   { malloc(); ... free(); }  ← cada malloc con su free

¿Por qué CFG no puede expresarlos?
```

## Teoría: Límites Computacionales

### Repaso: Jerarquía de Chomsky

```
Tipo 0: Recursivamente Enumerable    (máquinas de Turing)
         ↑
         └─ Sin restricción, computable
Tipo 1: Context-Sensitive            (autómatas linealmente acotados)
         ↑
         └─ αAβ → αγβ, |γ| >= |A|
Tipo 2: Context-Free                 (autómatas de pila)
         ↑
         └─ A → β
Tipo 3: Regular                       (autómatas finitos)
         ↓
         └─ Muy restrictivo
```

### Por qué CFG (Tipo 2) No Puede

**Teorema del Pumping Lemma para CFGs**:

Si L es context-free, entonces existe constante n (la "longitud de pumping") tal que para todo string s en L con |s| >= n:
  s = uvwxy donde:
    - |vwx| <= n
    - |vx| > 0
    - Para todo i >= 0: uv^i wx^i y está en L

**Intuición**: En cualquier árbol de derivación suficientemente grande, debe haber un no-terminal que se repite en un camino. Esta repetición permite "bombear" (repetir) partes del string.

```
Visualización del Pumping Lemma:

       S
      / \
     /   \
    A     y      ← parte "y" (después del bombeo)
   /|\
  / | \
 u  A  x         ← parte "v" y "x" (bombeables)
    |
    w            ← parte "w" (centro)

La A repetida permite:
- i=0: u w y       (quitar v y x)
- i=1: u v w x y   (normal)
- i=2: u vv w xx y (repetir v y x)
- etc.
```

**Implicación**: CFG **no puede contar con dos contadores independientes**. Solo puede guardar en la pila, que es LIFO (último en, primero afuera).

### Cómo Usar el Pumping Lemma (Estructura de Prueba)

Para demostrar que un lenguaje L **NO es context-free**:

```
Estructura de prueba por contradicción:

1. SUPONER que L es context-free
2. Por el Pumping Lemma, existe constante n
3. ELEGIR un string s ∈ L con |s| >= n
   (Elegir estratégicamente para que cualquier división falle)
4. Por el lema, s = uvwxy con las condiciones
5. CONSIDERAR todos los casos posibles de división
6. MOSTRAR que para algún i, uv^i wx^i y ∉ L
7. CONTRADICCIÓN → L no es context-free
```

### Ejemplo Completo: a^n b^n c^n No es Context-Free

**Demostración formal:**

```
Paso 1: Supongamos que L = {a^n b^n c^n | n >= 0} es context-free.

Paso 2: Por el Pumping Lemma, existe constante de bombeo p.

Paso 3: Elegimos s = a^p b^p c^p
        - s está en L (tiene p de cada letra)
        - |s| = 3p >= p

Paso 4: Por el lema, s = uvwxy donde |vwx| <= p y |vx| > 0

Paso 5: Analizamos dónde pueden estar v y x:

  Caso A: vwx está completamente dentro de las a's
          vwx = a...a
          Entonces v y x solo contienen a's
          Al bombear (i=2): tenemos más a's que b's y c's
          → a^(p+k) b^p c^p ∉ L  ✗

  Caso B: vwx está completamente dentro de las b's
          Similar al caso A
          → a^p b^(p+k) c^p ∉ L  ✗

  Caso C: vwx está completamente dentro de las c's
          Similar
          → a^p b^p c^(p+k) ∉ L  ✗

  Caso D: vwx cruza frontera a-b (contiene a's y b's pero no c's)
          Al bombear: cambian a's y b's, pero no c's
          → Los tres conteos ya no son iguales  ✗

  Caso E: vwx cruza frontera b-c (contiene b's y c's pero no a's)
          Similar
          → Los tres conteos ya no son iguales  ✗

  Caso F: vwx contiene a's, b's y c's
          Imposible porque |vwx| <= p, y las a's, b's, c's
          están separadas por p posiciones cada una.
          Si vwx contuviera los tres, |vwx| > p. ✗

Paso 6: En todos los casos posibles, al bombear obtenemos
        un string que NO está en L.

Paso 7: Contradicción. Por lo tanto, L no es context-free. ∎
```

```{admonition} Error Común en Pruebas de Pumping Lemma
:class: warning

**Incorrecto:** "Elijo v = a^k y muestro que falla"
- ¡El adversario elige la división, no tú!

**Correcto:** "Para CUALQUIER división posible de s en uvwxy..."
- Debes considerar todos los casos donde v y x podrían estar
```

### Por Qué la Intuición Funciona

```
CFG y su PDA equivalente tienen UNA pila.

Para a^n b^n c^n necesitamos:
  1. Contar a's (push n veces)
  2. Verificar b's = a's (pop n veces)
  3. Verificar c's = a's (¡pero la pila ya está vacía!)

Con una pila:
  - Podemos hacer a^n b^n (push a's, pop con b's)
  - Podemos hacer b^n c^n (push b's, pop con c's)
  - NO podemos verificar a^n = b^n = c^n simultáneamente

Necesitaríamos DOS pilas independientes (Type 1 power).
```

### Ejemplo que CFG SÍ Puede Hacer

```
Contraste: a^n b^n (dos contadores relacionados)

Gramática:
  S → aSb | ε

Derivación:
  S → aSb → aaSbb → aaabbb

¿Por qué funciona?
  - Una pila es suficiente
  - Push a's, luego pop verificando b's

El Pumping Lemma NO da contradicción aquí porque
podemos elegir v = a, x = b, y bombear manteniendo igualdad.
```

## Problemas Prácticos: Declaración Antes de Uso

El problema clásico en lenguajes de programación.

### Intentar Expresar en CFG

```
Especificación:
  Toda variable usada debe estar declarada antes

Intento 1: Añadir a la gramática
  program = (declaration | statement)*
  declaration = type ID ";"
  statement = ID "=" expr ";"  (usa ID)

¿Funciona?
  Sí parsea válidos: int x; x = 5;
  Pero TAMBIÉN parsea inválidos: x = 5; int x;

La gramática no puede verificar que el ID en statement
fue previamente declarado.
```

### Solución: Análisis Semántico

```
Fase 1: CFG Parser
  program = (decl | stmt)*

  Parsea tanto:
    ✓ int x; x = 5;
    ✗ x = 5; int x;  ← parser acepta

Fase 2: Análisis Semántico
  1. Pasar 1: Recolectar declaraciones
     declared_vars = { }
     Para cada nodo en AST:
       Si es declaration(type, name):
         declared_vars.add(name)

  2. Pasar 2: Verificar uso
     Para cada nodo en AST:
       Si es statement(var, ...):
         Si var no en declared_vars:
           ERROR: variable no declarada

  Resultado:
    ✓ int x; x = 5;       aceptado
    ✗ x = 5; int x;       rechazado (error semántico)
```

Este es el **enfoque estándar** en lenguajes de programación.

## Problema Type-Checking

Otro clásico que CFG no puede expresar.

### CFG: Solo Estructura

```
Gramática:
  assign = ID "=" expr

Parsea:
  x = 5         ✓
  x = "hello"   ✓
  x = obj       ✓

CFG ve solo estructura, no tipos.

¿Tipo de x?
¿Tipo de 5, "hello", obj?
¿Son compatibles?

CFG no sabe, no puede saber.
```

### Solución: Tabla de Símbolos + Type Inference

```
Fase 1: Parsing (CFG)
  Construir AST

Fase 2: Recolección de Tipos (pasada 1)
  Recorrer AST, anotar tipos de literales y variables:
    5 → int
    "hello" → string
    obj → object (del contexto)

Fase 3: Type Checking (pasada 2)
  Para cada nodo Assign(x, expr):
    type_x = symbol_table[x]
    type_expr = infer_type(expr)
    Si type_x != type_expr:
      ERROR: tipo incompatible

Resultado:
  ✓ x: int = 5
  ✓ s: string = "hello"
  ✗ x: int = "hello"       ERROR
```

## Problema de Alcance (Scope)

Variables tienen "alcances": contextos donde son válidas.

```
Válido:
  {
    int x;
    { y = x; }      ← x visible en inner scope
  }

Inválido:
  {
    { int x; }
    y = x;          ← x no visible fuera del scope
  }
```

### CFG No Puede

```
CFG puede expresar estructura de bloques:
  block = "{" statement* "}"

Pero no puede expresar:
  "Una variable es válida solo dentro del scope donde se declaró"

¿Por qué?
  Scope es una propiedad semántica.
  Depende del contexto de declaración.
  CFG ve solo sintaxis, no semántica.
```

### Solución: Symbol Table con Scope Management

```
Algoritmo: Verificación de Scope

1. Mantener stack de scopes
   scopes = []

2. Entrar a block
   scopes.push(new_scope)

3. Encontrar declaración
   scopes[-1].add(name)

4. Encontrar uso
   Para scope en scopes (desde final):
     Si name en scope:
       OK, variable visible
   Si no encontrado en ningún scope:
       ERROR: variable no visible

5. Salir de block
   scopes.pop()

Ejemplo:
  Entrar block:     scopes = [{}]
  int x;            scopes = [{x}]
  Entrar inner:     scopes = [{x}, {}]
  y = x;            buscar x, encontrado en scopes[0] ✓
  Salir inner:      scopes = [{x}]
  Salir block:      scopes = []
```

## Problema de Indentación (Python-style)

Python y otros lenguajes usan **indentación** para significado sintáctico.

```{code-cell} ipython3
x = 5  # Example value to demonstrate control flow
if x > 0:
    y = 1
    z = 2
w = 3

print(f"x = {x}, y = {y}, z = {z}, w = {w}")
```

La indentación determina qué va en el bloque if.

### ¿Por qué CFG Falla?

```text
CFG especificaría:
  if_stmt = "if" expr ":" statement+

Pero ¿cómo sabe cuándo terminan los statements?
  Por indentación. Cuando la indentación vuelve a disminuir.

CFG no puede expresar:
  "Todos los statements con indentación >= N"

Es una restricción sobre **tokens**, no sobre estructura.
```

### Solución: Pre-procesamiento o Tokens Especiales

```
Opción 1: Insertar tokens de control
  Léxer produce:
    if x > 0:
    INDENT           ← generado por léxer (no en código fuente)
      y = 1
      z = 2
    DEDENT           ← generado por léxer

  Gramática puede usar:
    if_stmt = "if" expr ":" INDENT statement+ DEDENT

  El parser ve tokens explícitos, no confusión de espacios.

Opción 2: Preprocesar
  Convertir indentación a estructuras explícitas:
    if x > 0: { y = 1; z = 2; }

Ambas opciones: CFG puede parsear si le ayudamos
```

## Problema de Balanceo de Recursos

Algunos lenguajes permiten recursos que deben ser "balanceados".

```
Válido:
  mem = malloc(100)
  use(mem)
  free(mem)

Inválido:
  mem = malloc(100)        ← sin correspondiente free
  use(mem)

Inválido:
  mem = malloc(100)
  use(mem)
  free(mem)
  free(mem)                ← two frees
```

### ¿Por qué CFG Falla?

```
Este es realmente un problema Type 1 (Context-Sensitive).

Necesita contar:
  #malloc's vs #free's
  Orden temporal
  Scope

No es expresable en CFG.
```

### Solución: Análisis de Flujo de Control

```
Después del parsing:
  1. Build control flow graph (CFG - diferente del CFG de Chomsky)
  2. Data flow analysis
  3. Verificar que cada malloc tiene correspondiente free
  4. Verificar orden

Herramientas reales:
  - Valgrind, AddressSanitizer (para memoria)
  - Rust ownership/borrow checker (para garantías)
```

## Estrategia General: Análisis Multi-Pasada

La solución práctica es usar **múltiples pasadas**:

```
Pasada 1: Lexing
  Caracteres → Tokens

Pasada 2: Parsing (CFG)
  Tokens → AST (estructura)

Pasada 3: Análisis Semántico
  - Construcción de tabla de símbolos
  - Type checking
  - Scope analysis
  - Validación de restricciones semánticas

Pasada 4: Optimización + Generación de Código
  - Análisis de flujo
  - Optimizaciones
  - Emitir código ejecutable
```

:::{figure} diagrams/compiler_multipass.png
:name: fig-compiler-multipass
:alt: Diagrama de compilación multi-pasada mostrando las fases de lexing, parsing, análisis semántico y generación de código
:align: center
:width: 90%

**Figura 1:** Arquitectura de compilación multi-pasada. Mientras que CFG solo puede verificar sintaxis (pasadas 1-2), las restricciones semánticas (tipo de datos, scope, declaración antes de uso) requieren pasadas adicionales de análisis. Cada pasada del compilador valida diferentes aspectos del programa.
:::

Cada pasada añade más validación.

## Ejemplo: Kernel CUDA vs XGrammar

### CUDA (Lenguaje Completo)

```cuda
__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2;
    }
}
```

Para verificar esto válido:
```
CFG: ✓ Estructura sintáctica
Semántica:
  ✓ blockIdx, blockDim, threadIdx son variables built-in válidas
  ✓ Tipos correctos (int × int = int)
  ✓ data es parámetro válido
  ✓ Acceso a índice válido
  ✓ Función kernel valida según CUDA semantics
```

Todo esto **más allá** de CFG.

### XGrammar para Kernels GPU

```
XGrammar especifica estructura de GPU kernels:
  kernel_def = "@triton.jit" newline
               "def" name "(" params ")" "->" type ":" block

Validaciones que XGrammar CFG hace:
  ✓ Sintaxis de función
  ✓ Parámetros bien formados
  ✓ Bloque de código bien formado

Validaciones que requieren análisis semántico:
  ✓ Parámetros tienen tipos válidos
  ✓ Block_id, thread_id usados correctamente
  ✓ Índices de acceso dentro de límites
  ✓ Memory coherence en GPU (sincronización)
  ✓ Threadgroup colabora correctamente

Implementación:
  XGrammar = CFG (estructura)
  + análisis post-parsing (semántica)
```

## Error Recovery en Parsers

Cuando el input viola la gramática, hay dos opciones:

### Opción 1: Fallar y Reportar

```
Input:
  x = 5 6;

Lexer: [ID(x), ASSIGN, INT(5), INT(6), SEMICOLON]

Parser:
  Espera expr después de =
  Ve INT(5) ✓, acepta
  Ahora espera ";"
  Ve INT(6) ✗

  Error: Expected ";", got INT(6)
  Ubicación: línea 1, columna 7
  Contexto: En asignación
```

### Opción 2: Recuperarse (Error Recovery)

Algunos parsers intentan continuar:

```
Técnica 1: Skip Tokens
  Cuando error, saltear tokens hasta encontrar sincronización
  Punto de sincronización: ";"

  Continuar parsear desde siguiente ";"

Técnica 2: Insert Tokens
  Asumir que falta ";"
  Insertar implícitamente
  x = 5 6;  → x = 5; 6;

Técnica 3: Reparse Local
  Cuando error, retroceder y reinterpret

Ventaja: Encontrar múltiples errores de una vez
Desventaja: Mensajes de error a veces confusos
```

## Mejora de Mensajes de Error

XGrammar puede usar información de la gramática para mejorar errors:

```
Técnica 1: Lookahead
  Si esperamos ID pero vemos NUM,
  Mensaje: "Expected identifier, got number"

Técnica 2: FOLLOW Sets
  Si vemos token no esperado,
  Mostrar qué era posible:
  "Expected one of: {+, -, *, /, ), ;}"

Técnica 3: Context
  De qué regla vinimos, qué estamos parseando:
  "In expression after '+': expected term, got ';'"

Técnica 4: Suggestion
  Si error es "typo" común:
  "Did you mean ':=' instead of '='?"
```

## Resumen de Limitaciones y Soluciones

```
╔════════════════════╦══════════════╦═══════════════════════════╗
║ Restricción        ║ Tipo Chomsky ║ Solución                  ║
╠════════════════════╬══════════════╬═══════════════════════════╣
║ Estructura básica  ║ Tipo 2 (CFG) ║ Gramática, parsing        ║
║ Scope              ║ Tipo 1+      ║ Tabla de símbolos         ║
║ Tipos              ║ Tipo 1+      ║ Type inference, checking  ║
║ Declaración antes  ║ Tipo 1+      ║ Análisis semántico        ║
║ Indentación        ║ Tipo 2       ║ Tokens de control (INDENT)║
║ Balance recursos   ║ Tipo 1+      ║ Análisis de flujo         ║
║ Optimización       ║ N/A          ║ Transformaciones post-AST ║
╚════════════════════╩══════════════╩═══════════════════════════╝
```

```{admonition} 🎯 Conexión con el Proyecto
:class: important
En XGrammar para kernels GPU, la CFG valida la estructura sintáctica del código, pero restricciones como "cada variable de thread debe estar dentro del rango válido" o "memoria compartida correctamente sincronizada" requieren análisis semántico adicional. Esta separación permite que la gramática sea simple y reutilizable, mientras la semántica se personaliza por caso de uso.
```

```{admonition} Resumen
:class: important
**Conceptos clave:**
- Las CFGs NO pueden expresar: declaración antes de uso, tipos, scope, balanceo de recursos
- El Pumping Lemma demuestra que lenguajes como a^n b^n c^n no son context-free
- La solución práctica es análisis multi-pasada: CFG para sintaxis + código para semántica
- Tabla de símbolos con stack de scopes verifica visibilidad y declaración de variables
- Error recovery y mensajes contextuales mejoran la experiencia del desarrollador

**Para la siguiente lectura:**
Concluiremos el módulo comparando parser generators industriales (ANTLR, Tree-Sitter, Bison) y reflexionando sobre lecciones aprendidas.
```

## Ejercicios

1. **Pumping Lemma**: Demuestra que a^n b^n c^n no es CFG.

2. **Diseño Multi-Pasada**: Para una asignación a un array:
   ```
   a[i] = value;
   ```
   ¿Qué validaciones necesita cada pasada?

3. **Scope Analysis**: Traza el stack de scopes:
   ```
   { int x;
     { int y;
       { y = x; }      ← ¿x visible?
     }
   }
   ```

4. **Error Recovery**: Propón estrategia para:
   ```
   int x = ;    ← falta expr
   y = x + ;    ← falta operando
   ```

5. **Type Checking**: Qué errores semánticos identifica:
   ```
   int x = "hello";
   float y = 3 + 4.5;
   string[] arr = new int[10];
   ```

## Preguntas de Reflexión

- ¿Hay lenguajes de programación modernos que no requieren análisis semántico más allá de CFG?
- ¿Cómo el use de "tipos" en la especificación afecta qué se necesita para validación?
- En GPU kernels específicamente, ¿qué restricciones semánticas son críticas que no CFG puede expresar?
- ¿Es posible extender CFG a Tipo 1 para kernels GPU? ¿Cuál sería el costo?

---

## Referencias

- Bar-Hillel, Y., Perles, M., & Shamir, E. (1961). [On Formal Properties of Simple Phrase Structure Grammars](https://doi.org/10.1524/stuf.1961.14.14.143). Zeitschrift für Phonetik.
- Chomsky, N. (1956). Three Models for the Description of Language. IRE Transactions on Information Theory.
- Sipser, M. (2012). Introduction to the Theory of Computation (3rd ed.). Cengage Learning.
