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

# Gramáticas Libres de Contexto: Estructurando el Caos

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/04-gramaticas-libres-contexto.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q plotly
print('Dependencies installed!')
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Definir los componentes de una gramática libre de contexto (CFG)
- Realizar derivaciones leftmost y rightmost para generar strings
- Construir árboles de análisis sintáctico a partir de derivaciones
- Identificar y resolver ambigüedades en gramáticas
- Diseñar gramáticas CFG que codifiquen precedencia de operadores correctamente
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[Context Free Grammar CFG (Neso Academy)](https://www.youtube.com/watch?v=hOofv-Yy1k0)** - Producciones arbóreas (Parse Trees) dibujadas nodo por nodo con explicaciones claras.
```

## ¿Por qué Necesitamos Algo Más?

Hasta ahora hemos trabajado con lenguajes regulares y DFAs. Estos son perfectos para patrones planos: identificadores, números, palabras clave.

Pero, ¿qué sobre **estructura**? Considera una expresión matemática:

```
2 + 3 * 4

¿Es (2 + 3) * 4 = 20?
¿O es 2 + (3 * 4) = 14?
```

La respuesta depende de la **precedencia de operadores**, que crea una **jerarquía anidada**. Similarmente:

```
if (x > 0) {
    for (int i = 0; i < 10; i++) {
        y = x + i;
    }
}
```

Los bloques están **anidados**. Los paréntesis deben **balancearse**.

Los DFAs son **memoryless** - solo tienen un número fijo de estados. No pueden "recordar" que vieron un paréntesis abierto y esperar el cierre. **Necesitamos una pila.**

Aquí entran las **Gramáticas Libres de Contexto (CFG)** y sus máquinas reconocedoras: **Autómatas de Pila (PDA)**.

## Gramáticas Libres de Contexto: Definición

Una **Gramática Libre de Contexto** es una forma sistemática de generar un lenguaje.

![**Figura 1:** Jerarquía de Chomsky mostrando los cuatro tipos de gramáticas.](diagrams/chomsky_hierarchy.png)

***Figura 1:** Jerarquía de Chomsky mostrando los cuatro tipos de gramáticas.*


### Componentes

Una CFG es una tupla: **G = (V, Σ, R, S)**

- **V**: No-terminales (símbolos intermedios, no aparecen en output final)
- **Σ**: Terminales (símbolos finales, caracteres/tokens)
- **R**: Reglas de producción (cómo expandir no-terminales)
- **S**: Símbolo inicial

### Notación

Las reglas se escriben:
```
A → α  (A es no-terminal, α es secuencia de terminales y no-terminales)
A → β | γ  (múltiples opciones)
```

:::{figure} diagrams/chomsky_detail.png
:name: fig-chomsky-detail
:alt: Detalles de la jerarquía de Chomsky con ejemplos de cada tipo
:align: center
:width: 90%

**Figura 2:** Análisis detallado de la jerarquía de Chomsky: restricciones progresivas en las reglas de producción y los lenguajes que cada tipo puede reconocer.
:::

### Ejemplo 1: Expresiones Aritméticas Simples

```
Terminales: {+, *, (, ), número}
No-terminales: {E, T, F}
Inicial: E

Reglas:
E → E + T | T
T → T * F | F
F → ( E ) | número
```

Aquí:
- **E** (Expresión) maneja suma (menor precedencia)
- **T** (Término) maneja multiplicación (mayor precedencia)
- **F** (Factor) maneja paréntesis y números (máxima precedencia)

La **estructura de la gramática** impone orden de operaciones.

### Ejemplo 2: Paréntesis Balanceados

```
Terminales: {(, )}
No-terminales: {S}
Inicial: S

Reglas:
S → ( S ) S | ε

Interpretación:
- Un '(' abre, un ')' cierra
- Dentro podemos tener más S (anidamiento)
- Podemos terminar en cualquier momento (ε)

Derivación para "(())":
S → (S)S → (S)ε → ((S)S)ε → (()S)ε → (()ε)ε → (())
```

### Ejemplo 3: Un Lenguaje Simple JSON-like

```
Terminales: {:, ,, [, ], {, }, string, number}
No-terminales: {Value, Object, Array, Pair}
Inicial: Value

Reglas:
Value → Object | Array | string | number

Object → { } | { Pairs }
Pairs → Pair | Pair , Pairs

Array → [ ] | [ Values ]
Values → Value | Value , Values

Pair → string : Value
```

Este reconoce estructuras JSON válidas.

## Derivaciones: Cómo se Generan Strings

Una **derivación** es una secuencia de pasos aplicando reglas.

### Derivación Izquierdista (Leftmost)

Siempre expandimos el no-terminal **más a la izquierda**:

```
Expresión: 2 + 3 * 4

Derivación desde E:
E
→ E + T           (expandir E, producción: E → E + T)
→ T + T           (expandir E izquierdo a T)
→ F + T           (expandir T izquierdo a F)
→ 2 + T           (expandir F a 2)
→ 2 + T * F       (expandir T a T * F)
→ 2 + F * F       (expandir T izquierdo a F)
→ 2 + 3 * F       (expandir F a 3)
→ 2 + 3 * 4       (expandir F a 4)

Resultado: 2 + 3 * 4
```

### Derivación Derechista (Rightmost)

Siempre expandimos el no-terminal **más a la derecha**:

```
E
→ E + T           (expandir E)
→ E + T * F       (expandir T)
→ E + T * 4       (expandir F)
→ E + 3 * 4       (expandir F)
→ 2 + 3 * 4       (expandir E)
```

Ambas derivaciones producen la **misma cadena final**, pero el **orden** de expansión es diferente. Los parsers generalmente usan leftmost o rightmost para construir estructuras.

## Árboles de Análisis Sintáctico (Parse Trees)

Un **árbol de análisis** representa la estructura jerárquica del string:

```
Para: 2 + 3 * 4

               E
             / | \
            E  +  T
            |     /|\
            T    T*F
            |    | |
            F    F 4
            |    |
            2    3

Altura representa anidamiento/precedencia.
Hojas son terminales.
Nodos internos son no-terminales.
```

El árbol muestra que 3*4 se agrupa antes de sumarlo con 2, lo que es correcto según precedencia.

### Construcción del Árbol

Durante parsing, el compilador construye este árbol bottom-up o top-down:

```
Bottom-up (como hacen muchos parsers):
1. Ver 2 → es F
2. F es T (aplicar regla T → F)
3. T es E (aplicar regla E → T)
4. Ver +
5. Ver 3 → es F
6. F es T (aplicar regla T → F)
7. Ver *
8. Ver 4 → es F
9. Reducir 4 → T (T → F)
10. Reducir T*F → T (T → T * F)
11. Ahora tenemos E + T
12. Reducir E + T → E (E → E + T)

Resultado: E (árbol construido)
```

## Ambigüedad en Gramáticas

Una gramática es **ambigua** si un string puede tener más de un árbol de análisis.

### Ejemplo de Ambigüedad

Considera esta gramática simplificada:

```
E → E + E | E * E | número

Para: 2 + 3 * 4

Árbol 1:               Árbol 2:
      E                     E
     /|\                   /|\
    E + E               E * E
    | /|\               /|   |
    2 E * E            E + E 4
      | | |            | | |
      3   4            2   3

Árbol 1: (2 + 3) * 4 = 20
Árbol 2: 2 + (3 * 4) = 14
```

Diferentes árboles, diferentes significados. **Malo**.

### Resolución de Ambigüedad

La gramática anterior es ambigua. La solución: **reescribir la gramática** para hacer precedencia explícita:

```
E → E + T | T
T → T * F | F
F → ( E ) | número

Ahora solo un árbol para 2 + 3 * 4:
        E
       /|\
      E + T
      |  /|\
      T T*F
      | | |
      F F 4
      | |
      2 3

Forzamos que * se agrupe más apretadamente que +.
```

**Las CFGs ambiguas son problemáticas** porque no sabemos qué interpretación es correcta. Los compiladores reales van a extremos para evitarlas.

```{admonition} ⚠️ Error Común
:class: warning
Los estudiantes frecuentemente escriben gramáticas ambiguas como `E → E + E | E * E | número` sin darse cuenta. Para evitar ambigüedad, estructura la gramática en niveles: expresión → término → factor, donde cada nivel representa un nivel de precedencia.
```

```{admonition} 🤔 Reflexiona
:class: hint
¿Por qué la estructura de la gramática (expresión → término → factor) determina automáticamente la precedencia de operadores? Piensa en cómo el árbol de análisis refleja el orden de evaluación.
```

## El Problema de la Precedencia

Para operadores, la forma de escribir la gramática determina la precedencia:

```
Baja precedencia (superior en árbol):
E → E + T | T

Mediana precedencia:
T → T * F | F

Alta precedencia (hojas):
F → ( E ) | número
```

:::{figure} diagrams/parse_tree_precedence.png
:name: fig-parse-tree-precedence
:alt: Árboles de análisis mostrando cómo la precedencia se codifica en la estructura gramatical
:align: center
:width: 90%

**Figura 3:** Estructura jerárquica de árboles de análisis para expresiones aritméticas: la profundidad del árbol refleja la precedencia de operadores. Operadores con mayor precedencia aparecen más profundos en el árbol.
:::

Esto es **tan importante** que dedicaremos toda una lectura a diseño de gramáticas.

## Ejemplo: Gramática Triton-like

Para kernels GPU, podríamos tener:

```
Terminales: {for, in, range, (, ), :, =, +, -, *, /, block_id, thread_id, ...}
No-terminales: {Program, Statement, LoopStmt, Assignment, Expr, ...}

Program → Statement*
Statement → Assignment | LoopStmt
LoopStmt → for ID in range ( Expr , Expr ) : Block
Assignment → ID = Expr
Expr → Expr + Term | Term
Term → Term * Factor | Factor
Factor → ( Expr ) | ID | NUMBER | thread_id . ID

Block → Statement | { Statement* }
```

Con esta gramática, podemos expresar:

```
for i in range(0, 10):
    x = block_id.x + i * 2
```

Y el parser construiría un árbol que el compilador después usaría para generar CUDA.

## Lenguajes Libres de Contexto vs Regulares

```
Lenguajes Regulares:
- Reconocidos por DFA
- No pueden contar
- No pueden anidar (excepto finito)
- Memoria: finita

Lenguajes Libres de Contexto:
- Reconocidos por PDA (pushdown automaton)
- Pueden contar (balanceo de paréntesis)
- Pueden anidar arbitrariamente
- Memoria: pila (unbounded)
```

### Lo que CFG **SÍ** puede expresar

✅ Paréntesis balanceados: (()())
✅ Expresiones anidadas: ((1+2)*3)
✅ Estructuras de datos: JSON completo
✅ Bloques de código con indentación (con cuidado)

### Lo que CFG **NO** puede expresar

❌ a^n b^n c^n (3 contadores, necesita Type 1)
❌ "Cada variable declarada antes de usarse" (necesita Type 1 o análisis semántico)
❌ "Tipos deben coincidir" (necesita análisis semántico)

**Solución práctica**: Combinamos CFG para estructura + análisis semántico para restricciones contextuales.

## Reconocimiento con PDA (Preview)

Un **autómata de pila** es como un DFA pero con una pila infinita:

```
Estados, transiciones como DFA, PERO:
- Puede leer y escribir la pila
- Transición: (estado, entrada, top_pila) → (nuevo_estado, push/pop/nada)

Ejemplo para paréntesis balanceados:
Estado: {q₀ (iniciar), q₁ (aceptar)}

δ(q₀, '(', ⊥) → (q₀, push '(')  // ⊥ es el fondo de la pila
δ(q₀, ')', '(') → (q₀, pop)
δ(q₀, ε, ⊥) → (q₁, nada)         // aceptar si pila vacía

Entrada: "()"
- (q₀, (), ⊥) → push ( → pila: [(
- (q₀, ), () → pop → pila: []
- ε-transición → (q₁, nada) → ACEPTADO
```

En la próxima lectura sobre parsing, veremos cómo los parsers reales (Earley, LALR) implementan la idea de PDA para CFGs.

```{admonition} 🎯 Conexión con el Proyecto
:class: important
Las CFGs son el corazón de XGrammar. Cuando defines una gramática para generar código Python o CUDA, estás especificando exactamente qué estructuras sintácticas son válidas. XGrammar compila esta CFG a un parser que guía al LLM token por token, asegurando que cada generación respete la estructura jerárquica del lenguaje.
```

```{admonition} Resumen
:class: important
**Conceptos clave:**
- Una CFG consta de terminales, no-terminales, reglas de producción y símbolo inicial
- Las derivaciones generan strings aplicando reglas; el árbol de análisis muestra la estructura
- La ambigüedad ocurre cuando un string tiene múltiples árboles; se resuelve reestructurando la gramática
- La precedencia de operadores se codifica mediante niveles jerárquicos en la gramática

**Para la siguiente lectura:**
Exploraremos BNF y EBNF, notaciones prácticas para escribir gramáticas, y compararemos diferentes estrategias de parsing (top-down vs bottom-up).
```

## Ejercicios

1. **Derivación**: Deriva "((()))" usando la gramática S → (S)S | ε

2. **Árbol de análisis**: Para "a + b * c", dibuja el árbol completo usando:
   ```
   E → E + T | T
   T → T * F | F
   F → a | b | c
   ```

3. **Ambigüedad**: ¿Esta gramática es ambigua?
   ```
   S → aS | Sa | ε
   ```
   Muestra los árboles para "aa".

4. **Diseño de gramática**: Escribe una CFG para direcciones IPv4 (X.X.X.X donde X es 0-255).
   Pista: Esto es difícil sin expresiones regulares. ¿Por qué?

5. **Lenguaje generado**: ¿Qué lenguaje genera?
   ```
   S → aSb | ε
   ```

## Preguntas de Reflexión

- ¿Por qué la estructura de la gramática determina la precedencia de operadores?
- ¿Cuál es la relación entre "árbol de análisis" y "significado" del programa?
- En XGrammar, cuando diseñamos una gramática para generar kernels, ¿cómo usamos precedencia para asegurar que el código generado es semánticamente válido?
- ¿Cuál es el trade-off entre "escribir gramáticas simples pero ambiguas" vs "gramáticas complejas pero no-ambiguas"?

## Conexión con Constrained Decoding en LLMs

```{admonition} 🔗 Aplicación en el proyecto
:class: tip

Las CFGs permiten generar estructuras complejas con LLMs mientras garantizamos corrección sintáctica:
- El árbol de análisis guía la generación jerárquica del modelo
- En cada nodo del árbol, el LLM solo puede elegir producciones válidas según la gramática
- Ejemplo: Generando código Python, si el parser está en "statement", el LLM no puede generar ")" sin abrir antes un paréntesis
- XGrammar usa PDAs para rastrear el estado del parsing y generar máscaras de tokens válidos en tiempo real
```

## Visualización de Árbol de Derivación

```{code-cell} ipython3
# Visualización de árbol de derivación
import plotly.graph_objects as go

# Árbol para: E → E + T → T + T → F + T → id + T → id + F → id + id
fig = go.Figure()

# Nodos del árbol
nodes = {
    'E': (0.5, 1.0),
    'E1': (0.25, 0.8), '+': (0.5, 0.8), 'T1': (0.75, 0.8),
    'T2': (0.25, 0.6), 'F1': (0.75, 0.6),
    'F2': (0.25, 0.4), 'id1': (0.75, 0.4),
    'id2': (0.25, 0.2)
}

labels = {'E': 'E', 'E1': 'E', '+': '+', 'T1': 'T', 'T2': 'T', 'F1': 'F', 'F2': 'F', 'id1': 'id', 'id2': 'id'}
colors = {'E': '#FF6B6B', 'E1': '#FF6B6B', 'T1': '#4ECDC4', 'T2': '#4ECDC4',
          'F1': '#45B7D1', 'F2': '#45B7D1', '+': '#FFE66D', 'id1': '#95E1A3', 'id2': '#95E1A3'}

# Dibujar conexiones
edges = [('E', 'E1'), ('E', '+'), ('E', 'T1'), ('E1', 'T2'), ('T1', 'F1'),
         ('T2', 'F2'), ('F1', 'id1'), ('F2', 'id2')]

for e1, e2 in edges:
    x0, y0 = nodes[e1]
    x1, y1 = nodes[e2]
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                             line=dict(color='gray', width=2), showlegend=False))

# Dibujar nodos
for name, (x, y) in nodes.items():
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text',
                             marker=dict(size=40, color=colors.get(name, 'white')),
                             text=[labels[name]], textposition='middle center',
                             textfont=dict(size=14, color='white'), showlegend=False))

fig.update_layout(title="Árbol de Derivación: id + id",
                  xaxis=dict(visible=False), yaxis=dict(visible=False), height=450)
fig.show()
```

## Ejercicio Práctico: Derivación Leftmost Paso a Paso

Realiza derivaciones leftmost para expresiones aritméticas y visualiza el árbol de análisis resultante.

```{code-cell} ipython3
# Gramática de expresiones aritméticas
# E → E + T | T
# T → T * F | F
# F → ( E ) | número

def leftmost_derivation(expression):
    """
    Simula derivación leftmost para expresiones aritméticas simples.
    Muestra cada paso de la derivación.
    """
    steps = []

    # Diccionario de producciones aplicadas para el ejemplo "2 + 3 * 4"
    if expression == "2 + 3 * 4":
        steps = [
            ("E", "Símbolo inicial"),
            ("E + T", "Expandir E → E + T (vemos + en la entrada)"),
            ("T + T", "Expandir E (izquierdo) → T"),
            ("F + T", "Expandir T (izquierdo) → F"),
            ("2 + T", "Expandir F → 2"),
            ("2 + T * F", "Expandir T → T * F (vemos * después del +)"),
            ("2 + F * F", "Expandir T (izquierdo) → F"),
            ("2 + 3 * F", "Expandir F → 3"),
            ("2 + 3 * 4", "Expandir F → 4"),
        ]
    elif expression == "5 * 6":
        steps = [
            ("E", "Símbolo inicial"),
            ("T", "Expandir E → T (no hay +)"),
            ("T * F", "Expandir T → T * F (vemos *)"),
            ("F * F", "Expandir T (izquierdo) → F"),
            ("5 * F", "Expandir F → 5"),
            ("5 * 6", "Expandir F → 6"),
        ]
    elif expression == "(1 + 2)":
        steps = [
            ("E", "Símbolo inicial"),
            ("T", "Expandir E → T"),
            ("F", "Expandir T → F"),
            ("( E )", "Expandir F → ( E )"),
            ("( E + T )", "Expandir E → E + T"),
            ("( T + T )", "Expandir E (izquierdo) → T"),
            ("( F + T )", "Expandir T → F"),
            ("( 1 + T )", "Expandir F → 1"),
            ("( 1 + F )", "Expandir T → F"),
            ("( 1 + 2 )", "Expandir F → 2"),
        ]
    else:
        return [("Expresión no implementada", "Prueba con: '2 + 3 * 4', '5 * 6', o '(1 + 2)'")]

    return steps

# Visualizar derivaciones
test_expressions = ["2 + 3 * 4", "5 * 6", "(1 + 2)"]

for expr in test_expressions:
    print(f"\n{'='*60}")
    print(f"Derivación leftmost para: {expr}")
    print('='*60)

    steps = leftmost_derivation(expr)
    for i, (form, explanation) in enumerate(steps):
        print(f"Paso {i}: {form:20s} | {explanation}")

    print(f"\n✓ Cadena generada: {expr}")

# Visualizar árbol de análisis (representación textual)
print(f"\n{'='*60}")
print("Árbol de análisis sintáctico para: 2 + 3 * 4")
print('='*60)
print("""
                E
              / | \\
             E  +  T
             |    /|\\
             T   T * F
             |   |   |
             F   F   4
             |   |
             2   3

Observa cómo:
- La multiplicación (3 * 4) se agrupa en un subárbol más profundo
- Esto fuerza que se evalúe ANTES que la suma
- La estructura del árbol = orden de evaluación = precedencia
""")
```

---

## Referencias

- Aho, A., Lam, M., Sethi, R., & Ullman, J. (2006). Compilers: Principles, Techniques, and Tools (2nd ed.). Pearson.
- Chomsky, N. (1959). [On Certain Formal Properties of Grammars](https://doi.org/10.1016/S0019-9958(59)90362-6). Information and Control.
- Sipser, M. (2012). Introduction to the Theory of Computation (3rd ed.). Cengage Learning.
