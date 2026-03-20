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

# Lenguajes Regulares y Autómatas Finitos Deterministas

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/02-lenguajes-regulares-dfas.ipynb)
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
- Escribir expresiones regulares para patrones comunes (identificadores, números, emails)
- Diseñar DFAs para reconocer lenguajes regulares específicos
- Identificar las limitaciones fundamentales de los lenguajes regulares
- Distinguir entre la fase léxica (DFAs) y la fase sintáctica (CFGs) en un compilador
- Explicar cómo los DFAs se usan para tokenización en XGrammar
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[Deterministic Finite Automata (Neso Academy)](https://www.youtube.com/watch?v=40i4PKpM0cE)** - Dibujos paso a paso diagramando estados y transiciones en DFAs.
```

```{admonition} 🔧 Herramientas Interactivas
:class: seealso

- **[AutomataVerse](https://automataverse.com/)** - Plataforma interactiva para dibujar grafos de máquinas de estados y simular paso a paso cómo una cadena es procesada.
- **[FSM Simulator](https://ivanzuzak.info/noam/webapps/fsm_simulator/)** - Define expresiones regulares y genera automáticamente la visualización del autómata correspondiente.
```

## Contexto
Esta lectura presenta los fundamentos del análisis léxico: cómo reconocer patrones usando autómatas finitos. Los DFAs son la base para construir **lexers** (primera fase del compilador), NO el parser completo.

## Introducción: La Potencia de lo Simple

Antes de construir parsers complejos, necesitamos dominar lo básico: reconocer **patrones simples**. Un identificador en C como `variable_name`, un número como `3.14`, una palabra clave como `if`. Todos estos son ejemplos de **lenguajes regulares**.

Los lenguajes regulares son los más simples en la jerarquía de Chomsky (Tipo 3), pero tienen una propiedad especial: pueden reconocerse **muy eficientemente** usando máquinas simples llamadas **autómatas finitos**. Esta es la base de la fase léxica en cualquier compilador.

## Expresiones Regulares: El Lenguaje de los Patrones

Una **expresión regular** (regex) es una forma compacta de describir patrones en strings. Si has usado alguna vez `grep` o validado un email con regex, ya has trabajado con esto.

### Notación Básica

```
a        → la letra 'a' exactamente
a|b      → 'a' OR 'b'
ab       → 'a' seguida de 'b' (concatenación)
a*       → cero o más 'a'
a+       → uno o más 'a'
a?       → cero o uno 'a'
[abc]    → cualquiera de 'a', 'b', 'c'
[a-z]    → cualquier letra minúscula
.        → cualquier carácter (en la mayoría de sistemas)
```

### Ejemplos Prácticos

```
Identificador:     [a-zA-Z_][a-zA-Z0-9_]*
Número entero:     -?[0-9]+
Número flotante:   -?[0-9]+(\.[0-9]+)?
Email simple:      [a-z]+@[a-z]+\.[a-z]+
Palabra clave:     (if|else|while|for)
```

**En contexto de GPU**: Imagine especificar patrones válidos en un kernel CUDA:

```
thread_id:  blockIdx\.x|blockIdx\.y|blockIdx\.z
kernel_name: [a-zA-Z_][a-zA-Z0-9_]*
```

### Composición

Las expresiones regulares son **composables**. Puedes definir una en términos de otras:

```
DIGIT = [0-9]
INT = -?{DIGIT}+
FLOAT = {INT}\.{DIGIT}+
NUMBER = {INT}|{FLOAT}
```

Esta es la base de cómo se definen tokens en la mayoría de generadores de parsers (lex, flex).

```{admonition} 🤔 Reflexiona
:class: hint
¿Por qué las expresiones regulares son "composables"? ¿Qué ventaja ofrece esto al diseñar lexers complejos?
```

## Autómatas Finitos Deterministas (DFA)

Un **DFA** (Deterministic Finite Automaton) es una máquina imaginaria que reconoce patrones. Está compuesto de:

1. **Estados**: Puntos de decisión
2. **Transiciones**: Aristas etiquetadas con caracteres
3. **Estado inicial**: Por donde empezamos
4. **Estados de aceptación**: Estados "felices" que significan "acepté el string"

### Definición Formal

Un DFA es una tupla: **M = (Q, Σ, δ, q₀, F)**

- **Q**: Conjunto de estados (finito)
- **Σ**: Alfabeto (conjunto de caracteres que procesamos)
- **δ**: Función de transición (Q × Σ → Q)
- **q₀**: Estado inicial
- **F ⊆ Q**: Estados de aceptación

### Ejemplo 1: Reconocer "hola"

```
q₀ --h--> q₁ --o--> q₂ --l--> q₃ --a--> q₄(accept)

Transiciones por carácter:
δ(q₀, 'h') = q₁
δ(q₁, 'o') = q₂
δ(q₂, 'l') = q₃
δ(q₃, 'a') = q₄

Entrada: "hola" → Aceptado (terminamos en q₄)
Entrada: "hol"  → Rechazado (terminamos en q₃, no aceptación)
Entrada: "hola mundo" → Rechazado (caracter extra después)
```

### Ejemplo 2: Reconocer Números Binarios

Queremos aceptar cualquier string de 0s y 1s (con al menos uno).

```
Estados:
- q₀: Inicial, sin dígitos vistos aún
- q₁: Hemos visto al menos un dígito binario
  ↓ es el estado de aceptación

Diagrama:
              ┌─0,1─┐
              v     │
q₀ --0,1--> q₁ ----┘
^
│ (otro = rechazo)
```

La función de transición:

```
δ(q₀, '0') = q₁
δ(q₀, '1') = q₁
δ(q₁, '0') = q₁
δ(q₁, '1') = q₁
δ(q₀, cualquier otro) = error (rechazo)
δ(q₁, cualquier otro) = error (rechazo)
```

Entradas:
- `"0"` → Aceptado (q₀ → q₁)
- `"101"` → Aceptado (q₀ → q₁ → q₁ → q₁)
- `""` → Rechazado (nunca dejamos q₀)
- `"102"` → Rechazado (no hay transición para '2' desde q₁)

:::{figure} diagrams/dfa_binary.png
:name: fig-dfa-binary
:alt: Diagrama del DFA para reconocer números binarios
:align: center
:width: 90%

**Figura 1:** DFA que acepta números binarios (cadenas de 0s y 1s). El estado q₁ es de aceptación, permitiendo uno o más dígitos binarios.
:::

### Ejemplo 3: Identificadores en Lenguajes de Programación

Un identificador: comienza con letra o `_`, seguido de letras, dígitos, o `_`.

```
Expresión regular: [a-zA-Z_][a-zA-Z0-9_]*

DFA:
       ┌─[a-zA-Z0-9_]─┐
       │               v
q₀ --> q₁(accept) ---→ q₁
^      ^
│      └─[a-zA-Z_]─┘
└─(cualquier otro char) → error


δ(q₀, [a-zA-Z_]) = q₁
δ(q₁, [a-zA-Z0-9_]) = q₁
(todo lo demás lleva a rechazo)

Aceptados: x, _var, myFunction, i2c_channel
Rechazados: 2x (comienza con número), @var (carácter inválido)
```

## Tabla de Transición

En lugar de dibujar estados, podemos representar un DFA como tabla:

```
Para el DFA de números binarios:

        0    1   otro
q₀     q₁   q₁   err
q₁     q₁   q₁   err

Estados finales: {q₁}
```

Esta es exactamente la representación que usaría un compilador internamente.

## Proceso de Reconocimiento

Cuando un DFA procesa una entrada, sigue este algoritmo:

```
Algoritmo: Simular DFA
1. estado_actual = q₀
2. Para cada carácter en entrada:
      transición = δ(estado_actual, carácter)
      si transición es indefinida:
          RECHAZAR
      estado_actual = transición
3. Si estado_actual está en F:
      ACEPTAR
   Si no:
      RECHAZAR
```

Este algoritmo es **O(n)** donde n es la longitud de la entrada. Muy eficiente.

## Eficiencia del Léxer

En la práctica, un **léxer** (analizador léxico) mantiene múltiples DFAs pequeños corriendo en paralelo, cada uno buscando un token diferente:

```
Entrada: "int x = 42;"

Ejecutar simultáneamente:
- DFA para palabras clave → reconoce "int"
- DFA para identificadores → reconoce "x"
- DFA para operadores → reconoce "="
- DFA para números → reconoce "42"

Resultado: [KEYWORD(int), ID(x), ASSIGN, INT(42), SEMICOLON]
```

XGrammar usa esta idea: después de léxing (que generalmente hacemos con regex o DFAs), el parser toma los tokens y estructura.

## Conexión con Expresiones Regulares

Hay un **teorema fundamental** en teoría de autómatas:

**Teorema**: Un lenguaje es regular (reconocible por DFA) **si y solo si** puede describirse con una expresión regular.

Esto significa:
- Cada regex puede convertirse a un DFA
- Cada DFA puede convertirse a una regex

Por esto, herramientas como `flex` pueden tomar regexes y automáticamente generar DFAs eficientes.

## Limitaciones de los Lenguajes Regulares

Aunque potentes, los DFAs tienen limitaciones claras:

### No pueden contar

DFAs no pueden reconocer:
```
a^n b^n  (misma cantidad de 'a' y 'b')
```

¿Por qué? Los DFAs solo tienen **memoria finita** (número fijo de estados). Para verificar que el número de 'a' sea igual al de 'b', necesitarían "recordar" cuántas 'a' vieron, lo cual requiere infinitos estados potenciales.

### No pueden anidar balanceadamente

DFAs no pueden reconocer:
```
(()())   (paréntesis balanceados)
```

Esto requiere una **pila** para recordar paréntesis abiertos. Los DFAs no tienen pilas; solo un estado.

**Solución**: Necesitamos algo más poderoso. Aquí es donde entran los **autómatas de pila** (PDAs) y las **gramáticas libres de contexto**.

```{admonition} ⚠️ Error Común
:class: warning
Los estudiantes frecuentemente intentan usar DFAs para reconocer paréntesis balanceados o expresiones anidadas. Esto es imposible: los DFAs tienen memoria finita fija y no pueden contar arbitrariamente. Para estructuras anidadas necesitas una pila (PDAs/CFGs).
```

## Aplicación a XGrammar

En XGrammar, los lenguajes regulares son **la fase 1** de tokenización:

1. **Regex → DFA**: Convertimos expresiones regulares a DFAs eficientes
2. **Tokenización**: Los DFAs corren para identificar tokens individuales
3. **Salida**: Stream de tokens (sin estructura)

Ejemplo en un kernel Triton:

```
# Código fuente:
@triton.jit
def kernel(x_ptr: tl.tensor):
    ...

# Fase 1 (Tokens):
[DECORATOR(@), ID(triton), DOT, ID(jit),
 KEYWORD(def), ID(kernel), LPAREN, ID(x_ptr), COLON, ...]

# Fase 2 (Parsing CFG): Estructura los tokens
```

```{admonition} 🎯 Conexión con el Proyecto
:class: important
En XGrammar, los lenguajes regulares definen la tokenización inicial. Antes de que el parser CFG estructure el código, el lexer usa DFAs para identificar tokens individuales (palabras clave, identificadores, operadores). Esta separación permite que el parser trabaje con tokens de alto nivel en lugar de caracteres individuales, mejorando eficiencia y claridad.
```

```{admonition} Resumen
:class: important
**Conceptos clave:**
- Los lenguajes regulares se expresan con expresiones regulares y se reconocen con DFAs
- Un DFA tiene estados finitos, transiciones deterministas, y reconoce patrones sin memoria ilimitada
- Los DFAs no pueden contar (a^n b^n) ni anidar balanceadamente (paréntesis)
- La tokenización (fase léxica) usa DFAs; la estructura sintáctica requiere CFGs (siguiente nivel)

**Para la siguiente lectura:**
Exploraremos los NFAs (Autómatas Finitos No-Deterministas) y cómo convertirlos a DFAs mediante la construcción de subconjuntos, una técnica fundamental en la compilación de gramáticas.
```

```{admonition} ✅ Verifica tu comprensión
:class: note
1. ¿Qué tipo de lenguaje reconoce un DFA? ¿Y qué NO puede reconocer?
2. Dibuja el DFA para reconocer identificadores que empiezan con letra y contienen letras/dígitos
3. ¿Por qué un DFA no puede reconocer paréntesis balanceados?
4. En XGrammar, ¿qué fase usa DFAs y para qué?
```

## Ejercicios

1. **Diseña una DFA** para aceptar números en punto flotante válidos (`3.14`, `.5`, `10.`). Dibuja el diagrama de estados.

2. **Escribe una expresión regular** para:
   - Hexadecimal color: `#RRGGBB`
   - Dirección IPv4: `XXX.XXX.XXX.XXX` (donde X es dígito)
   - Identificador Python válido

3. **Tabla de transición**: Crea la tabla para un DFA que reconoce la palabra "gpu".

4. **Reconocimiento de token**: Traza manualmente estos strings a través de tu DFA de identificadores:
   - `variable_1` → ¿Aceptado?
   - `_private` → ¿Aceptado?
   - `123_name` → ¿Aceptado?

5. **Limitación teórica**: Explica por qué un DFA no puede reconocer todos los strings donde el número de 'a' es menor que el número de 'b'.

## Preguntas de Reflexión

- ¿Cuál es la relación entre "estados en un DFA" y "memoria requerida"?
- En el contexto de GPU kernels, ¿qué patrones son "suficientemente simples" como para usar regex? ¿Cuáles necesitan algo más poderoso?
- ¿Por qué crees que la industria usa herramientas como `flex` para generar lexers automáticamente en lugar de escribir DFAs manualmente?

## Conexión con Constrained Decoding en LLMs

```{admonition} 🔗 Aplicación en el proyecto
:class: tip

En generación de código con LLMs, los DFAs restringen qué tokens puede generar el modelo:
- Estado actual del DFA = contexto de lo generado
- Transiciones válidas = tokens permitidos
- Ejemplo: Generando un número de teléfono `(XXX)`, el DFA solo permite dígitos en posiciones específicas.
```

```{admonition} 🎮 Simulador DFA Interactivo (Plotly — Sin Internet)
:class: tip
El siguiente simulador visualiza DFAs directamente en el notebook con Plotly.
```

```{admonition} Teoría vs Implementación
:class: warning
**Definición formal de DFA:** Una tupla M = (Q, Σ, δ, q₀, F) donde Q es el conjunto de estados, Σ el alfabeto, δ la función de transición, q₀ el estado inicial, y F los estados de aceptación.

**Implementación práctica:** El código Python a continuación es una *simulación* del concepto matemático. Usa diccionarios para representar δ, listas para Q y F, y funciones para visualizar. No confundas la elegancia matemática de la definición formal con los detalles de implementación.
```

```{code-cell} ipython3
# Simulador DFA interactivo con Plotly
# Permite visualizar cualquier DFA y trazar el procesamiento de una cadena

import plotly.graph_objects as go
import math

def plot_dfa(states, transitions, initial, accepting,
             input_string=None, title="Simulador DFA"):
    """
    Visualiza un DFA como grafo de estados usando Plotly.

    Args:
        states:       Lista de nombres de estados, ej. ['q0','q1','q2']
        transitions:  Dict {(estado, símbolo): siguiente_estado}
        initial:      Estado inicial (str)
        accepting:    Conjunto/lista de estados de aceptación
        input_string: String a trazar (opcional). Si se pasa, colorea el camino.
        title:        Título del gráfico
    """
    n = len(states)
    # Posicionar estados en círculo
    angle = 2 * math.pi / n
    pos = {s: (math.cos(i * angle), math.sin(i * angle)) for i, s in enumerate(states)}

    # --- Determinar camino si se da input_string ---
    path_states = [initial]
    path_valid = True
    if input_string is not None:
        current = initial
        for c in input_string:
            nxt = transitions.get((current, c))
            if nxt is None:
                path_valid = False
                break
            current = nxt
            path_states.append(current)
        accepted = path_valid and current in accepting

    # --- Dibujar aristas ---
    edge_traces = []
    drawn_pairs = set()
    for (src, sym), dst in transitions.items():
        pair = (src, dst)
        offset = 0.08 if pair in drawn_pairs else 0
        drawn_pairs.add(pair)
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        # Color destacado si esta transición está en el camino
        on_path = False
        if input_string:
            for k in range(len(path_states) - 1):
                if path_states[k] == src and path_states[k+1] == dst:
                    on_path = True; break
        color = '#E74C3C' if on_path else '#7f8c8d'
        width = 3 if on_path else 1.5

        # Arista (o auto-loop)
        if src == dst:
            # Auto-loop como arco (aproximado)
            mid_x = x0 + 0.15
            mid_y = y0 + 0.15
            edge_traces.append(go.Scatter(
                x=[x0, mid_x, x0], y=[y0, mid_y, y0],
                mode='lines', line=dict(color=color, width=width),
                hoverinfo='none', showlegend=False
            ))
        else:
            mx = (x0 + x1) / 2 + offset * -(y1 - y0)
            my = (y0 + y1) / 2 + offset * (x1 - x0)
            edge_traces.append(go.Scatter(
                x=[x0, mx, x1], y=[y0, my, y1],
                mode='lines+text',
                line=dict(color=color, width=width, dash='solid'),
                text=['', sym, ''],
                textposition='middle center',
                textfont=dict(size=13, color='#2c3e50'),
                hoverinfo='none', showlegend=False
            ))

    # --- Dibujar nodos ---
    node_x = [pos[s][0] for s in states]
    node_y = [pos[s][1] for s in states]

    def node_color(s):
        if input_string:
            if s == path_states[-1]:
                return '#27AE60' if accepted else '#E74C3C'
            if s in path_states:
                return '#F39C12'
        if s in accepting:
            return '#2ECC71'
        if s == initial:
            return '#3498DB'
        return '#BDC3C7'

    node_colors = [node_color(s) for s in states]
    symbols = ['circle-open' if s in accepting else 'circle' for s in states]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=40, color=node_colors,
                    line=dict(width=3, color='#2c3e50'),
                    symbol=symbols),
        text=states,
        textposition='middle center',
        textfont=dict(size=13, color='white', family='monospace'),
        hoverinfo='text',
        hovertext=[f"{'INICIAL ' if s==initial else ''}{'ACEPTA' if s in accepting else ''}" for s in states],
        showlegend=False
    )

    # --- Layout ---
    fig = go.Figure(data=edge_traces + [node_trace])
    subtitle = ""
    if input_string is not None:
        status = "✓ ACEPTADA" if accepted else "✗ RECHAZADA"
        subtitle = f"<br><sup>Cadena: '{input_string}' → {status}   |   Camino: {' → '.join(path_states)}</sup>"
    fig.update_layout(
        title=f"<b>{title}</b>{subtitle}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        plot_bgcolor='white', height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.show()

# ─────────────────────────────────────────────────────────────
# EJEMPLO 1: DFA para Números Binarios
# ─────────────────────────────────────────────────────────────
print("DFA 1: Números Binarios  (acepta: '0','1','101','110' | rechaza: '','102')")
plot_dfa(
    states=['q0', 'q1'],
    transitions={
        ('q0', '0'): 'q1', ('q0', '1'): 'q1',
        ('q1', '0'): 'q1', ('q1', '1'): 'q1',
    },
    initial='q0',
    accepting={'q1'},
    input_string='1011',
    title='DFA — Números Binarios'
)

# ─────────────────────────────────────────────────────────────
# EJEMPLO 2: DFA para Identificadores  [a-z_][a-z0-9_]*
# ─────────────────────────────────────────────────────────────
print("\nDFA 2: Identificadores  (acepta: '_var','hello' | rechaza: '2fast','hi!')")
id_chars = [chr(c) for c in range(ord('a'), ord('z')+1)] + ['_']
id_full  = id_chars + [str(d) for d in range(10)]

transitions_id = {}
for c in id_chars:
    transitions_id[('q0', c)] = 'q1'
for c in id_full:
    transitions_id[('q1', c)] = 'q1'

plot_dfa(
    states=['q0', 'q1'],
    transitions=transitions_id,
    initial='q0',
    accepting={'q1'},
    input_string='_kernel',
    title='DFA — Identificadores  [a-z_][a-z0-9_]*'
)

# ─────────────────────────────────────────────────────────────
# EJEMPLO 3: DFA para reconocer la palabra "gpu"
# ─────────────────────────────────────────────────────────────
print('\nDFA 3: Palabra exacta "gpu"')
plot_dfa(
    states=['q0', 'q1', 'q2', 'q3'],
    transitions={
        ('q0', 'g'): 'q1',
        ('q1', 'p'): 'q2',
        ('q2', 'u'): 'q3',
    },
    initial='q0',
    accepting={'q3'},
    input_string='gpu',
    title='DFA — Palabra exacta "gpu"'
)
```

```{code-cell} ipython3
# Traza paso a paso: ¿qué pasa en cada estado?
def trace_dfa(transitions, initial, accepting, input_string):
    """Muestra el trace completo de un DFA procesando una cadena."""
    current = initial
    print(f"Procesando: '{input_string}'")
    print(f"Estado inicial: {current}")
    print("-" * 50)

    for i, symbol in enumerate(input_string):
        next_state = transitions.get((current, symbol))
        if next_state is None:
            print(f"  Paso {i+1}: δ({current}, '{symbol}') → ∅  ✗ RECHAZA (sin transición)")
            return False
        status = " ← ACEPTA" if next_state in accepting else ""
        print(f"  Paso {i+1}: δ({current}, '{symbol}') → {next_state}{status}")
        current = next_state

    accepted = current in accepting
    print("-" * 50)
    print(f"Estado final: {current} → {'✓ ACEPTADO' if accepted else '✗ RECHAZADO'}")
    return accepted

# Tabla comparativa de casos
casos = ['1010', '0', '', '102', '11111']
bin_trans = {
    ('q0','0'):'q1', ('q0','1'):'q1',
    ('q1','0'):'q1', ('q1','1'):'q1',
}
print("=== DFA de Números Binarios — Tabla de Casos ===")
print(f"{'Cadena':<12} {'Resultado'}")
print("-" * 25)
import sys
for c in casos:
    # capturar print silenciosamente
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    result = trace_dfa(bin_trans, 'q0', {'q1'}, c)
    sys.stdout = old_stdout
    label = '✓ Acepta' if result else '✗ Rechaza'
    print(f"  '{c}'{'':>{10-len(c)}}  {label}")
```

## Ejercicio Práctico: Traza de DFA

Traza manualmente el procesamiento de cadenas en un DFA de identificadores.

```{code-cell} ipython3
# DFA para identificadores: [a-z_][a-z0-9_]*
def dfa_identifier(s):
    """Simula un DFA que reconoce identificadores válidos."""
    if not s:
        return False, "Cadena vacía"

    # Estado inicial: verificar primer carácter
    if not (s[0].isalpha() or s[0] == '_'):
        return False, f"Carácter inicial inválido: '{s[0]}'"

    # Estados siguientes: letras, dígitos o _
    trace = [f"q0 --[{s[0]}]--> q1 (acepta primer char)"]
    for i, c in enumerate(s[1:], 1):
        if c.isalnum() or c == '_':
            trace.append(f"q1 --[{c}]--> q1 (continúa)")
        else:
            return False, f"Carácter inválido en posición {i}: '{c}'"

    return True, trace

# Probar con diferentes cadenas
test_cases = ["_var123", "2fast", "my_function", "hello-world"]

for s in test_cases:
    valid, result = dfa_identifier(s)
    print(f"\n'{s}':")
    if valid:
        print(f"  ✓ ACEPTA")
        for step in result[:3]:  # Primeros 3 pasos
            print(f"    {step}")
        if len(result) > 3:
            print(f"    ... ({len(result)-3} pasos más)")
    else:
        print(f"  ✗ RECHAZA: {result}")
```

---

## Referencias

- Hopcroft, J. & Ullman, J. (1979). Introduction to Automata Theory, Languages, and Computation. Addison-Wesley.
- Sipser, M. (2012). [Introduction to the Theory of Computation](https://math.mit.edu/~sipser/book.html) (3rd ed.). Cengage Learning.
- Rabin, M. & Scott, D. (1959). [Finite Automata and Their Decision Problems](https://www.cse.chalmers.se/~coquand/AUTOMATA/rs.pdf). IBM Journal of Research and Development.
