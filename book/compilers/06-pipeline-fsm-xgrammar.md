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

# Pipeline FSM de XGrammar: De Gramática a Autómata Eficiente

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Describir las 10 fases del pipeline de compilación de XGrammar
- Explicar cómo se normaliza una gramática a forma canónica
- Calcular conjuntos FIRST y FOLLOW para análisis de lookahead
- Comprender el adaptive token mask cache y su beneficio para performance
- Relacionar cada fase del pipeline con optimizaciones específicas del parser generado
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[XGRAMMAR: Flexible Structured Generation Engine (MLC)](https://www.youtube.com/watch?v=l_aB2w_ZqI8)** - Explicación técnica de la intersección entre gramáticas, GPU y LLMs.
```

## Resumen Ejecutivo
XGrammar compila gramáticas a parsers eficientes: parsing → normalización → NFA → DFA → minimización → análisis lookahead → generación de código. Resultado: parser Earley con máscaras de bits O(1).

## La Misión: Compilar una Gramática

XGrammar es un compilador especializado. Su entrada no es código fuente tradicional, sino **especificaciones de gramática**. Su salida es un **parser eficiente** que reconoce exactamente ese lenguaje.

Entender el pipeline interno es crucial para usar XGrammar efectivamente.

## Visión General del Pipeline

```
Entrada:           Descripción de Gramática (CFG o EBNF)
                              ↓
[PARSING]          Parsear la especificación misma
                              ↓
[NORMALIZATION]    Convertir a forma canónica
                              ↓
[NFA CONSTRUCTION] Construir autómata no-determinista
                              ↓
[EPSILON ELIM]     Eliminar transiciones ε
                              ↓
[SUBSET CONSTRUCTION] Convertir NFA a DFA
                              ↓
[DFA MINIMIZATION] Eliminar estados redundantes
                              ↓
[RULE INLINING]    Optimizar reglas simples
                              ↓
[DEAD CODE ELIM]   Remover código inalcanzable
                              ↓
[LOOKAHEAD ANALYSIS] Calcular conjuntos FIRST/FOLLOW
                              ↓
[PARSER GENERATION] Generar código ejecutable (Earley o similar)
                              ↓
Salida:            Parser compilado (C++, Python, etc.)
```

![**Figura 1:** Pipeline completo de XGrammar desde gramática hasta bitmask de tokens.](diagrams/xgrammar_pipeline.png)

***Figura 1:** Pipeline completo de XGrammar desde gramática hasta bitmask de tokens.*


Veamos cada fase en detalle.

## Fase 1: Parsing de la Especificación

**Objetivo**: La especificación de gramática misma debe parsearse.

```
Entrada: Una especificación como:
  expr = term ("+" term)*
  term = factor ("*" factor)*
  factor = "(" expr ")" | identifier | number

XGrammar debe reconocer que esto es una especificación de gramática válida.
Internamente usa su propio parser (bootstrap) para esto.

Resultado: Estructura interna AST (Abstract Syntax Tree) de la gramática
```

## Fase 2: Normalización

**Objetivo**: Convertir la gramática a una **forma normal** - una representación canónica.

### ¿Por qué normalizar?

Diferentes formas de escribir la misma idea deben normalizarse:

```
Original:
  expr = term + expr | term
  term = factor * term | factor
  factor = "(" expr ")" | ID | NUM

Problemas:
- Recursión mixta (derechista e izquierdista)
- Alternativas implícitas
- Posibles ambigüedades

Normalizado (forma CNF-like):
  expr → term expr_rest
  expr_rest → "+" expr | ε
  term → factor term_rest
  term_rest → "*" term | ε
  factor → "(" expr ")" | ID | NUM

O también válido:
  expr → term | expr "+" term  (Greibach Normal Form preparada)
```

### Ventajas de Normalización

```
1. Consistencia: Todos los algoritmos posteriores asumen forma normal
2. Simplificación: Eliminamos variaciones que complican análisis
3. Optimización: Es más fácil optimizar código regular
4. Portabilidad: Salida normalizada es más transferible
```

## Fase 3: Construcción de NFA

**Objetivo**: Convertir la CFG normalizada a un **NFA equivalente** que reconoce el lenguaje.

Este es conceptualmente el paso más interesante. **Cómo se mapea una CFG a un autómata?**

### Construcción de Thompson

Para una gramática simple como:
```
S → a b
```

Se construye:
```
q₀ --a--> q₁ --b--> q₂ (accept)
```

Para alternancia:
```
S → a | b

    ┌─a─→ q₁ ─┐
q₀─┤        ├→ q₃ (accept)
    └─b─→ q₂ ─┘
```

Para repetición:
```
S → a*

    ┌─a─┐
q₀─┤    ├→ q₂ (accept)
    └──ε──┘

Permite cero o más 'a'
```

### Construcción Recursiva para Gramáticas

Para una regla como:
```
expr → term "+" term
```

Se construye:
```
NFA(expr) = NFA(term) ⊕ "+" ⊕ NFA(term)

Donde ⊕ significa concatenación de autómatas
```

### Ejemplo Completo: Expresiones Aritméticas

```
Gramática:
  expr → term ("+" term)*
  term → factor ("*" factor)*
  factor → "(" expr ")" | ID | NUM

NFA para expresión:
  [NFA_term] -ε→ [LOOP: "+" → NFA_term]*

NFA para término:
  [NFA_factor] -ε→ [LOOP: "*" → NFA_factor]*

NFA para factor:
         ┌─"("─→ [NFA_expr] ─")"─┐
    q₀─ε┤                        ├─ε→ accept
         ├─ID─→ q₁ ─ε────────────┤
         └─NUM─→ q₂ ─ε───────────┘

Resultado: NFA gigante que reconoce toda la gramática
```

## Fase 4: Eliminación de Epsilon

**Objetivo**: Remover transiciones ε para simplificar posterior procesamiento.

### ¿Por qué eliminar ε?

Los algoritmos de conversión NFA→DFA funcionan mejor sin ε. Aunque técnicamente funcionarían con ε (con ε-clausura), la representación interna es más limpia sin ellos.

### Algoritmo de Eliminación

```
Para cada transición ε:
  p -ε-> q

1. Calcular ε-clausura(p) = {p, ...}
2. Calcular ε-clausura(q) = {q, ...}
3. Para cada transición q -a-> r:
     Añadir p -a-> r (directamente, saltando ε)
4. Si q es aceptación:
     Hacer p también aceptación

Resultado: Mismo lenguaje, sin transiciones ε
```

### Ejemplo

```
Antes:
q₀ -ε-> q₁ --a--> q₂

Después de eliminación ε:
q₀ --a--> q₂

(q₁ se elimina si solo servía para la transición ε)
```

## Fase 5: Construcción de Subconjuntos (NFA → DFA)

**Objetivo**: Convertir NFA a DFA determinista para ejecución eficiente.

Ya estudiamos esto en detalle. Brevemente:

```
Algoritmo: Subset construction
1. Crear estado inicial del DFA: ε-clausura(q₀_nfa)
2. BFS a través de todos los subconjuntos alcanzables
3. Cada subconjunto es un estado DFA
4. Transiciones siguen la unión de transiciones NFA

Resultado: DFA con hasta 2^|Q_nfa| estados (peor caso)
```

## Fase 6: Minimización de DFA

**Objetivo**: Eliminar estados redundantes.

Usamos algoritmos como **Hopcroft** para agrupar estados equivalentes:

```
Antes (después de subset construction):
q₁ -a-> q₃
q₂ -a-> q₃
(q₁ y q₂ son idénticos en comportamiento)

Después de minimización:
q₁ = q₂ (fusionados)

Ahora: {q₁,q₂} -a-> q₃
```

### Ganancia

```
Tamaño del DFA antes: potencialmente exponencial en #reglas
Tamaño después: típicamente 1-10% del inicial

Ejecución: Mucho más rápida (menos estados = menos memoria)
```

## Fase 7: Inlining de Reglas

**Objetivo**: Simplificar la gramática eliminando reglas triviales.

### Casos de Inlining

```
Caso 1: Regla con una sola producción
  factor → atom    (si solo aparece una vez)

  Antes:  factor → atom
          atom → ID

  Después: factor → ID

Caso 2: Regla que aparece una sola vez
  Si 'factor' solo se usa en 'term', podemos inline su definición

Ventaja: Menos estados en el autómata final
Desventaja: Menos modularidad (si queremos reuso)
```

## Fase 8: Eliminación de Código Muerto

**Objetivo**: Remover estados/transiciones que nunca se alcanzan.

### Detección

```
Algoritmo: BFS desde estado inicial
  Alcanzables = todos los estados visitables desde q₀
  Muertos = todos los estados en Q - Alcanzables
  Remover muertos

Ejemplo:
  q₀ --a--> q₁ --b--> q₂ (accept)
  q₃ --c--> q₄            (no alcanzable desde q₀)

  Resultado: q₃, q₄ se eliminan
```

## Fase 9: Análisis de Lookahead (FIRST/FOLLOW)

**Objetivo**: Calcular qué tokens pueden aparecer en cada punto.

Esto es crucial para:
- Errores informativos (esperábamos X pero vimos Y)
- Optimizaciones (saltar ramas imposibles)
- Generación de parsers eficientes (tablas de decisión)

### FIRST(symbol)

Conjunto de terminales que pueden ser **primero** en una derivación de `symbol`.

```
expr → term ("+" term)*
term → factor ("*" factor)*
factor → "(" expr ")" | ID | NUM

FIRST(factor) = {"(", ID, NUM}
FIRST(term) = FIRST(factor) = {"(", ID, NUM}
FIRST(expr) = FIRST(term) = {"(", ID, NUM}
```

### FOLLOW(symbol)

Conjunto de terminales que pueden aparecer **después** de `symbol` en una derivación válida.

```
expr → term ("+" term)*

FOLLOW(term en expr) = {"+" , EOF}   (puede venir + o fin)

term → factor ("*" factor)*

FOLLOW(factor en term) = {"*", FOLLOW(term)}
```

**Uso en compilación**: Si parseamos un `factor` en contexto `term`, sabemos que después debe venir `*` o algo en `FOLLOW(term)`. Si vemos algo distinto, es error.

## Fase 10: Generación de Parser

**Objetivo**: Producir código ejecutable en lenguaje de destino.

XGrammar genera **Earley parsers** (u otro tipo), con optimizaciones basadas en análisis anterior.

```
Entrada a generador:
  - DFA minimizado
  - Tabla de transiciones
  - Información FIRST/FOLLOW
  - Información de aceptación

Generador escribe (ejemplo Python):

def parse(tokens):
    stack = [initial_state]
    position = 0

    while position < len(tokens):
        state = stack[-1]
        token = tokens[position]

        # Búsqueda en tabla DFA
        if transition := dfa_table[state].get(token):
            stack.append(transition)
            position += 1
        elif # regla de reducción
            # aplicar reducción
            ...

Resultado: Función ejecutable que acepta/rechaza input
```

:::{figure} diagrams/xgrammar_detail.png
:name: fig-xgrammar-detail
:alt: Detalles del pipeline FSM de XGrammar con todas sus fases de optimización
:align: center
:width: 90%

**Figura 2:** Detalles completos del pipeline de XGrammar mostrando todas las 10 fases de compilación: desde normalización de gramática hasta generación de código del parser Earley optimizado.
:::

## Optimización: Adaptive Token Mask Cache

XGrammar usa una optimización sofisticada llamada **"adaptive token mask cache"**:

```
Observación: En cada estado del DFA, solo algunos tokens son válidos.

Sin cache:
  Cada token-lookahead requiere búsqueda en tabla completa: O(|alphabet|)

Con cache:
  Calcular máscara de tokens válidos → bitmap
  Lookahead: lookup O(1) en bitmap

Adaptativo:
  Si muchos tokens válidos → usar tabla
  Si pocos → usar bitmap
  Cambiar estrategia dinámicamente según acceso
```

## Ciclo Completo: Ejemplo

Especificación:
```
json_value = json_object | json_array | string | number | "true" | "false" | "null"
json_object = "{" | (string ":" json_value ("," string ":" json_value)*) "}"
json_array = "[" | (json_value ("," json_value)*) "]"
string = '"' char* '"'
number = "-"? digit+ ("." digit+)?
digit = "0".."9"
char = /* any */
```

Pipeline:
```
1. [PARSING] → AST de la especificación
2. [NORMALIZATION] → Forma normal interna
3. [NFA CONSTRUCTION] → NFA con ~100+ estados
4. [EPSILON ELIM] → NFA limpio
5. [SUBSET CONST] → DFA con ~50-200 estados
6. [MINIMIZATION] → DFA con ~20-30 estados
7. [INLINING] → Fusión de reglas simples
8. [DEAD CODE ELIM] → Remover estados inalcanzables
9. [LOOKAHEAD] → Tablas FIRST/FOLLOW
10. [CODEGEN] → Parser ejecutable optimizado

Resultado: Parser que reconoce JSON válido en O(n) tiempo.
```

## Rendimiento en Práctica

Para una gramática típica de kernel GPU:

```
Especificación: ~20 reglas EBNF
NFA resultante: ~200 estados
DFA después subset: ~500 estados (peor caso)
DFA después minimización: ~30-50 estados (típico)

Parsing: Token-by-token, ~10-100 nanosegundos por token
Memoria: Tabla de transiciones ≈ 50-200 KB

Comparación LL(1): ~O(n log n) para construcción manual
XGrammar Earley: O(n) o O(n²) para gramáticas especiales
```

```{admonition} Resumen
:class: important
**Conceptos clave:**
- XGrammar compila gramáticas en 10 fases: parsing → normalización → NFA → eliminación ε → DFA → minimización → inlining → dead code → lookahead → codegen
- FIRST(X) calcula terminales que pueden iniciar X; FOLLOW(X) calcula qué puede seguir a X
- El adaptive token mask cache usa bitmasks para validación O(1) de tokens válidos por estado
- Cada fase optimiza el parser: normalización simplifica, minimización reduce estados, lookahead acelera decisiones

**Para la siguiente lectura:**
Estudiaremos principios de diseño de gramáticas DSL: cómo estructurar gramáticas para que sean claras, eficientes y libres de ambigüedad.
```

## Ejercicios

1. **Traza NFA**: Construye manualmente un NFA para:
   ```
   S → "a" "b" | "a" "c"
   ```

2. **Epsilon Elimination**: Elimina transiciones ε de:
   ```
   q₀ -ε-> q₁ --a--> q₂
   q₁ -ε-> q₃ --b--> q₂
   ```

3. **FIRST/FOLLOW**: Calcula para:
   ```
   S → A "x" | B "y"
   A → "a"
   B → A | "b"
   ```

4. **Minimización**: ¿Qué estados serían equivalentes?
   ```
   q₁ --a--> accept
   q₂ --a--> accept
   q₃ --a--> reject
   ```

5. **Lookahead**: ¿Cuál es FIRST(term) en?
   ```
   expr → term "+" expr | term
   term → "(" expr ")" | ID
   ```

## Preguntas de Reflexión

- ¿Por qué es importante pasar por un NFA intermedio en lugar de ir directo CFG → DFA?
- ¿Cuál es el trade-off entre "minimizar DFA" y "tiempo de compilación"?
- ¿Cómo el adaptive token mask cache mejora el rendimiento de parsing?
- En un GPU kernel DSL, ¿qué reglas serían candidatas para inlining?

## Conexión con Constrained Decoding en LLMs

```{admonition} 🔗 Aplicación en el proyecto
:class: tip

El pipeline de XGrammar compila gramáticas a parsers eficientes que controlan la generación de LLMs:
- **FIRST/FOLLOW** calculan qué tokens son válidos en cada posición → máscaras de bits para el LLM
- **DFA minimizado** permite verificación O(1) de validez de tokens durante generación
- **Adaptive token mask cache** acelera la inferencia: en lugar de revisar miles de tokens, solo verificamos los probables
- En el proyecto, verás cómo XGrammar usa estas optimizaciones para generar código Python/CUDA sintácticamente correcto en tiempo real

**Pipeline en acción:**
1. Escribes gramática EBNF para Python
2. XGrammar la compila a DFA optimizado
3. Durante inferencia, el LLM genera token por token
4. En cada paso, FIRST/FOLLOW determinan qué tokens son sintácticamente válidos
5. Solo estos tokens reciben probabilidad no-cero
6. Resultado: código 100% sintácticamente correcto
```

## Visualización del Pipeline XGrammar

```{code-cell} ipython3
# Pipeline de XGrammar interactivo
import plotly.graph_objects as go

fig = go.Figure()

# Fases del pipeline
fases = ['JSON Schema', 'Parser', 'AST', 'FSM', 'Token Mask']
x_pos = [0.1, 0.3, 0.5, 0.7, 0.9]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D', '#95E1A3']

for i, (fase, x, color) in enumerate(zip(fases, x_pos, colors)):
    # Cajas
    fig.add_shape(type="rect", x0=x-0.08, y0=0.4, x1=x+0.08, y1=0.6,
                  fillcolor=color, line=dict(color='black'))
    fig.add_annotation(x=x, y=0.5, text=fase, showarrow=False,
                       font=dict(size=11, color='black'))
    # Flechas
    if i < len(fases) - 1:
        fig.add_annotation(x=x+0.12, y=0.5, ax=x+0.08, ay=0.5,
                           arrowhead=2, arrowsize=1, arrowcolor='black')

fig.update_layout(title="Pipeline de Compilación XGrammar",
                  xaxis=dict(visible=False, range=[0, 1]),
                  yaxis=dict(visible=False, range=[0, 1]), height=200)
fig.show()
```

## Ejercicio Práctico: Cálculo de FIRST/FOLLOW

Aprende a calcular conjuntos FIRST y FOLLOW manualmente para entender cómo XGrammar optimiza el parsing.

```{code-cell} ipython3
# Cálculo de conjuntos FIRST y FOLLOW

def calculate_first_follow(grammar):
    """
    Calcula FIRST y FOLLOW para una gramática simple.
    Esto es una simulación educativa del análisis que hace XGrammar.
    """
    print(f"\n{'='*70}")
    print("CÁLCULO DE FIRST Y FOLLOW")
    print('='*70)

    # Gramática de ejemplo: expresiones aritméticas
    print("\nGramática:")
    print("-" * 70)
    for rule in grammar['rules']:
        print(f"  {rule}")

    print(f"\n{'='*70}")
    print("CONJUNTOS FIRST")
    print('='*70)
    print("\nFIRST(X) = conjunto de terminales que pueden aparecer primero")
    print("al derivar desde X\n")

    for nt, first_set in grammar['FIRST'].items():
        print(f"FIRST({nt:10s}) = {{{', '.join(sorted(first_set))}}}")

    print(f"\n{'='*70}")
    print("CONJUNTOS FOLLOW")
    print('='*70)
    print("\nFOLLOW(X) = conjunto de terminales que pueden aparecer")
    print("inmediatamente después de X en una derivación válida\n")

    for nt, follow_set in grammar['FOLLOW'].items():
        print(f"FOLLOW({nt:10s}) = {{{', '.join(sorted(follow_set))}}}")

    return grammar

# Ejemplo 1: Expresiones aritméticas
grammar1 = {
    'rules': [
        "expr   → term (('+' | '-') term)*",
        "term   → factor (('*' | '/') factor)*",
        "factor → '(' expr ')' | number | identifier"
    ],
    'FIRST': {
        'expr': {'(', 'number', 'identifier'},
        'term': {'(', 'number', 'identifier'},
        'factor': {'(', 'number', 'identifier'}
    },
    'FOLLOW': {
        'expr': {')', '$', '+', '-'},  # $ = fin de entrada
        'term': {')', '$', '+', '-', '*', '/'},
        'factor': {')', '$', '+', '-', '*', '/'}
    }
}

calculate_first_follow(grammar1)

# Ejemplo 2: Statements con if-else
grammar2 = {
    'rules': [
        "statement → if_stmt | assign_stmt | block",
        "if_stmt   → 'if' expr ':' statement ['else' ':' statement]",
        "assign_stmt → identifier '=' expr",
        "block     → '{' statement+ '}'",
        "expr      → identifier | number"
    ],
    'FIRST': {
        'statement': {'if', 'identifier', '{'},
        'if_stmt': {'if'},
        'assign_stmt': {'identifier'},
        'block': {'{'},
        'expr': {'identifier', 'number'}
    },
    'FOLLOW': {
        'statement': {'else', '}', '$'},
        'if_stmt': {'else', '}', '$'},
        'assign_stmt': {'else', '}', '$'},
        'block': {'else', '}', '$'},
        'expr': {':', '=', 'else', '}', '$'}
    }
}

calculate_first_follow(grammar2)

# Ejercicio práctico: Usar FIRST/FOLLOW para constrained decoding
print(f"\n{'='*70}")
print("APLICACIÓN: Constrained Decoding con FIRST/FOLLOW")
print('='*70)

def constrained_tokens_at_position(non_terminal, grammar):
    """
    Simula cómo XGrammar determina tokens válidos en una posición.
    """
    first_set = grammar['FIRST'].get(non_terminal, set())
    print(f"\nPosición actual en el parsing: {non_terminal}")
    print(f"Tokens válidos según FIRST({non_terminal}): {sorted(first_set)}")
    print(f"\nInterpretación para el LLM:")
    print(f"  → Solo estos tokens pueden aparecer ahora")
    print(f"  → Todos los demás tokens tienen probabilidad 0")
    print(f"  → Esto GARANTIZA corrección sintáctica")
    return first_set

# Caso 1: Comenzando a parsear una expresión
constrained_tokens_at_position('expr', grammar1)

# Caso 2: Comenzando a parsear un statement
constrained_tokens_at_position('statement', grammar2)

# Visualizar el pipeline completo
print(f"\n{'='*70}")
print("PIPELINE XGRAMMAR: De Gramática a Máscaras de Tokens")
print('='*70)
print("""
1. Gramática EBNF
   ↓
2. Normalización → forma canónica
   ↓
3. NFA construcción → autómata no-determinista
   ↓
4. NFA → DFA conversión → autómata determinista
   ↓
5. DFA minimización → estados óptimos
   ↓
6. FIRST/FOLLOW cálculo ← ESTO ES LO QUE ACABAMOS DE VER
   ↓
7. Token mask generation → máscaras de bits
   ↓
8. Constrained decoding → LLM solo genera tokens válidos

RESULTADO: Código sintácticamente correcto garantizado
""")
```

---

## Referencias

- XGrammar. [Efficient, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar). GitHub.
- Willard, B. & Louf, R. (2023). [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702). arXiv.
- Aho, A., Lam, M., Sethi, R., & Ullman, J. (2006). Compilers: Principles, Techniques, and Tools (2nd ed.). Pearson.
