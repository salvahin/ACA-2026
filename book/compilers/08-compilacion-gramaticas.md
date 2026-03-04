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

# Compilación de Gramáticas: De Especificación a Parser

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Explicar las optimizaciones específicas que XGrammar aplica durante compilación
- Calcular conjuntos FIRST y FOLLOW manualmente para una gramática dada
- Comprender cómo las máscaras de tokens aceleran la validación a O(1)
- Construir tablas de parsing LL(1) a partir de conjuntos FIRST/FOLLOW
- Identificar estados equivalentes en DFAs para fusión y optimización
```

## Cerrando la Brecha: Especificación → Código Ejecutable

Sabemos que XGrammar toma una especificación de gramática y genera un parser. Ahora profundizaremos en cómo **específicamente** XGrammar realiza esta compilación, con énfasis en las decisiones de diseño que la hacen eficiente para kernels GPU.

## Fases Adicionales de XGrammar

Revisamos el pipeline FSM en la lectura anterior. Ahora vamos más profundo en las fases específicas de XGrammar que van más allá del pipeline estándar.

## Fase de Normalización Profunda

XGrammar realiza una normalización más agresiva que compiladores típicos.

### Conversión a Forma Normal Greibach

En lugar de solo normalizar, XGrammar convierte a **Greibach Normal Form (GNF)**:

```
Forma normal Greibach: Toda regla es:
  A → aα

Donde:
  a es un terminal
  α es una secuencia de cero o más no-terminales

Ventaja: Garantiza que cada paso de derivación consume exactamente un carácter.
```

### Ejemplo de Conversión

```
Original:
  expr → term "+" expr | term
  term → factor "*" term | factor

Convertir a GNF:

  expr → factor expr_middle
  expr_middle → "*" term expr | "+" expr | ε
  term → factor term_middle
  term_middle → "*" term | ε

Ahora cada producción empieza con un terminal.
```

### Beneficio para GPU

```
Conocer exactamente qué terminal viene después
→ Permite predicción exacta de cual rama tomar
→ Menos backtracking
→ Mejor paralelismo en GPU (sin sincronización de control)
```

## Análisis de Lookahead Adaptativo

XGrammar no solo calcula FIRST/FOLLOW, sino que realiza un análisis **adaptativo**.

### Análisis FIRST Extendido

```
FIRST(A) en computadores generales:
  Conjunto de terminales que puede comenzar A

FIRST en XGrammar:
  FIRST_k(A) = terminales que pueden comenzar A, viendo k tokens adelante
  Adaptativo: Elige k dinámicamente según gramática

Ejemplo:
  expr → term rest
  rest → "+" expr | ε

  FIRST_1(expr) = FIRST_1(term) = {ID, NUM, "("} (factor)
  FIRST_2(expr) = {ID, NUM, "("} × {"+" o EOF}

  Si necesitamos distinguir ID+ vs ID, usamos FIRST_2
  Si FIRST_1 es suficiente, economizamos memoria/tiempo
```

### Análisis FOLLOW Inteligente

```
FOLLOW en contexto de GPU kernels:

Para una regla como:
  loop = "for" ID "in" range_expr ":" block

FOLLOW(range_expr) debe incluir ":"
Pero XGrammar además analiza:
  - ¿Qué puede venir en la próxima línea del kernel?
  - ¿Qué indentación es válida?

Esto permite desambiguación en kernels donde estructura y tokens interactúan.
```

## Construcción Inteligente de Tabla de Parsing

XGrammar no genera una simple tabla de transiciones, sino una **tabla optimizada**.

### Tabla de Parsing Clásica (SLR)

```
Formato clásico:
  [estado][terminal] = acción (shift/reduce)

Tamaño: |estados| × |terminales|
Acceso: O(log |terminales|) si binaria, O(1) si hash

Ejemplo:
        id  "="  num  ";"
q0      s1   -    -    -
q1      -    s2   -    -
q2      -    -    s3   -
q3      -    -    -    s4
q4      r1   r1   r1   r1
```

### Tabla de XGrammar: Representación Comprimida

XGrammar usa **table compression**:

```
Observación: Muchas filas (estados) son similares

Técnica: State merging y goto-table compression

En lugar de:
        a    b    c    d    e    ...
q0      2    -    -    1    5
q1      2    3    -    1    5
q2      2    3    -    1    5    ← igual que q1
q3      2    -    4    1    5
...

Mergear q1 y q2 (idénticos)

Resultado: Tabla 30-40% más pequeña
Acceso sigue O(1) con hash
```

## Caching Dinámico de Máscara de Token

Esta es una optimización sofisticada específica de XGrammar.

### Problema

```
En cada estado del parsing, solo algunos tokens son válidos.

Búsqueda naïve:
  Para cada token en entrada:
    Iterar sobre todos los terminales
    ¿Hay transición válida?
    Complejidad: O(|terminales|) por token

En un kernel CUDA con 1M tokens:
  1M × |terminales| operaciones (potencialmente lento)
```

### Solución: Token Mask

```
Token mask = bitmap de qué tokens son válidos en este estado

Ejemplo (8 tokens, representado como byte):
  Estado q0: 01010101 (tokens 0,2,4,6 válidos)
  Estado q1: 11110000 (tokens 0,1,2,3 válidos)

Validación:
  token_valid = (state_mask & (1 << token_id)) != 0
  Complejidad: O(1) con lookup bitwise

Pero espera: ¿Cómo se construye la máscara?
```

### Construcción Adaptativa

```
La máscara se computa una sola vez durante compilación:

Para cada estado q:
  mask[q] = 0
  Para cada terminal t:
    Si existe transición δ(q, t):
      mask[q] |= (1 << t)

Luego, en tiempo de ejecución:
  token_id = lexer(input)
  if (mask[current_state] & (1 << token_id)):
      transicionar
  else:
      error

¿Qué pasa si hay más de 64 terminales?
  Usar múltiples palabras (u64[2] para 128 terminales, etc.)
  O usar tabla Hash si alfabeto es muy grande
```

### Adaptatividad

El nombre "adaptativo" viene de que XGrammar **elige dinámicamente**:

```
Si |terminales| <= 64:
  Usar máscara de bits (u64)
  Acceso O(1), memoria O(1) por estado

Si 64 < |terminales| <= 256:
  Usar múltiples u64
  Acceso O(1), memoria O(log |terminales|) por estado

Si |terminales| > 256:
  Usar tabla Hash de estados válidos
  Acceso O(1) promedio, memoria O(#transiciones)

Elección automática basada en métricas
```

## Análisis de Alcanzabilidad

Después de construir el autómata, XGrammar realiza análisis de alcanzabilidad.

### Eliminación de Estados Muertos

```
Estado muerto = no alcanzable desde q0 O no puede llegar a aceptación

Algoritmo (revisado):
  1. BFS desde q0, marcar alcanzables
  2. BFS reverso desde estados de aceptación, marcar útiles
  3. Eliminar estados que no están marcados en ambos

Ejemplo:
  q0 -a-> q1 -b-> q2 (accept)
  q3 -c-> q4          (aislado)
  q1 -d-> q5 (no llega aceptación)

  Resultado: Eliminar q3, q4, q5
  Mantener: q0, q1, q2
```

### Simplificación de Transiciones

```
Transición simplificable: Tiene múltiples caracteres/tokens

Antes:
  q0 --[a,b,c]--> q1  (acepte a, b, o c)

Después (explícito):
  q0 --a--> q1
  q0 --b--> q1
  q0 --c--> q1

O mantener como:
  δ(q0, {a,b,c}) = q1

Decisión: Depende de la representación interna
(¿máscara de bits o tabla?)
```

## Inlining Inteligente de Reglas

XGrammar inlinea reglas **selectivamente**, no todas.

### Heurísticas de Inlining

```
Candidato para inlining:
  1. Regla usada una sola vez
  2. Regla muy simple (1-2 alternativas)
  3. Expansión no causa crecimiento exponencial
  4. No es punto de recursión

Ejemplo 1: Candidate (inlinear)
  number = digit+
  factor = "(" expr ")" | number | ID

  Aquí, 'number' → digit+ aparece una sola vez
  Inline: factor = "(" expr ")" | digit+ | ID

Ejemplo 2: No candidate (recursiva)
  expr = expr "+" term | term

  Recursión: No inlinear, es punto clave

Ejemplo 3: No candidate (usado múltiples veces)
  identifier = letter (letter | digit | "_")*

  Usado en: param, variable, function_name
  Mantener como regla separada
```

### Beneficio

```
Inlining reduce:
  - Número de estados en autómata
  - Overhead de llamadas a subparsers
  - Complejidad de árbol de análisis

Pero aumenta:
  - Tamaño de tabla de transición
  - Duplicación de código

XGrammar busca balance
```

## Compilación del Árbol de Análisis

XGrammar no solo genera un parser que acepta/rechaza. También **construye estructuras de datos**.

### AST Construction

```
Gramática:
  expr = term ("+" term)*

Parser resulta en árbol:
        Expr
       /    \
     Term   Op("+")+
     /       |
  Factor    Term
     |       /
    42    Factor
             |
            3

O en estructura de datos:
  Expr {
    left: Term { value: 42 },
    ops: [Op("+", Term { value: 3 })]
  }
```

### Código Generado para AST

XGrammar genera funciones que:
1. Parsean la entrada
2. Construyen nodos del AST
3. Retornan la raíz

```python
# Generado por XGrammar:

def parse_expr(tokens):
    left = parse_term(tokens)
    ops = []
    while current_token() == "+":
        consume("+")
        ops.append(("add", parse_term(tokens)))
    return Expr(left, ops)

def parse_term(tokens):
    factors = []
    factors.append(parse_factor(tokens))
    while current_token() == "*":
        consume("*")
        factors.append(parse_factor(tokens))
    return Term(factors)
```

## Optimizaciones Finales

Antes de generar código, XGrammar realiza optimizaciones.

### Fusión de Estados

```
Si dos estados tienen exactamente las mismas transiciones:
  Fusionar en un estado
  Actualizar referencias

Antes:
  q3 -a-> q5
  q3 -b-> q6
  q4 -a-> q5  ← mismo que q3
  q4 -b-> q6

Después:
  q3 -a-> q5
  q3 -b-> q6
  (q4 = q3 internamente)
```

### Especialización de Transiciones

```
Patrón común: Alternancia de tokens

Antes:
  q0 -ID-> q1
  q0 -NUM-> q2
  q0 -STR-> q3

Specialización:
  if (token == ID) state = q1
  else if (token == NUM) state = q2
  else if (token == STR) state = q3

Compilador genera código especializado, no table lookup
```

## Ejemplo Completo: JSON Simplificado

Especificación:
```
value = object | array | string | number | "true" | "false" | "null"
object = "{" "}"
array = "[" "]"
string = '"' letter* '"'
number = digit+
letter = "a".."z" | "A".."Z"
digit = "0".."9"
```

Compilación:

```
1. [PARSING] Especificación → AST
2. [NORMALIZATION] → Greibach Normal Form
3. [NFA CONSTRUCTION] → ~50 estados
4. [DFA CONVERSION] → ~30 estados (después minimización)
5. [LOOKAHEAD] → FIRST/FOLLOW calculados
6. [TOKEN MASKING] → Máscaras de bits construidas
7. [INLINING] → 'letter' inlineada en 'string'
8. [DEAD CODE] → Estadows inalcanzables removidos
9. [CODEGEN] → Función parse_value() generada

Resultado: Parser de ~20KB, ejecutable en ~100ns por token
```

## Compilación Incremental

Para gramáticas grandes (kernels complejos), XGrammar soporta compilación **incremental**.

```
Primera compilación:
  grammar_v1.xg → parser_v1

Cambio pequeño:
  grammar_v2.xg (solo una regla modificada)

XGrammar puede:
  - Detectar qué reglas cambiaron
  - Recompilar solo esas partes
  - Reusar autómatas para partes sin cambios
  - Relinkar

Ahorra: 70-80% del tiempo de recompilación en cambios locales
```

## Validación post-compilación

Después de generar el parser, XGrammar **valida** que:

```
1. Parser acepta todos los strings que debería
   Pruebas positivas: ¿Acepta ejemplos válidos?

2. Parser rechaza strings inválidos
   Pruebas negativas: ¿Rechaza contraejemplos?

3. No ambigüedad
   Verifica que cada string tiene árbol único

4. Completitud
   ¿Está cubierta toda la gramática? ¿Hay producciones muertas?

Si cualquier validación falla:
  Emitir advertencia/error
  Sugerir enmiendas
```

## Implementación de Técnicas de Compilación

Vamos a implementar algunas de las técnicas avanzadas de compilación de gramáticas:

```{code-cell} ipython3
# Cálculo de conjuntos FIRST y FOLLOW

from typing import Dict, Set, List
from collections import defaultdict

class GrammarAnalyzer:
    """Analiza gramáticas y calcula FIRST y FOLLOW"""

    def __init__(self):
        self.productions: Dict[str, List[List[str]]] = defaultdict(list)
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = set()
        self.first: Dict[str, Set[str]] = defaultdict(set)
        self.follow: Dict[str, Set[str]] = defaultdict(set)

    def add_production(self, lhs: str, rhs: List[str]):
        """Añade una producción a la gramática"""
        self.productions[lhs].append(rhs)
        self.non_terminals.add(lhs)

        for symbol in rhs:
            if symbol.islower() or symbol in ['+', '*', '(', ')', 'ε']:
                self.terminals.add(symbol)
            else:
                self.non_terminals.add(symbol)

    def compute_first(self):
        """Calcula el conjunto FIRST para todos los símbolos"""
        # Inicializar FIRST para terminales
        for terminal in self.terminals:
            self.first[terminal] = {terminal}

        # Iterar hasta convergencia
        changed = True
        while changed:
            changed = False

            for lhs, alternatives in self.productions.items():
                for rhs in alternatives:
                    # Primera regla: si la producción es X → ε
                    if rhs == ['ε']:
                        if 'ε' not in self.first[lhs]:
                            self.first[lhs].add('ε')
                            changed = True
                        continue

                    # Segunda regla: si la producción es X → Y...
                    for symbol in rhs:
                        # Añadir FIRST(symbol) - {ε} a FIRST(X)
                        new_symbols = self.first[symbol] - {'ε'}
                        if not new_symbols.issubset(self.first[lhs]):
                            self.first[lhs].update(new_symbols)
                            changed = True

                        # Si ε no está en FIRST(symbol), parar
                        if 'ε' not in self.first[symbol]:
                            break
                    else:
                        # Si todos los símbolos pueden derivar ε
                        if 'ε' not in self.first[lhs]:
                            self.first[lhs].add('ε')
                            changed = True

    def compute_follow(self, start_symbol: str):
        """Calcula el conjunto FOLLOW para todos los no-terminales"""
        # Inicializar: $ está en FOLLOW del símbolo inicial
        self.follow[start_symbol].add('$')

        # Iterar hasta convergencia
        changed = True
        while changed:
            changed = False

            for lhs, alternatives in self.productions.items():
                for rhs in alternatives:
                    for i, symbol in enumerate(rhs):
                        if symbol not in self.non_terminals:
                            continue

                        # Si hay símbolos después
                        if i + 1 < len(rhs):
                            next_symbols = rhs[i + 1:]

                            # Añadir FIRST de lo que sigue
                            for next_symbol in next_symbols:
                                new_symbols = self.first[next_symbol] - {'ε'}
                                if not new_symbols.issubset(self.follow[symbol]):
                                    self.follow[symbol].update(new_symbols)
                                    changed = True

                                if 'ε' not in self.first[next_symbol]:
                                    break
                            else:
                                # Todos pueden derivar ε
                                if not self.follow[lhs].issubset(self.follow[symbol]):
                                    self.follow[symbol].update(self.follow[lhs])
                                    changed = True
                        else:
                            # Es el último símbolo
                            if not self.follow[lhs].issubset(self.follow[symbol]):
                                self.follow[symbol].update(self.follow[lhs])
                                changed = True

    def analyze(self, start_symbol: str):
        """Realiza análisis completo"""
        self.compute_first()
        self.compute_follow(start_symbol)

    def display(self):
        """Muestra los conjuntos FIRST y FOLLOW"""
        print("ANÁLISIS DE GRAMÁTICA")
        print("="*60)

        print("\nProducciones:")
        for lhs, alternatives in self.productions.items():
            for rhs in alternatives:
                print(f"  {lhs} → {' '.join(rhs)}")

        print("\nConjuntos FIRST:")
        for symbol in sorted(self.non_terminals):
            print(f"  FIRST({symbol}) = {{{', '.join(sorted(self.first[symbol]))}}}")

        print("\nConjuntos FOLLOW:")
        for symbol in sorted(self.non_terminals):
            print(f"  FOLLOW({symbol}) = {{{', '.join(sorted(self.follow[symbol]))}}}")


# Ejemplo: Gramática de expresiones
print("EJEMPLO 1: Expresiones Aritméticas")
print("="*60)

grammar = GrammarAnalyzer()

# E → T E'
# E' → + T E' | ε
# T → F T'
# T' → * F T' | ε
# F → ( E ) | id

grammar.add_production('E', ['T', "E'"])
grammar.add_production("E'", ['+', 'T', "E'"])
grammar.add_production("E'", ['ε'])
grammar.add_production('T', ['F', "T'"])
grammar.add_production("T'", ['*', 'F', "T'"])
grammar.add_production("T'", ['ε'])
grammar.add_production('F', ['(', 'E', ')'])
grammar.add_production('F', ['id'])

grammar.analyze('E')
grammar.display()
```

```{code-cell} ipython3
# Token Masking: Implementación de máscaras de bits

class TokenMask:
    """Implementa token masking para validación eficiente"""

    def __init__(self, num_tokens: int):
        self.num_tokens = num_tokens
        self.state_masks: Dict[str, int] = {}

    def set_valid_tokens(self, state: str, valid_tokens: List[int]):
        """Establece qué tokens son válidos en un estado"""
        mask = 0
        for token in valid_tokens:
            mask |= (1 << token)
        self.state_masks[state] = mask

    def is_valid(self, state: str, token: int) -> bool:
        """Verifica si un token es válido en un estado dado"""
        if state not in self.state_masks:
            return False
        mask = self.state_masks[state]
        return (mask & (1 << token)) != 0

    def display(self):
        """Muestra las máscaras de forma legible"""
        print("TOKEN MASKING")
        print("="*60)
        print(f"Número de tokens: {self.num_tokens}\n")

        for state, mask in sorted(self.state_masks.items()):
            binary = format(mask, f'0{self.num_tokens}b')
            valid_tokens = [i for i in range(self.num_tokens) if self.is_valid(state, i)]
            print(f"Estado {state}:")
            print(f"  Máscara: {binary} (decimal: {mask})")
            print(f"  Tokens válidos: {valid_tokens}")
            print()


# Ejemplo de uso
print("\nEJEMPLO 2: Token Masking")
print("="*60)

# Mapping de tokens
token_names = {0: 'ID', 1: '+', 2: '*', 3: '(', 4: ')'}
print("\nMapping de tokens:")
for token_id, name in token_names.items():
    print(f"  {token_id}: {name}")

print()

masker = TokenMask(5)

# Estado q0: puede ver ID o (
masker.set_valid_tokens('q0', [0, 3])

# Estado q1: puede ver +, *, )
masker.set_valid_tokens('q1', [1, 2, 4])

# Estado q2: puede ver +, )
masker.set_valid_tokens('q2', [1, 4])

masker.display()

# Probar validación
print("Pruebas de validación:")
test_cases = [
    ('q0', 0, 'ID'),
    ('q0', 3, '('),
    ('q0', 1, '+'),
    ('q1', 1, '+'),
    ('q1', 0, 'ID'),
]

for state, token, name in test_cases:
    valid = masker.is_valid(state, token)
    symbol = "✓" if valid else "✗"
    print(f"  {symbol} Estado {state}, Token {name} ({token}): {'válido' if valid else 'inválido'}")
```

```{code-cell} ipython3
# Construcción de Tabla de Parsing

class ParsingTable:
    """Construye tabla de parsing LL(1)"""

    def __init__(self, grammar: GrammarAnalyzer):
        self.grammar = grammar
        self.table: Dict[tuple, List[str]] = {}

    def build(self):
        """Construye la tabla de parsing"""
        for lhs, alternatives in self.grammar.productions.items():
            for rhs in alternatives:
                # Calcular FIRST de la producción
                first_of_rhs = set()

                for symbol in rhs:
                    if symbol == 'ε':
                        first_of_rhs.add('ε')
                        break

                    first_of_rhs.update(self.grammar.first[symbol] - {'ε'})

                    if 'ε' not in self.grammar.first[symbol]:
                        break
                else:
                    first_of_rhs.add('ε')

                # Para cada terminal en FIRST(rhs)
                for terminal in first_of_rhs - {'ε'}:
                    self.table[(lhs, terminal)] = rhs

                # Si ε está en FIRST(rhs), usar FOLLOW(lhs)
                if 'ε' in first_of_rhs:
                    for terminal in self.grammar.follow[lhs]:
                        self.table[(lhs, terminal)] = rhs

    def display(self):
        """Muestra la tabla de parsing"""
        print("\nTABLA DE PARSING LL(1)")
        print("="*60)

        # Obtener terminales únicos
        terminals = set()
        for (_, terminal) in self.table.keys():
            terminals.add(terminal)
        terminals = sorted(terminals)

        # Encabezado
        print(f"{'':8}", end='')
        for terminal in terminals:
            print(f"{terminal:12}", end='')
        print()
        print("-" * (8 + 12 * len(terminals)))

        # Filas
        for non_terminal in sorted(self.grammar.non_terminals):
            print(f"{non_terminal:8}", end='')
            for terminal in terminals:
                production = self.table.get((non_terminal, terminal), [])
                if production:
                    prod_str = ' '.join(production)
                    print(f"{prod_str:12}", end='')
                else:
                    print(f"{'':12}", end='')
            print()


# Construir tabla de parsing
parsing_table = ParsingTable(grammar)
parsing_table.build()
parsing_table.display()
```

```{code-cell} ipython3
# Simulación de Parser LL(1)

class LL1Parser:
    """Parser LL(1) usando tabla de parsing"""

    def __init__(self, parsing_table: ParsingTable, start_symbol: str):
        self.table = parsing_table
        self.start_symbol = start_symbol

    def parse(self, input_tokens: List[str], verbose: bool = True) -> bool:
        """Parsea una secuencia de tokens"""
        # Inicializar pila con símbolo inicial y $
        stack = ['$', self.start_symbol]
        input_tokens = input_tokens + ['$']
        index = 0

        if verbose:
            print("\nPARSING PASO A PASO")
            print("="*60)
            print(f"Entrada: {' '.join(input_tokens[:-1])}\n")

        step = 0
        while stack:
            step += 1
            top = stack.pop()
            current = input_tokens[index]

            if verbose:
                stack_str = ' '.join(reversed(stack)) if stack else 'vacía'
                remaining = ' '.join(input_tokens[index:])
                print(f"Paso {step}:")
                print(f"  Pila: {stack_str}")
                print(f"  Top: {top}")
                print(f"  Entrada: {remaining}")

            # Si es terminal o $
            if top in self.table.grammar.terminals or top == '$':
                if top == current:
                    if verbose:
                        print(f"  Acción: Coincidir '{top}'")
                    index += 1
                else:
                    if verbose:
                        print(f"  Error: Se esperaba '{top}', se encontró '{current}'")
                    return False
            else:
                # Es no-terminal
                production = self.table.table.get((top, current))
                if production:
                    if verbose:
                        prod_str = ' '.join(production)
                        print(f"  Acción: Aplicar {top} → {prod_str}")

                    # Añadir producción a la pila (en reversa)
                    if production != ['ε']:
                        for symbol in reversed(production):
                            stack.append(symbol)
                else:
                    if verbose:
                        print(f"  Error: No hay producción para [{top}, {current}]")
                    return False

            if verbose:
                print()

        if verbose:
            print("✓ Parsing exitoso!")
        return True


# Probar el parser
print("\nEJEMPLO 3: Parsing con Tabla LL(1)")
print("="*60)

parser = LL1Parser(parsing_table, 'E')

# Probar con diferentes entradas
test_cases = [
    (['id', '+', 'id'], "expresión simple"),
    (['id', '*', 'id', '+', 'id'], "con precedencia"),
    (['(', 'id', '+', 'id', ')'], "con paréntesis"),
]

for tokens, description in test_cases:
    print(f"\nPrueba: {description}")
    print(f"Tokens: {tokens}")
    success = parser.parse(tokens.copy(), verbose=False)
    print(f"Resultado: {'✓ Aceptado' if success else '✗ Rechazado'}")
```

```{code-cell} ipython3
# Optimización: Fusión de Estados

class DFAOptimizer:
    """Optimiza DFAs fusionando estados equivalentes"""

    def __init__(self):
        self.states: Set[str] = set()
        self.transitions: Dict[tuple, str] = {}
        self.accept_states: Set[str] = set()

    def add_state(self, state: str, is_accept: bool = False):
        self.states.add(state)
        if is_accept:
            self.accept_states.add(state)

    def add_transition(self, from_state: str, symbol: str, to_state: str):
        self.transitions[(from_state, symbol)] = to_state

    def find_equivalent_states(self) -> Dict[str, str]:
        """Encuentra estados equivalentes (mismo comportamiento)"""
        equivalent = {}

        # Comparar cada par de estados
        for state1 in self.states:
            for state2 in self.states:
                if state1 >= state2:  # Evitar duplicados
                    continue

                # Verificar si tienen mismo tipo (aceptación o no)
                if (state1 in self.accept_states) != (state2 in self.accept_states):
                    continue

                # Verificar si tienen las mismas transiciones
                same = True
                for symbol in set(s for (_, s) in self.transitions.keys()):
                    trans1 = self.transitions.get((state1, symbol))
                    trans2 = self.transitions.get((state2, symbol))
                    if trans1 != trans2:
                        same = False
                        break

                if same:
                    equivalent[state2] = state1

        return equivalent

    def display(self, title: str = "DFA"):
        """Muestra el DFA"""
        print(f"\n{title}")
        print("="*60)
        print(f"Estados: {sorted(self.states)}")
        print(f"Aceptación: {sorted(self.accept_states)}")
        print("Transiciones:")
        for (from_state, symbol), to_state in sorted(self.transitions.items()):
            print(f"  {from_state} --{symbol}--> {to_state}")


# Ejemplo de optimización
print("\nEJEMPLO 4: Fusión de Estados")
print("="*60)

dfa = DFAOptimizer()
dfa.add_state('q0')
dfa.add_state('q1')
dfa.add_state('q2')
dfa.add_state('q3')
dfa.add_state('q4', is_accept=True)

# q1 y q2 tienen el mismo comportamiento
dfa.add_transition('q0', 'a', 'q1')
dfa.add_transition('q0', 'b', 'q2')
dfa.add_transition('q1', 'c', 'q3')
dfa.add_transition('q2', 'c', 'q3')
dfa.add_transition('q3', 'd', 'q4')

dfa.display("DFA Original")

# Encontrar estados equivalentes
equivalent = dfa.find_equivalent_states()
print(f"\nEstados equivalentes encontrados:")
for state2, state1 in equivalent.items():
    print(f"  {state2} ≡ {state1} (pueden fusionarse)")

if not equivalent:
    print("  Ninguno (DFA ya está minimizado)")
```

```{admonition} Resumen
:class: important
**Conceptos clave:**
- XGrammar normaliza gramáticas a Greibach Normal Form para garantizar progreso predecible
- FIRST/FOLLOW calculan qué tokens son válidos en cada punto del parsing
- Las máscaras de tokens (bitmasks) permiten validación O(1) de tokens válidos por estado
- La construcción de tablas de parsing LL(1) usa FIRST/FOLLOW para decisiones deterministas
- La fusión de estados equivalentes reduce el tamaño del DFA sin cambiar el lenguaje aceptado

**Para la siguiente lectura:**
Exploraremos las limitaciones de las CFGs y lenguajes context-sensitive, entendiendo qué restricciones NO pueden expresarse solo con gramáticas.
```

## Ejercicios

1. **Greibach Normal Form**: Convierte esta gramática a GNF:
   ```
   S → AA | b
   A → a | AS
   ```

2. **Token Masking**: Para estos estados, ¿cuál sería la máscara de bits?
   ```
   Terminal mapping:
     0: ID, 1: "+", 2: "*", 3: "(", 4: ")"

   q0: transiciones a ID, "(", ID
   q1: transiciones a "+", "*", ")"
   ```

3. **Inlining Decision**: ¿Debería inlinearse?
   ```
   hex_digit = "0".."9" | "a".."f"
   color = "#" hex_digit hex_digit hex_digit

   vs

   expr = expr "+" term | term  (muy usado)
   ```

4. **Optimización**: Analiza qué estados podrían fusionarse:
   ```
        ID   NUM
   q0  q1   q2
   q1  err  err
   q2  err  err
   q3  q1   q2   ← ¿igual que q0?
   ```

5. **AST Construction**: Dibuja el AST para `"hello"` usando:
   ```
   value = string | number
   string = '"' letter* '"'
   letter = "a".."z"
   ```

## Preguntas de Reflexión

- ¿Por qué Greibach Normal Form es ventajosa para parsers ejecutables?
- ¿Cuál es el trade-off entre "máscara de bits" y "tabla Hash" para validación de tokens?
- ¿Cómo el inlining afecta la "depurabilidad" del parser generado?
- Para un kernel GPU con millones de tokens, ¿cuáles optimizaciones son más críticas?
