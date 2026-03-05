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

# Parser Generators y Reflexión: De Teoría a Práctica Industrial

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/10-parser-generators-reflexion.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton xgrammar
print('Dependencies installed!')
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Comparar las características principales de ANTLR, Tree-Sitter, Bison, Earley y XGrammar
- Seleccionar la herramienta de parsing adecuada según requisitos del proyecto
- Implementar un parser Earley básico desde cero
- Evaluar trade-offs entre facilidad de uso, performance y flexibilidad en parser generators
- Sintetizar las lecciones clave del módulo de compiladores
```

## El Ecosistema Real

Durante este módulo, hemos estudiado teoría elegante: autómatas, gramáticas, jerarquía de Chomsky. Ahora conectaremos eso con **herramientas reales** que usan miles de proyectos diarios.

## Herramientas Principales: Comparativa

### ANTLR (Another Language Tool Recognition)

**Creador**: Terence Parr (Universidad de San Francisco)

**Características**:
- Genera parsers a partir de EBNF
- Soporta múltiples lenguajes de salida (Java, Python, C#, JavaScript, Go, C++)
- Errores descriptivos con contexto
- Listener/Visitor patterns para AST traversal
- Ampliamente usado (IDE JetBrains, Netflix, Slack)

**Estrategia de Parsing**:
- ALL(*): Adaptive LL with Lookahead
- Puede parsear cualquier CFG (aunque no todas)
- Mejor que LL(k) para gramáticas complejas

```antlr
// Archivo ANTLR: expr.g4
grammar Expr;

expr    : expr '+' term
        | term
        ;

term    : term '*' factor
        | factor
        ;

factor  : '(' expr ')'
        | NUMBER
        ;

NUMBER  : [0-9]+ ;
WS      : [ \t\r\n] -> skip ;
```

**Generación**:
```bash
antlr4 expr.g4 -Dlanguage=Python3
```

Genera: `ExprParser.py`, `ExprLexer.py`, `ExprListener.py`

**Ventajas**:
- Maduro, bien documentado
- Comunidad grande
- Debugging tools

**Desventajas**:
- Overhead para casos simples
- Mensajes de error a veces crípticos
- Compatibilidad de versiones

### Tree-Sitter

**Creador**: Max Brunsfeld (GitHub)

**Propósito**:
- Parser incremental para editores y herramientas
- Reconstruir AST solo de partes cambiadas
- Tolerante a errores (continúa parseando aunque haya syntax error)

**Estrategia**:
- LR-style bottom-up
- Técnica de prioridad de error (error recovery avanzado)

```javascript
// tree-sitter grammar: JavaScript
{
  rules: {
    program: $ => repeat($.statement),
    statement: $ => choice(
      $.variable_declaration,
      $.expression_statement
    ),
    variable_declaration: $ => seq(
      'var',
      $.identifier,
      '=',
      $.expression,
      ';'
    ),
    expression: $ => choice(
      $.number,
      $.identifier,
      seq($.expression, '+', $.expression)
    ),
    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,
    number: $ => /[0-9]+/
  }
}
```

**Ventajas**:
- Incremental (rápido para editores)
- Error recovery (nunca falla completamente)
- Eficiente en memoria

**Desventajas**:
- Menos documentación que ANTLR
- Comunidad más pequeña
- Bindings a lenguajes menos maduro

### Bison / Yacc

**Historia**: Herramientas clásicas (1970s-1980s)

**Estrategia**: LALR(1) parser generation

```yacc
%{
#include <stdio.h>
%}

%token NUMBER
%token PLUS
%token TIMES

%%

expr : expr PLUS term   { $$ = $1 + $3; }
     | term
     ;

term : term TIMES factor { $$ = $1 * $3; }
     | factor
     ;

factor : '(' expr ')'    { $$ = $3; }
       | NUMBER          { $$ = $1; }
       ;

%%
```

**Ventajas**:
- Muy rápido (compilado a C/C++)
- Bien entendido en industria
- Control fino sobre acciones

**Desventajas**:
- Sintaxis críptica
- Poco amigable para principiantes
- Menos flexible que ANTLR

### Earley Parsers

**Algoritmo**: Desarrollado por Jay Earley (1970)

**Característica única**: Parsea **cualquier CFG** sin necesidad de transformación

**Complejidad**:
- O(n³) caso general
- O(n²) para CFGs ambiguas
- O(n) para LL/LR-like

**Ventajas**:
- Muy general
- Menos transformación de gramática
- Errores claros (qué se esperaba vs qué se vio)

**Desventajas**:
- Más lento que LR
- Mayor overhead de memoria

**Usado por**: XGrammar, algunos DSL modernos

```python
# Pseudo-código Earley
def parse(tokens):
    chart = [set() for _ in range(len(tokens) + 1)]

    # Predictor, scanner, completer
    for i in range(len(tokens) + 1):
        while chart[i] changes:
            for item in chart[i]:
                if not_yet_scanned(item):
                    scanner(item, tokens[i])
                if not_yet_predicted(item):
                    predictor(item)
                if completed(item):
                    completer(item)

    return extract_tree(chart)
```

## XGrammar: Posición en el Ecosistema

### ¿Por qué XGrammar es Diferente?

XGrammar es **especializado** para un caso de uso específico:

```
Caso Clásico (ej. ANTLR):
  "Quiero parsear lenguaje X con buenas características"
  Objetivo: Parser general y flexible

XGrammar:
  "Quiero generar código GPU que valide estructura de kernel"
  Objetivo: Parser optimizado + generación de GPU code
  Restricción: CFG solamente
```

### Arquitectura XGrammar

```
Entrada: Especificación de gramática DSL
    ↓
[XGrammar Compiler]
    ├─ Parsea especificación (bootstrap)
    ├─ Valida no-ambigüedad
    ├─ Optimiza (NFA→DFA, minimización)
    ├─ Analiza lookahead
    └─ Genera parser
    ↓
Salida 1: Parser ejecutable (Python, C++)
Salida 2: GPU kernel validator (CUDA)
```

### Ventajas de XGrammar

```
1. Integración GPU nativa
   - Genera kernels que ejecutan parsing en GPU
   - Paralelización automática

2. Optimizaciones especializadas
   - Adaptive token mask cache
   - Compresión de tabla de estados
   - Kernel fusion

3. Garantías teóricas
   - No permite ambigüedad
   - Validación de cobertura

4. Performance
   - Parsing O(n) en GPU
   - Tasa de tokens/segundo alta
```

### Limitaciones de XGrammar

```
1. Solo CFG
   - No puede expresar restricciones context-sensitive
   - Requiere análisis post-parsing para semántica

2. Especializado
   - Optimizado para GPU
   - No ideal para parsers "tradicionales"

3. Comunidad Pequeña
   - Menos tutoriales/ejemplos que ANTLR
   - Menos integración con IDEs

4. Experimentales
   - Más nuevo, menos probado en wild
```

## Comparativa de Decisión

```
┌─────────────────────────────────────────────────────────────┐
│               ¿Cuál herramienta elegir?                      │
└─────────────────────────────────────────────────────────────┘

ANTLR si:
  ✓ Necesitas parsear lenguaje completo (Java, C++, etc.)
  ✓ Errores descriptivos criticales
  ✓ Necesitas soporte de comunidad
  ✓ Presupuesto para learning curve

Tree-Sitter si:
  ✓ Incremental (editor, IDE)
  ✓ Error recovery crucial
  ✓ Lenguajes dinámicos (JS, Python)
  ✓ Rendimiento en editor importante

Bison si:
  ✓ Legacy system (UNIX, compiladores tradicionales)
  ✓ Performance crítica (C/C++)
  ✓ Control fino sobre parsing
  ✓ Ya tienes experiencia

Earley si:
  ✓ Gramática ambigua (generar todos los árboles)
  ✓ Lógica/AI (parsing natural language)
  ✓ Prototipado rápido (menos transformación)

XGrammar si:
  ✓ GPU kernel validation
  ✓ DSL especializado para GPU
  ✓ Performance de parsing crítico
  ✓ Necesitas ejecutar en GPU
```

## Reflexión: Lecciones del Módulo

### 1. Teoría Importa... Pero Seleccionadamente

```
Cosas teóricas que realmente importan:
  ✓ Jerarquía de Chomsky (entender qué es posible)
  ✓ Autómatas (comprender máquinas parsing)
  ✓ Ambigüedad (evitar estructuras malas)
  ✓ Precedencia (operadores correctos)

Cosas teóricas académicas (nice pero no crítico):
  ✗ Minimización Hopcroft (generadores lo hacen)
  ✗ Subset construction (generadores lo hacen)
  ✗ Greibach Normal Form (generadores lo hacen)

Moral: Entender principios ayuda a usar herramientas mejor.
```

### 2. Diseño de Lenguaje es un Oficio

```
No hay una sola forma "correcta".

Ejemplo: ¿Qué es más fácil?
  Java:   if (condition) { statement; }
  Python: if condition: statement

Java: Más símbolos, menos ambigüedad
Python: Menos símbolos, requiere track de indentación

Trade-off:
  Sintaxis visual vs Complejidad parsing
  Facilidad lectura vs Facilidad tipeo
```

### 3. Error Recovery es Infraestimado

```
Errores son inevitables.

Compilador clásico:
  Primer error → termina
  Programador: frustrado

Compilador moderno:
  Error → recupera, continúa buscando más errores
  Compilador output: lista de todos los problemas
  Programador: arregla múltiples cosas en iteración

Lección: Tiempo de desarrollo cae cuando tienes error recovery.
```

### 4. Especificidad es Poderosa

```
Herramienta general (ANTLR):
  Maneja 100 lenguajes
  Overhead para cada uno

Herramienta específica (XGrammar):
  Optimizada para GPU kernels
  ~2-5x más rápido en su dominio

Lección: No siempre quieres general. A veces, específico es mejor.
```

### 5. De CFG a Tipo 1+ es Manual

```
CFG (teórico): A → α
Tipo 1 (context-sensitive): αAβ → αγβ

En práctica:
  - Escribes CFG
  - Generador produce parser
  - Análisis semántico manual (código)

No hay "generador de Type 1 parser universal".
Necesitas código custom para cada restricción.

Lección: Conocer limitaciones de CFG ayuda a planificar.
```

## Síntesis: Del Compilador Perfecto

Si fueras a diseñar el compilador "perfecto":

```
Fase 1: Lexing (DFA)
  Entrada: Caracteres
  Salida: Tokens
  Herramienta: flex/lex (o hand-written)

Fase 2: Parsing (CFG)
  Entrada: Tokens
  Salida: AST
  Herramienta: ANTLR/Bison/XGrammar
  Propiedad: No-ambiguo, error recovery

Fase 3: Análisis Semántico (Custom)
  Entrada: AST
  Salida: AST anotado, tabla de símbolos
  Tareas:
    - Type checking
    - Scope resolution
    - Semantic validation
  Código: Custom por lenguaje

Fase 4: Optimización (Custom)
  Entrada: AST anotado
  Salida: AST optimizado
  Técnicas: Constant folding, dead code elim, etc.
  Código: Custom por lenguaje

Fase 5: Codegen (Custom)
  Entrada: AST optimizado
  Salida: Código ejecutable (máquina, GPU, otro lenguaje)
  Código: Custom por target
```

## Trend Moderno: Parser Combinators

En lugar de generadores, algunos lenguajes modernos (Rust, Scala) favorecen **parser combinators**:

```rust
// Rust: nom parser combinator
use nom::IResult;
use nom::character::complete::{char, digit1};

fn number(input: &str) -> IResult<&str, i32> {
    let (input, num) = digit1(input)?;
    Ok((input, num.parse().unwrap()))
}

fn factor(input: &str) -> IResult<&str, i32> {
    nom::branch::alt((
        nom::sequence::delimited(char('('), expr, char(')')),
        number
    ))(input)
}

fn expr(input: &str) -> IResult<&str, i32> {
    let (input, left) = term(input)?;
    // ... parsear más
}
```

**Ventaja**: Parsers son **composables**, testeable, sin generación.
**Desventaja**: Overhead si número de combinaciones explosivo.

## Implementación de un Parser Earley

Vamos a implementar un parser Earley desde cero para entender cómo funciona:

```{code-cell} ipython3
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple

@dataclass(frozen=True)
class EarleyItem:
    """
    Representa un item de Earley: [A → α • β, i]
    - rule_name: nombre de la regla (ej: 'E')
    - production: la producción completa
    - dot_position: posición del punto •
    - start_position: índice donde empezó esta derivación
    """
    rule_name: str
    production: Tuple[str, ...]
    dot_position: int
    start_position: int

    def __repr__(self):
        prod_with_dot = list(self.production)
        prod_with_dot.insert(self.dot_position, '•')
        prod_str = ' '.join(prod_with_dot)
        return f"[{self.rule_name} → {prod_str}, {self.start_position}]"

    def next_symbol(self) -> Optional[str]:
        """Retorna el símbolo después del punto, o None si está al final"""
        if self.dot_position < len(self.production):
            return self.production[self.dot_position]
        return None

    def is_complete(self) -> bool:
        """Verifica si el punto está al final"""
        return self.dot_position >= len(self.production)


class EarleyParser:
    """Parser Earley que acepta cualquier CFG"""

    def __init__(self):
        self.grammar: Dict[str, List[Tuple[str, ...]]] = {}
        self.start_symbol: str = None

    def add_rule(self, lhs: str, rhs: List[str]):
        """Añade una regla a la gramática"""
        if lhs not in self.grammar:
            self.grammar[lhs] = []
        self.grammar[lhs].append(tuple(rhs))

        if self.start_symbol is None:
            self.start_symbol = lhs

    def parse(self, tokens: List[str], verbose: bool = False) -> bool:
        """Parsea una secuencia de tokens usando el algoritmo Earley"""
        n = len(tokens)
        chart: List[Set[EarleyItem]] = [set() for _ in range(n + 1)]

        # Inicializar chart[0] con reglas del símbolo inicial
        for production in self.grammar.get(self.start_symbol, []):
            chart[0].add(EarleyItem(self.start_symbol, production, 0, 0))

        # Procesar cada posición
        for i in range(n + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Chart[{i}]:")
                print(f"{'='*60}")

            # Procesar items hasta que no haya cambios
            changed = True
            while changed:
                changed = False
                current_items = list(chart[i])

                for item in current_items:
                    if verbose and item not in chart[i]:
                        continue

                    next_sym = item.next_symbol()

                    if item.is_complete():
                        # COMPLETER
                        new_items = self._completer(item, chart)
                        for new_item in new_items:
                            if new_item not in chart[i]:
                                chart[i].add(new_item)
                                changed = True
                                if verbose:
                                    print(f"  Complete: {new_item}")

                    elif next_sym and next_sym in self.grammar:
                        # PREDICTOR
                        new_items = self._predictor(next_sym, i)
                        for new_item in new_items:
                            if new_item not in chart[i]:
                                chart[i].add(new_item)
                                changed = True
                                if verbose:
                                    print(f"  Predict: {new_item}")

                    elif next_sym and i < n and next_sym == tokens[i]:
                        # SCANNER
                        new_item = self._scanner(item, i + 1)
                        if new_item not in chart[i + 1]:
                            chart[i + 1].add(new_item)
                            if verbose:
                                print(f"  Scan '{tokens[i]}': {new_item}")

        # Verificar si hay un item completo que representa el parsing exitoso
        for item in chart[n]:
            if (item.rule_name == self.start_symbol and
                item.is_complete() and
                item.start_position == 0):
                return True

        return False

    def _predictor(self, non_terminal: str, position: int) -> List[EarleyItem]:
        """Predictor: Añade items para las reglas del no-terminal"""
        items = []
        for production in self.grammar.get(non_terminal, []):
            items.append(EarleyItem(non_terminal, production, 0, position))
        return items

    def _scanner(self, item: EarleyItem, next_position: int) -> EarleyItem:
        """Scanner: Mueve el punto sobre un terminal"""
        return EarleyItem(
            item.rule_name,
            item.production,
            item.dot_position + 1,
            item.start_position
        )

    def _completer(self, completed_item: EarleyItem, chart: List[Set[EarleyItem]]) -> List[EarleyItem]:
        """Completer: Propaga items completados"""
        items = []
        start_pos = completed_item.start_position

        for item in chart[start_pos]:
            next_sym = item.next_symbol()
            if next_sym == completed_item.rule_name:
                items.append(EarleyItem(
                    item.rule_name,
                    item.production,
                    item.dot_position + 1,
                    item.start_position
                ))

        return items


# Crear parser para gramática de expresiones
print("PARSER EARLEY: Expresiones Aritméticas")
print("="*60)

parser = EarleyParser()

# Gramática:
# E → E + T | T
# T → T * F | F
# F → ( E ) | id

parser.add_rule('E', ['E', '+', 'T'])
parser.add_rule('E', ['T'])
parser.add_rule('T', ['T', '*', 'F'])
parser.add_rule('T', ['F'])
parser.add_rule('F', ['(', 'E', ')'])
parser.add_rule('F', ['id'])

print("\nGramática:")
for lhs, productions in parser.grammar.items():
    for prod in productions:
        print(f"  {lhs} → {' '.join(prod)}")

# Probar parser
test_cases = [
    ['id'],
    ['id', '+', 'id'],
    ['id', '*', 'id', '+', 'id'],
    ['(', 'id', '+', 'id', ')', '*', 'id'],
    ['id', '+'],  # Error
    ['(', 'id'],  # Error
]

print("\n" + "="*60)
print("PRUEBAS")
print("="*60)

for tokens in test_cases:
    result = parser.parse(tokens)
    status = "✓ Aceptado" if result else "✗ Rechazado"
    print(f"{status}: {' '.join(tokens)}")
```

```{code-cell} ipython3
# Ejemplo detallado con verbose=True

print("\nPARSING DETALLADO: 'id + id'")
print("="*60)

parser2 = EarleyParser()
parser2.add_rule('E', ['E', '+', 'T'])
parser2.add_rule('E', ['T'])
parser2.add_rule('T', ['id'])

result = parser2.parse(['id', '+', 'id'], verbose=True)

print("\n" + "="*60)
print(f"Resultado final: {'✓ Aceptado' if result else '✗ Rechazado'}")
print("="*60)
```

## Comparación de Estrategias de Parsing

```{code-cell} ipython3
import time
from abc import ABC, abstractmethod

class ParserBenchmark(ABC):
    """Clase base para benchmarking de parsers"""

    @abstractmethod
    def parse(self, tokens: List[str]) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class RecursiveDescentParser(ParserBenchmark):
    """Parser recursivo descendente (LL)"""

    def name(self) -> str:
        return "Recursivo Descendente (LL)"

    def parse(self, tokens: List[str]) -> bool:
        self.tokens = tokens
        self.position = 0
        try:
            self.parse_expr()
            return self.position == len(tokens)
        except:
            return False

    def current_token(self) -> Optional[str]:
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def consume(self, expected: str):
        if self.current_token() == expected:
            self.position += 1
        else:
            raise SyntaxError(f"Expected {expected}")

    def parse_expr(self):
        self.parse_term()
        while self.current_token() == '+':
            self.consume('+')
            self.parse_term()

    def parse_term(self):
        self.parse_factor()
        while self.current_token() == '*':
            self.consume('*')
            self.parse_factor()

    def parse_factor(self):
        if self.current_token() == '(':
            self.consume('(')
            self.parse_expr()
            self.consume(')')
        elif self.current_token() == 'id':
            self.consume('id')
        else:
            raise SyntaxError("Expected factor")


class EarleyParserBenchmark(ParserBenchmark):
    """Wrapper para el parser Earley"""

    def __init__(self):
        self.parser = EarleyParser()
        self.parser.add_rule('E', ['E', '+', 'T'])
        self.parser.add_rule('E', ['T'])
        self.parser.add_rule('T', ['T', '*', 'F'])
        self.parser.add_rule('T', ['F'])
        self.parser.add_rule('F', ['(', 'E', ')'])
        self.parser.add_rule('F', ['id'])

    def name(self) -> str:
        return "Earley"

    def parse(self, tokens: List[str]) -> bool:
        return self.parser.parse(tokens)


def benchmark_parsers(parsers: List[ParserBenchmark], test_cases: List[List[str]], iterations: int = 100):
    """Compara el rendimiento de diferentes parsers"""
    print("BENCHMARK DE PARSERS")
    print("="*60)

    results = {}

    for parser in parsers:
        print(f"\nProbando: {parser.name()}")
        total_time = 0

        for test_case in test_cases:
            start = time.time()
            for _ in range(iterations):
                parser.parse(test_case)
            elapsed = time.time() - start
            total_time += elapsed

        avg_time = total_time / len(test_cases) / iterations * 1000  # ms
        results[parser.name()] = avg_time
        print(f"  Tiempo promedio: {avg_time:.4f} ms por parsing")

    # Mostrar comparación
    print("\n" + "="*60)
    print("COMPARACIÓN")
    print("="*60)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    fastest = sorted_results[0][1]

    for name, time_ms in sorted_results:
        speedup = time_ms / fastest
        print(f"{name:30} {time_ms:8.4f} ms  ({speedup:.2f}x)")


# Crear test cases
test_cases = [
    ['id'],
    ['id', '+', 'id'],
    ['id', '*', 'id', '+', 'id'],
    ['(', 'id', '+', 'id', ')', '*', 'id'],
    ['id', '+', 'id', '+', 'id', '+', 'id'],
]

# Ejecutar benchmark
parsers = [
    RecursiveDescentParser(),
    EarleyParserBenchmark(),
]

benchmark_parsers(parsers, test_cases, iterations=1000)
```

## Visualización de Herramientas de Parsing

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

# Comparar características de diferentes herramientas

tools = ['ANTLR', 'Tree-Sitter', 'Bison', 'Earley', 'XGrammar']

# Características (escala 1-10)
characteristics = {
    'Facilidad de uso': [8, 6, 4, 7, 6],
    'Performance': [7, 9, 10, 5, 9],
    'Flexibilidad': [9, 7, 6, 10, 6],
    'Comunidad': [10, 7, 8, 4, 3],
    'Error recovery': [8, 10, 4, 6, 5],
}

# Crear gráfico de radar
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Calcular ángulos
categories = list(characteristics.keys())
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plotear cada herramienta
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for i, tool in enumerate(tools):
    values = [characteristics[cat][i] for cat in categories]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=tool, color=colors[i])
    ax.fill(angles, values, alpha=0.15, color=colors[i])

# Configurar gráfico
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
ax.grid(True)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Comparación de Herramientas de Parsing\n', size=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\nINTERPRETACIÓN:")
print("="*60)
print("ANTLR: Mejor balance general, excelente comunidad")
print("Tree-Sitter: Mejor performance y error recovery (ideal para editores)")
print("Bison: Máxima performance pero menos amigable")
print("Earley: Máxima flexibilidad pero más lento")
print("XGrammar: Alta performance pero comunidad pequeña")
```

```{code-cell} ipython3
# Tabla de decisión interactiva

class ParserSelector:
    """Ayuda a seleccionar la herramienta de parsing adecuada"""

    def __init__(self):
        self.criteria = {
            'performance': 0,
            'ease_of_use': 0,
            'flexibility': 0,
            'community': 0,
            'error_recovery': 0,
        }

    def recommend(self) -> str:
        """Recomienda una herramienta basada en criterios"""
        scores = {
            'ANTLR': 0,
            'Tree-Sitter': 0,
            'Bison': 0,
            'Earley': 0,
            'XGrammar': 0,
        }

        # Weights para cada herramienta
        weights = {
            'ANTLR': {'performance': 7, 'ease_of_use': 8, 'flexibility': 9,
                     'community': 10, 'error_recovery': 8},
            'Tree-Sitter': {'performance': 9, 'ease_of_use': 6, 'flexibility': 7,
                           'community': 7, 'error_recovery': 10},
            'Bison': {'performance': 10, 'ease_of_use': 4, 'flexibility': 6,
                     'community': 8, 'error_recovery': 4},
            'Earley': {'performance': 5, 'ease_of_use': 7, 'flexibility': 10,
                      'community': 4, 'error_recovery': 6},
            'XGrammar': {'performance': 9, 'ease_of_use': 6, 'flexibility': 6,
                        'community': 3, 'error_recovery': 5},
        }

        # Calcular scores
        for tool, tool_weights in weights.items():
            for criterion, importance in self.criteria.items():
                scores[tool] += tool_weights[criterion] * importance

        # Retornar el mejor
        best_tool = max(scores.items(), key=lambda x: x[1])
        return best_tool[0], scores

    def display_recommendation(self):
        """Muestra recomendación con explicación"""
        print("RECOMENDACIÓN DE HERRAMIENTA")
        print("="*60)

        print("\nCriterios seleccionados:")
        for criterion, importance in self.criteria.items():
            stars = "★" * importance + "☆" * (5 - importance)
            print(f"  {criterion.replace('_', ' ').title():20} {stars}")

        best_tool, scores = self.recommend()

        print("\nPuntuaciones:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for tool, score in sorted_scores:
            bar = "█" * int(score / 10)
            print(f"  {tool:15} {bar} ({score:.1f})")

        print(f"\n✓ Recomendación: {best_tool}")


# Ejemplo de uso
print("\nEJEMPLO 1: Necesito parsear un lenguaje completo con buen soporte")
selector1 = ParserSelector()
selector1.criteria = {
    'performance': 3,
    'ease_of_use': 5,
    'flexibility': 4,
    'community': 5,
    'error_recovery': 3,
}
selector1.display_recommendation()

print("\n" + "="*60)
print("\nEJEMPLO 2: Necesito máxima performance para producción")
selector2 = ParserSelector()
selector2.criteria = {
    'performance': 5,
    'ease_of_use': 2,
    'flexibility': 3,
    'community': 3,
    'error_recovery': 2,
}
selector2.display_recommendation()

print("\n" + "="*60)
print("\nEJEMPLO 3: Parser incremental para editor de código")
selector3 = ParserSelector()
selector3.criteria = {
    'performance': 4,
    'ease_of_use': 3,
    'flexibility': 3,
    'community': 3,
    'error_recovery': 5,
}
selector3.display_recommendation()
```

## Lecciones Aprendidas: Visualización

```{code-cell} ipython3
# Resumen visual de conceptos clave

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Lecciones Clave de Compiladores', fontsize=16, fontweight='bold')

# 1. Jerarquía de Chomsky
ax1 = axes[0, 0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Jerarquía de Chomsky', fontweight='bold')

layers = [
    (2, 1.5, 'Tipo 3: Regular\n(DFA/NFA)', '#E3F2FD'),
    (1.5, 3, 'Tipo 2: CFG\n(PDA)', '#C8E6C9'),
    (1, 4.5, 'Tipo 1: CSG\n(LBA)', '#FFF9C4'),
    (0.5, 6, 'Tipo 0: RE\n(Turing)', '#FFCCBC'),
]

y = 2
for width, height, label, color in layers:
    x = 5 - width
    rect = FancyBboxPatch((x, y), width * 2, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(5, y + height/2, label, ha='center', va='center', fontsize=10)
    y += height

ax1.text(5, 0.5, 'Poder expresivo ↑', ha='center', fontsize=10, style='italic')

# 2. Fases del compilador
ax2 = axes[0, 1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Fases de Compilación', fontweight='bold')

phases = ['Lexing', 'Parsing', 'Semántica', 'Optimización', 'Codegen']
colors = ['#BBDEFB', '#C5CAE9', '#D1C4E9', '#F8BBD0', '#FFCCBC']

y = 8.5
for phase, color in zip(phases, colors):
    rect = Rectangle((2, y), 6, 1, facecolor=color, edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(5, y + 0.5, phase, ha='center', va='center', fontsize=11, fontweight='bold')

    if y > 1:
        arrow = FancyArrowPatch((5, y), (5, y - 0.5),
                              arrowstyle='->', lw=2, color='black')
        ax2.add_patch(arrow)
    y -= 1.5

# 3. Trade-offs
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Trade-offs Importantes', fontweight='bold')

tradeoffs = [
    ('Poder expresivo', 'Eficiencia parsing'),
    ('Facilidad diseño', 'Performance'),
    ('Flexibilidad', 'Simplicidad'),
    ('Generalidad', 'Especialización'),
]

y = 8
for left, right in tradeoffs:
    # Left
    ax3.add_patch(Rectangle((0.5, y), 3.5, 1, facecolor='#FFE082', edgecolor='black'))
    ax3.text(2.25, y + 0.5, left, ha='center', va='center', fontsize=9)

    # Arrow
    ax3.annotate('', xy=(8.5, y + 0.5), xytext=(4.5, y + 0.5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))

    # Right
    ax3.add_patch(Rectangle((6, y), 3.5, 1, facecolor='#CE93D8', edgecolor='black'))
    ax3.text(7.75, y + 0.5, right, ha='center', va='center', fontsize=9)

    y -= 1.8

# 4. Herramientas recomendadas
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Guía de Selección', fontweight='bold')

recommendations = [
    ('ANTLR', 'Balance general\nProducción', '#4CAF50'),
    ('Tree-Sitter', 'Editores\nIncremental', '#2196F3'),
    ('Bison', 'Performance\nLegacy', '#FF9800'),
    ('Earley', 'Prototipado\nFlexibilidad', '#9C27B0'),
]

y = 8
for tool, use_case, color in recommendations:
    ax4.add_patch(FancyBboxPatch((1, y), 8, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='black',
                                linewidth=2, alpha=0.7))
    ax4.text(2, y + 0.65, tool, ha='left', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax4.text(7.5, y + 0.65, use_case, ha='right', va='center',
            fontsize=9, color='white')
    y -= 1.8

plt.tight_layout()
plt.show()

print("\nCONCLUSIONES CLAVE:")
print("="*60)
print("1. La jerarquía de Chomsky delimita qué es expresable")
print("2. Cada fase del compilador tiene responsabilidad específica")
print("3. Los trade-offs son inevitables - elige según tu caso de uso")
print("4. No hay una herramienta 'mejor' - depende del contexto")
print("5. CFG es el 'sweet spot' para lenguajes de programación")
```

```{admonition} 🤔 Reflexiona
:class: hint
¿Por qué CFG es el "sweet spot" para lenguajes de programación? Considera el balance entre poder expresivo (puede expresar estructuras anidadas) y eficiencia (parsing O(n) o O(n³) vs indecidible).
```

```{admonition} Resumen
:class: important
**Conceptos clave del módulo completo:**
- Los compiladores transforman código fuente a ejecutable mediante fases bien definidas (lexing, parsing, semántica, optimización, codegen)
- La Jerarquía de Chomsky delimita qué es expresable: Regular ⊂ CFG ⊂ CSG ⊂ RE
- DFAs reconocen patrones simples; CFGs reconocen estructuras anidadas; CSG añade contexto
- XGrammar usa CFG + análisis semántico para validar y generar kernels GPU sintácticamente correctos
- Las herramientas industriales (ANTLR, Tree-Sitter, Bison, Earley, XGrammar) ofrecen diferentes trade-offs según el caso de uso

**Aplicación al proyecto:**
Has aprendido a diseñar gramáticas DSL para kernels GPU, compilarlas a parsers eficientes, y entender las limitaciones que requieren análisis adicional. Este conocimiento es fundamental para usar XGrammar efectivamente en generación de código con LLMs.
```

```{admonition} ✅ Verifica tu comprensión final
:class: note
1. ¿Cuáles son las 5 lecciones más importantes que aprendiste sobre compiladores?
2. ¿En qué casos elegirías ANTLR vs Tree-Sitter vs XGrammar?
3. ¿Qué restricciones puede expresar una CFG y cuáles no?
4. ¿Cómo XGrammar usa el pipeline de compilación para generar parsers optimizados?
```

## Ejercicios

1. **Elección de Herramienta**: Para estos DSLs, ¿qué herramienta elegirías?
   - SQL query DSL
   - LaTeX math formula
   - GPU kernel specification
   - Natural language instructions

2. **ANTLR Práctico**: Escribe reglas ANTLR para:
   ```
   variable_assignment = ID "=" expression ";"
   expression = term ("+" term)*
   term = factor ("*" factor)*
   factor = "(" expression ")" | number
   number = [0-9]+
   ```

3. **Comparativa de Errores**: Para este error:
   ```
   x = 5 + ;
   ```
   Cómo diferentes herramientas lo reportarían.

4. **Optimización XGrammar**: Para una gramática con 50 tokens, ¿bitmap o hash table sería mejor para token masking?

5. **Reflexión Comparativa**: ¿Cuáles son las 3 cosas más importantes que aprendiste sobre compiladores?

## Preguntas de Reflexión

- ¿Cuál es la razón **realmente profunda** por la que CFG es el "sweet spot" para lenguajes de programación?

- Si pudieras agregar una característica a XGrammar, ¿qué sería? ¿Por qué?

- En el contexto de generación de GPU kernels: ¿Cuáles restricciones son mejor expresadas como gramática, y cuáles como código semántico?

- ¿Cómo la evolución de GPUs (más potentes, más paralelas) cambiaría la arquitectura ideal de un compilador para kernels?

- Si tuvieras que enseñar compiladores a alguien en 1 semana, ¿cuáles 5 conceptos clave enseñarías?

## Recursos Adicionales (Lecturas Sugeridas)

Aunque fuera del curso, si quieres profundizar:

**Clásicos**:
- "Compilers: Principles, Techniques, and Tools" (Dragon Book) - exhaustivo
- "Engineering a Compiler" - más práctico que Dragon Book
- "Modern Compiler Implementation" - accesible

**Específicos**:
- ANTLR oficial docs: antlr.org
- Earley parser overview: Erik Demaine MIT OpenCourseWare
- GPU compilation: NVIDIA CUDA programming guide

**Práctico**:
- Lean sobre escritura de parsers en tu lenguaje favorito
- Implementa un parser recursivo descendente para calc simple
- Usa ANTLR para pequeño proyecto

## Conclusión

Has aprendido que compiladores no son "magia negra". Son:

```
1. Máquinas (autómatas)
   - DFAs para tokens
   - PDAs para estructura

2. Matemáticas (gramáticas)
   - CFGs expresan exactamente lo parseabel sin ambigüedad
   - Jerarquía de Chomsky delimita qué es posible

3. Algoritmos (parsing)
   - Top-down vs bottom-up son trade-offs
   - Lookahead reduce ambigüedad

4. Ingeniería (tools & design)
   - Generadores automatizan lo tedioso
   - Diseño de lenguaje es arte, no ciencia
   - Trade-offs son inevitables

5. Herramientas (ANTLR, XGrammar, etc.)
   - Usan principios anteriores eficientemente
   - Eleges la herramienta según tu problema específico
```

En el contexto de XGrammar para GPU kernels: tienes ahora las bases para entender cómo funciona, diseñar gramáticas, y saber cuándo las limitaciones de CFG requieren análisis adicional.

La compilación es donde **teoría conoce práctica**. Espero que este módulo te haya mostrado ambos lados.

---

## Referencias

- ANTLR. [ANother Tool for Language Recognition](https://www.antlr.org/). ANTLR.
- Parr, T. (2013). The Definitive ANTLR 4 Reference. Pragmatic Bookshelf.
- Tree-sitter. [An Incremental Parsing System](https://tree-sitter.github.io/tree-sitter/). Tree-sitter.
- GNU Bison. [General-purpose Parser Generator](https://www.gnu.org/software/bison/). GNU.
- XGrammar. [Efficient, Flexible and Portable Structured Generation](https://github.com/mlc-ai/xgrammar). GitHub.
