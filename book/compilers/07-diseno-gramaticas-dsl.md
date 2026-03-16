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

# Diseño de Gramáticas DSL: Principios y Práctica

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/07-diseno-gramaticas-dsl.ipynb)
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
- Aplicar el principio de diseño incremental bottom-up para construir gramáticas robustas
- Estructurar gramáticas en niveles jerárquicos para codificar precedencia correctamente
- Identificar y resolver ambigüedades mediante refactorización de gramáticas
- Aplicar left-factoring para optimizar parsers LL
- Diseñar gramáticas modulares en capas (L0-L4) para facilitar mantenimiento
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[Domain Specific Languages (Martin Fowler)](https://www.youtube.com/watch?v=r3_u48J-bFE)** - Qué son los DSLs como Triton, explicados desde la perspectiva de su creador.
```

## El Arte, No Solo la Ciencia

Hemos cubierto la **teoría** de compiladores. Ahora viene la parte desafiante: **diseñar buenas gramáticas**.

Escribir una gramática es como escribir código. Puedes teclear sintaxis correcta que compile pero sea horrible: ambigua, ineficiente, difícil de mantener.

Una buena gramática:
- Expresa exactamente lo que quieres (no más, no menos)
- Es eficiente de parsear
- Es fácil de entender y mantener
- Evita ambigüedades

## Principio 1: Comienza Mínimo, Extiende Gradualmente

**Tentación**: Escribir una gramática completa de una vez.

**Realidad**: Las gramáticas son como escaleras. Subes paso a paso.

### Enfoque Bottom-Up: Empezar Simple

```
Fase 1: Lo más básico
  number = digit+
  digit = "0".."9"

Pruebas: ¿Funciona? Parseramos "123" ✓

Fase 2: Agregamos operaciones
  expr = number ("+" number)*

Pruebas: ¿Parsea "1+2+3"? ✓

Fase 3: Agregamos multiplicación
  expr = term ("+" term)*
  term = number ("*" number)*

Pruebas: ¿Respeta precedencia? 1+2*3 = 1+(2*3)? ✓

Fase 4: Agregamos paréntesis
  term = factor ("*" factor)*
  factor = "(" expr ")" | number

Pruebas: ¿Parsea "(1+2)*3"? ✓

Fase 5: Agregamos variables
  factor = "(" expr ")" | identifier | number
  identifier = letter (letter | digit | "_")*
  letter = "a".."z" | "A".."Z" | "_"

Pruebas: ¿Parsea "x + y * 2"? ✓
```

Cada paso añade **una** característica pequeña.

### Ventajas

```
✓ Pruebas incrementales detectan problemas rápido
✓ Fácil debuggear (la nueva característica causó el error)
✓ Más fácil para otros entender el desarrollo
✓ Cambios locales no rompen lo anterior
```

### Contraejemplo: Engullir Demasiado

```
Mal: Escribir todo de una:

cfg = (rule)+
rule = identifier "=" body
body = term ("OR" term)*
term = factor ("AND" factor)*
factor = "(" body ")" | "!" factor | atomic
atomic = identifier | literal | character_class | quantified
quantified = atomic "{" number "}"
           | atomic "{" number "," number "}"
           | atomic "*" | atomic "+" | atomic "?"
character_class = "[" char_range ("," char_range)* "]"
char_range = char | char ".." char
literal = '"' string '"'
...

Resultado: Confuso, difícil de depurar si no funciona
```

## Principio 2: Separar Niveles de Precedencia

Este es **tan importante** que dedicamos secciones enteras a ello.

### Jerarquía de Precedencia Típica

Para operadores aritméticos, de menor a mayor precedencia:

```
Nivel 1: Asignación (=)
Nivel 2: OR (|)
Nivel 3: AND (&)
Nivel 4: Comparación (==, !=, <, >)
Nivel 5: Suma/Resta (+, -)
Nivel 6: Multiplicación/División (*, /)
Nivel 7: Exponenciación (^)
Nivel 8: Unarios (-, !, ~)
Nivel 9: Indexación/Llamadas ([], ())
Nivel 10: Átomos (literales, variables)
```

En una gramática, esto se refleja como:

```
statement = assignment
assignment = or_expr ("=" or_expr)*
or_expr = and_expr ("|" and_expr)*
and_expr = comparison ("&" comparison)*
comparison = add_sub (("==" | "!=" | "<" | ">") add_sub)*
add_sub = mul_div (("+" | "-") mul_div)*
mul_div = power (("*" | "/") power)*
power = unary ("^" unary)*
unary = "-" unary | "!" unary | primary
primary = "(" expression ")" | literal | identifier
```

### Por qué Esto Funciona

```
Estructura refleja precedencia:
- Raíz (statement) = baja precedencia
- Hojas (primary) = alta precedencia

Ejemplo: a + b * c

Derivación:
  statement → assignment → or_expr → and_expr → comparison → add_sub
            → mul_div (b * c) + mul_div(a)

El árbol automáticamente agrupa (b*c) juntos
porque mul_div está más profundo que add_sub.
```

### Codificación de Precedencia

```
¿RECUERDO?
  expr → expr "+" expr   (ambiguo, MALO)
  expr → expr "*" expr

¿BIEN?
  expr → expr "+" term | term    (+ tiene baja prec)
  term → term "*" factor | factor (* tiene alta prec)
  factor → number | "(" expr ")"

La estructura **es** la precedencia.
```

## Principio 3: Diseño Modular en Capas

Para gramáticas complejas (como una DSL de kernel GPU), usamos **diseño en capas**.

### Estructura L1-L4

Adaptado para XGrammar, pensamos en niveles:

```
L4: AST (Abstract Syntax Tree)
    Lo que semánticamente significa

L3: Construcciones sintácticas de alto nivel
    bloques, funciones, declaraciones

L2: Construcciones sintácticas de nivel medio
    expresiones, operadores, control de flujo

L1: Tokens e construcciones básicas
    literales, identificadores, palabras clave

L0: Léxica
    caracteres, patrones básicos

Ejemplo en kernel Triton:

L4: [BlockProgram(kernel, grid_def, ...)]

L3: kernel = "def" name "(" params ")" ":" block
    grid = "@triton.jit\ndef launch(...): ..."

L2: expr = term (("+" | "-") term)*
    statement = assignment | loop | condition

L1: literal = number | string
    identifier = letter (letter|digit|"_")*
    keyword = "def" | "for" | "if" | ...

L0: letter = "a".."z" | "A".."Z" | "_"
    digit = "0".."9"
```

### Ventajas de Separación en Capas

```
✓ Independencia: Cambiar L1 no afecta L3
✓ Reuso: Reglas de L1 usadas en múltiples L2
✓ Claridad: Cada nivel tiene responsabilidad clara
✓ Testing: Probar cada nivel aisladamente
```

:::{figure} diagrams/grammar_layers.png
:name: fig-grammar-layers
:alt: Diagrama de capas de una gramática DSL con niveles L0-L4
:align: center
:width: 90%

**Figura 1:** Diseño jerárquico en capas (L0-L4) de una gramática DSL. Cada capa tiene responsabilidades claras: léxica (L0), tokens (L1), expresiones (L2), construcciones sintácticas (L3), y semántica (L4).
:::

## Principio 4: Evitar Ambigüedad

La ambigüedad es el **enemigo** de una buena gramática.

### Detección de Ambigüedad

Una gramática es ambigua si existe un string con **dos o más árboles de análisis distintos**.

```
Gramática ambigua:
  expr → expr "+" expr | expr "*" expr | number

Para "1+2*3":
Árbol 1: (1+2)*3 = 9
Árbol 2: 1+(2*3) = 7

Ambiguo: No sabemos qué significa.
```

### Resolución Tipo 1: Reescribir Gramática

```
Solución:
  expr → term ("+" term)*
  term → factor ("*" factor)*
  factor → number

Ahora solo un árbol para "1+2*3": 1+(2*3)
Precedencia codificada en estructura.
```

### Resolución Tipo 2: Directivas de Desambiguación

Algunos parseadores permiten directivas:

```
expr → expr "+" expr    %left      ← asociatividad izquierda
     | expr "*" expr    %left
     | number

Interpretación: En caso de ambigüedad, reducir izquierda.
1+2+3 → ((1+2)+3) ✓ no (1+(2+3))
```

XGrammar típicamente **no permite ambigüedad** - debes escribir gramáticas no-ambiguas.

## Principio 5: Caracteres de Escape y Delimitadores

A menudo necesitas capturar strings literales o caracteres especiales.

### Problema

```
Especificación:
  rule = name "=" body

¿Qué pasa si el usuario escribe?
  rule = "rule = exp"

Confusión: ¿El "=" es parte del string o delimitador de la regla?
```

### Solución: Escapar Explícitamente

```
EBNF estándar:
  string = '"' char* '"'
  char = /* cualquier char excepto comillas, o escaped */

Ejemplo:
  rule = "rule = exp"  ← string literal
  name = "my rule"    ← string con espacios

Y dentro, escapes:
  string = '"' (not_quote | "\\" any)* '"'
```

### Delimitadores

Para código embebido (como Triton en GPU):

```
code_block = "```" code "```"
code = /* cualquier cosa excepto ``` */

O:
code_block = "{{" code "}}"
```

## Principio 6: Left-Factoring para Parsers Eficientes

Si tienes múltiples reglas que comparten prefijo, factoriza.

### Problema

```
Sin factoring (parser LL hace muchos backtrack):
  statement = assignment | increment | decrement | ...

  assignment = identifier "=" expr ";"
  increment = identifier "++" ";"
  decrement = identifier "--" ";"

Para "x", no sé si es assignment, increment o decrement.
Necesito leer más (lookahead).
```

### Solución: Factorizar Común

```
Con factoring:
  statement = identifier statement_rest
  statement_rest = "=" expr ";"      ← assignment
                 | "++" ";"           ← increment
                 | "--" ";"           ← decrement

Ahora:
  Veo identifier → avanzo a statement_rest
  Veo "=" → es assignment
  Veo "++" → es increment

Decisión clara y rápida (LL(1)).
```

## Principio 7: Usar EBNF para Evitar Recursión Artificial

EBNF ahorra escritura y mejora claridad.

### Comparación

```
BNF puro (mucha recursión):
  expr_list = expr | expr_list "," expr

EBNF (clara):
  expr_list = expr ("," expr)*

Ambas aceptan el mismo lenguaje: 1, 2, 3
Pero EBNF es más legible y eficiente.
```

### Cuantificadores EBNF en Profundidad

```
{X}    = cero o más = X*
X?     = cero o uno
X+     = uno o más
{X,Y}  = entre X e Y repeticiones
[X]    = opcional = X?

Ejemplo:
  number = "-"? digit+ ("." digit+)?

Deconstruyendo:
  "-"?              → opcional signo negativo
  digit+            → uno o más dígitos
  ("." digit+)?     → opcionalmente: punto y más dígitos

Acepta: 42, -3.14, 0, .5
Rechaza: -, 3., (números incompletos)
```

## Principio 8: Debugging de Gramáticas

Las gramáticas incorrectas son fáciles de escribir, difíciles de debuggear.

### Técnica 1: Pruebas Incrementales

```
Para cada regla que añades:
1. Escribe tests que deberían aceptarse
2. Escribe tests que deberían rechazarse
3. Verifica ambos

Ejemplo para "number":
  Aceptar: 0, 42, -5, 3.14, 0.0, .5
  Rechazar: -, 3., abcd, 3..14, -.-5
```

### Técnica 2: Verbose Parsing

Algunos parseadores (XGrammar) pueden emitir debug:

```
parse(input, verbose=True)

Salida:
  Entering rule: expr
    Entering rule: term
      Entering rule: factor
        Matched: 42
      Exiting rule: factor (success)
    Exiting rule: term (success)
    Matched: +
    Entering rule: term
      ...
  Exiting rule: expr (success)
```

De aquí ves exactamente dónde falla.

### Técnica 3: Análisis de Cobertura

```
Test suite:

✓ expr = "1" (minimalist)
✓ expr = "1+2" (suma)
✓ expr = "1*2" (multiplicación)
✓ expr = "1+2*3" (precedencia)
✓ expr = "(1+2)*3" (paréntesis)
✓ expr = "-5" (unario)
✓ expr = "(1)" (paréntesis single)

Cobertura:
- Cada regla al menos una vez
- Cada alternancia (|) al menos una vez
- Casos límite (vacío, solo un elemento, muchos)
```

## Caso de Estudio: Gramática DSL para Kernel GPU Triton

Diseño progresivo:

### Iteración 1: Variables y Asignación

```
Especificación mínima:

program = statement*
statement = assignment ";"
assignment = identifier "=" expr
expr = identifier | number

Tests:
✓ x = 5;
✓ y = 42;
✓ x = y;
✗ x = 5  (falta ;)
```

### Iteración 2: Expresiones Aritméticas

```
Añadimos:
expr = term ("+" term | "-" term)*
term = factor ("*" factor | "/" factor)*
factor = "(" expr ")" | identifier | number

Tests:
✓ x = 1 + 2;
✓ x = 2 * 3 + 1;
✓ x = (1 + 2) * 3;
```

### Iteración 3: Loops

```
statement = assignment ";" | loop
loop = "for" identifier "in" "range" "(" expr "," expr ")" ":" block
block = "{" statement* "}"

Tests:
✓ for i in range(0, 10): { x = i; }
✓ for i in range(0, 10): { x = i; y = i * 2; }
```

### Iteración 4: Kernel Signature

```
program = kernel_def
kernel_def = "@triton.jit" newline
             "def" identifier "(" params? ")" "->" type ":" newline
             indent block
params = param ("," param)*
param = identifier ":" type
type = "Tensor" | "int32" | "float32"

Tests:
✓ @triton.jit
  def kernel(x: Tensor, n: int32) -> Tensor:
      return x
```

Cada paso: pequeño, testeable, se integra con lo anterior.

## Errores Comunes a Evitar

```
MALO: Gramáticas que generan autómatas enormes
  expr = expr "+" expr | expr "-" expr | expr "*" expr | ...
  Resultado: muchas reglas de reducción, parsing lento

MEJOR: Jerarquía clara de precedencia

MALO: Ambigüedad no resuelta
  if (cond) stmt
  if (cond) stmt else stmt

MEJOR: Usar reglas de desambiguación claras

MALO: Left-recursion en parsers LL(1)
  expr = expr "+" term | term

MEJOR: Transformar a right-recursion o usar EBNF

MALO: Aceptar strings no deseados
  expr = .* (aceptar cualquier cosa)

MEJOR: Especificar exactamente qué aceptar
```

## Implementación Práctica: Diseño Incremental de Gramáticas

Vamos a construir una gramática paso a paso, empezando simple y agregando características:

```{code-cell} ipython3
from dataclasses import dataclass
from typing import List, Optional, Union
import re

# Fase 1: Parser para números simples

@dataclass
class NumberNode:
    value: float

class SimpleNumberParser:
    """Gramática: number = digit+"""

    def __init__(self, text: str):
        self.text = text
        self.position = 0

    def parse(self) -> Optional[NumberNode]:
        """Parsea un número simple"""
        match = re.match(r'\d+', self.text[self.position:])
        if match:
            value = int(match.group(0))
            self.position += len(match.group(0))
            return NumberNode(value)
        return None

# Probar
parser = SimpleNumberParser("123")
result = parser.parse()
print("ITERACIÓN 1: Números simples")
print(f"Entrada: '123' → Resultado: {result}")
```

```{code-cell} ipython3
# Fase 2: Agregamos operaciones de suma

@dataclass
class AddNode:
    left: 'ExprNode'
    right: 'ExprNode'

ExprNode = Union[NumberNode, AddNode]

class SimpleExprParser:
    """Gramática: expr = number ("+" number)*"""

    def __init__(self, text: str):
        self.text = text.replace(' ', '')  # Eliminar espacios
        self.position = 0

    def parse_number(self) -> Optional[NumberNode]:
        match = re.match(r'\d+', self.text[self.position:])
        if match:
            value = int(match.group(0))
            self.position += len(match.group(0))
            return NumberNode(value)
        return None

    def parse(self) -> Optional[ExprNode]:
        """Parsea expresión con sumas"""
        left = self.parse_number()
        if not left:
            return None

        while self.position < len(self.text) and self.text[self.position] == '+':
            self.position += 1  # Consumir '+'
            right = self.parse_number()
            if not right:
                raise SyntaxError("Se esperaba un número después de '+'")
            left = AddNode(left, right)

        return left

# Probar
print("\nITERACIÓN 2: Suma")
test_cases = ["1+2", "1+2+3", "100+200"]
for test in test_cases:
    parser = SimpleExprParser(test)
    result = parser.parse()
    print(f"Entrada: '{test}' → AST: {result}")
```

```{code-cell} ipython3
# Fase 3: Agregamos multiplicación con precedencia correcta

@dataclass
class MulNode:
    left: 'TermNode'
    right: 'TermNode'

TermNode = Union[NumberNode, MulNode]
ExprNode2 = Union[TermNode, AddNode]

class PrecedenceParser:
    """
    Gramática con precedencia:
      expr = term ("+" term)*
      term = number ("*" number)*
    """

    def __init__(self, text: str):
        self.text = text.replace(' ', '')
        self.position = 0

    def parse_number(self) -> Optional[NumberNode]:
        match = re.match(r'\d+', self.text[self.position:])
        if match:
            value = int(match.group(0))
            self.position += len(match.group(0))
            return NumberNode(value)
        return None

    def parse_term(self) -> Optional[TermNode]:
        """Parsea un término (multiplicación)"""
        left = self.parse_number()
        if not left:
            return None

        while self.position < len(self.text) and self.text[self.position] == '*':
            self.position += 1
            right = self.parse_number()
            if not right:
                raise SyntaxError("Se esperaba un número después de '*'")
            left = MulNode(left, right)

        return left

    def parse_expr(self) -> Optional[ExprNode2]:
        """Parsea una expresión (suma)"""
        left = self.parse_term()
        if not left:
            return None

        while self.position < len(self.text) and self.text[self.position] == '+':
            self.position += 1
            right = self.parse_term()
            if not right:
                raise SyntaxError("Se esperaba un término después de '+'")
            left = AddNode(left, right)

        return left

    def parse(self) -> Optional[ExprNode2]:
        return self.parse_expr()

# Probar precedencia
print("\nITERACIÓN 3: Multiplicación con precedencia")
test_cases = ["1+2*3", "2*3+1", "1*2*3+4*5"]
for test in test_cases:
    parser = PrecedenceParser(test)
    result = parser.parse()
    print(f"Entrada: '{test}' → AST: {result}")
```

```{code-cell} ipython3
# Fase 4: Agregamos paréntesis

@dataclass
class ParenNode:
    expr: ExprNode2

FactorNode = Union[NumberNode, ParenNode]

class FullExprParser:
    """
    Gramática completa con paréntesis:
      expr = term ("+" term)*
      term = factor ("*" factor)*
      factor = "(" expr ")" | number
    """

    def __init__(self, text: str):
        self.text = text.replace(' ', '')
        self.position = 0

    def current_char(self) -> Optional[str]:
        if self.position < len(self.text):
            return self.text[self.position]
        return None

    def parse_number(self) -> Optional[NumberNode]:
        match = re.match(r'\d+', self.text[self.position:])
        if match:
            value = int(match.group(0))
            self.position += len(match.group(0))
            return NumberNode(value)
        return None

    def parse_factor(self) -> Optional[FactorNode]:
        """Parsea un factor (número o expresión entre paréntesis)"""
        if self.current_char() == '(':
            self.position += 1  # Consumir '('
            expr = self.parse_expr()
            if self.current_char() != ')':
                raise SyntaxError("Se esperaba ')'")
            self.position += 1  # Consumir ')'
            return ParenNode(expr)
        else:
            return self.parse_number()

    def parse_term(self) -> Optional[TermNode]:
        """Parsea un término (multiplicación)"""
        left = self.parse_factor()
        if not left:
            return None

        while self.current_char() == '*':
            self.position += 1
            right = self.parse_factor()
            if not right:
                raise SyntaxError("Se esperaba un factor después de '*'")
            left = MulNode(left, right)

        return left

    def parse_expr(self) -> Optional[ExprNode2]:
        """Parsea una expresión (suma)"""
        left = self.parse_term()
        if not left:
            return None

        while self.current_char() == '+':
            self.position += 1
            right = self.parse_term()
            if not right:
                raise SyntaxError("Se esperaba un término después de '+'")
            left = AddNode(left, right)

        return left

    def parse(self) -> Optional[ExprNode2]:
        result = self.parse_expr()
        if self.position < len(self.text):
            raise SyntaxError(f"Caracteres inesperados al final: '{self.text[self.position:]}'")
        return result

# Probar con paréntesis
print("\nITERACIÓN 4: Paréntesis")
test_cases = ["(1+2)*3", "1*(2+3)", "((1+2)*3)+4", "(1+2)*(3+4)"]
for test in test_cases:
    parser = FullExprParser(test)
    result = parser.parse()
    eval_result = eval(test)
    print(f"Entrada: '{test}' → Valor: {eval_result}")
```

## Análisis de Ambigüedad

```{code-cell} ipython3
# Ejemplo de gramática ambigua: if-then-else

class AmbiguousGrammar:
    """
    Gramática ambigua para if-then-else:
      stmt = "if" expr "then" stmt
           | "if" expr "then" stmt "else" stmt
           | "simple"

    Problema: ¿A cuál 'if' pertenece el 'else'?
    """

    @staticmethod
    def parse_ambiguous(code: str) -> List[str]:
        """Muestra las posibles interpretaciones"""
        if code == "if a then if b then s1 else s2":
            return [
                "Interpretación 1: if a then (if b then s1 else s2)",
                "Interpretación 2: if a then (if b then s1) else s2"
            ]
        return ["Caso no ambiguo"]

# Mostrar ambigüedad
print("\nANÁLISIS DE AMBIGÜEDAD: Problema del if-else colgante")
print("="*60)
code = "if a then if b then s1 else s2"
print(f"Código: {code}\n")
interpretations = AmbiguousGrammar.parse_ambiguous(code)
for i, interp in enumerate(interpretations, 1):
    print(f"{i}. {interp}")

print("\n" + "="*60)
print("SOLUCIÓN: Gramática no ambigua")
print("="*60)
print("""
stmt = matched_stmt | unmatched_stmt

matched_stmt = "if" expr "then" matched_stmt "else" matched_stmt
             | "simple"

unmatched_stmt = "if" expr "then" stmt
               | "if" expr "then" matched_stmt "else" unmatched_stmt

Regla: El 'else' siempre se asocia con el 'if' más cercano sin else.
""")
```

## Left-Factoring: Antes y Después

```{code-cell} ipython3
# Demostración de left-factoring

class BeforeFactoring:
    """
    Gramática sin factorizar:
      statement = "print" expr ";"
                | "println" expr ";"

    Problema: Backtracking necesario
    """

    def parse(self, tokens: List[str]) -> str:
        """Simula parsing con backtracking"""
        steps = []

        if tokens[0] in ['print', 'println']:
            steps.append(f"1. Veo '{tokens[0]}' - ¿es print o println?")
            steps.append(f"2. Intento primera alternativa: 'print'")
            if tokens[0] == 'print':
                steps.append("3. ✓ Coincide con primera alternativa")
            else:
                steps.append("3. ✗ No coincide, backtrack")
                steps.append("4. Intento segunda alternativa: 'println'")
                steps.append("5. ✓ Coincide con segunda alternativa")

        return "\n".join(steps)

class AfterFactoring:
    """
    Gramática factorizada:
      statement = "print" statement_rest
      statement_rest = "ln" expr ";" | expr ";"

    Ventaja: Sin backtracking
    """

    def parse(self, tokens: List[str]) -> str:
        """Simula parsing sin backtracking"""
        steps = []

        if tokens[0].startswith('print'):
            steps.append(f"1. Veo 'print' - consumo prefijo común")
            rest = tokens[0][5:]  # Después de 'print'
            if rest == 'ln':
                steps.append("2. Veo 'ln' → es 'println'")
                steps.append("3. ✓ Decisión directa, sin backtracking")
            else:
                steps.append("2. No veo 'ln' → es 'print'")
                steps.append("3. ✓ Decisión directa, sin backtracking")

        return "\n".join(steps)

print("\nLEFT-FACTORING: Comparación")
print("="*60)

print("\nSIN FACTORIZAR:")
print("-"*60)
parser1 = BeforeFactoring()
print(parser1.parse(['println', 'x', ';']))

print("\n\nCON FACTORIZAR:")
print("-"*60)
parser2 = AfterFactoring()
print(parser2.parse(['println', 'x', ';']))

print("\n" + "="*60)
print("Conclusión: Factorizar reduce backtracking y mejora performance")
```

## Diseño Modular en Capas

```{code-cell} ipython3
# Ejemplo de diseño en capas para un mini-lenguaje

class LayeredGrammar:
    """
    Gramática en capas para mini-lenguaje:

    L3 (Alto nivel):
      program = statement*
      statement = assignment | loop

    L2 (Nivel medio):
      assignment = identifier "=" expr ";"
      loop = "for" identifier "in" range ":" block
      block = "{" statement* "}"

    L1 (Bajo nivel):
      expr = term ("+" term)*
      term = factor ("*" factor)*
      factor = identifier | number | "(" expr ")"

    L0 (Léxico):
      identifier = letter (letter | digit)*
      number = digit+
      letter = "a".."z" | "A".."Z"
      digit = "0".."9"
    """

    def __init__(self):
        self.layers = {
            'L0': ['letter', 'digit'],
            'L1': ['identifier', 'number', 'expr', 'term', 'factor'],
            'L2': ['assignment', 'loop', 'block'],
            'L3': ['program', 'statement']
        }

    def show_layers(self):
        """Muestra la estructura en capas"""
        print("DISEÑO EN CAPAS")
        print("="*60)
        for layer in ['L3', 'L2', 'L1', 'L0']:
            print(f"\n{layer}:")
            print(f"  Reglas: {', '.join(self.layers[layer])}")

            if layer == 'L3':
                print("  Responsabilidad: Estructura del programa")
            elif layer == 'L2':
                print("  Responsabilidad: Construcciones sintácticas")
            elif layer == 'L1':
                print("  Responsabilidad: Expresiones y tokens")
            elif layer == 'L0':
                print("  Responsabilidad: Caracteres básicos")

grammar = LayeredGrammar()
grammar.show_layers()

print("\n" + "="*60)
print("VENTAJAS DEL DISEÑO EN CAPAS:")
print("  ✓ Independencia: Cambiar L0 no afecta L2")
print("  ✓ Reuso: Reglas de L1 usadas en múltiples L2")
print("  ✓ Claridad: Cada capa tiene responsabilidad clara")
print("  ✓ Testing: Probar cada capa aisladamente")
```

## Test Suite para Gramáticas

```{code-cell} ipython3
# Framework de testing para gramáticas

class GrammarTester:
    """Framework para probar gramáticas sistemáticamente"""

    def __init__(self, parser_class):
        self.parser_class = parser_class
        self.passed = 0
        self.failed = 0

    def test_accept(self, input_str: str, description: str = ""):
        """Verifica que la entrada sea aceptada"""
        try:
            parser = self.parser_class(input_str)
            result = parser.parse()
            if result is not None:
                print(f"✓ PASS: '{input_str}' {description}")
                self.passed += 1
            else:
                print(f"✗ FAIL: '{input_str}' debería ser aceptado {description}")
                self.failed += 1
        except Exception as e:
            print(f"✗ FAIL: '{input_str}' lanzó excepción: {e}")
            self.failed += 1

    def test_reject(self, input_str: str, description: str = ""):
        """Verifica que la entrada sea rechazada"""
        try:
            parser = self.parser_class(input_str)
            result = parser.parse()
            print(f"✗ FAIL: '{input_str}' debería ser rechazado {description}")
            self.failed += 1
        except:
            print(f"✓ PASS: '{input_str}' correctamente rechazado {description}")
            self.passed += 1

    def report(self):
        """Muestra reporte de pruebas"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"REPORTE: {self.passed}/{total} pruebas pasaron")
        if self.failed == 0:
            print("✓ Todos los tests pasaron!")
        else:
            print(f"✗ {self.failed} tests fallaron")
        print(f"{'='*60}")

# Probar la gramática completa
print("SUITE DE PRUEBAS PARA GRAMÁTICA DE EXPRESIONES")
print("="*60)

tester = GrammarTester(FullExprParser)

print("\nPruebas de aceptación:")
tester.test_accept("1", "(número simple)")
tester.test_accept("1+2", "(suma simple)")
tester.test_accept("1*2", "(multiplicación)")
tester.test_accept("1+2*3", "(precedencia)")
tester.test_accept("(1+2)*3", "(paréntesis)")
tester.test_accept("((1+2)*3)+4", "(paréntesis anidados)")

print("\nPruebas de rechazo:")
tester.test_reject("", "(entrada vacía)")
tester.test_reject("1+", "(operador sin operando)")
tester.test_reject("(1+2", "(paréntesis sin cerrar)")
tester.test_reject("1))", "(paréntesis extra)")
tester.test_reject("1 2", "(números sin operador)")

tester.report()
```

```{admonition} 🎯 Conexión con el Proyecto
:class: important
Al diseñar gramáticas DSL para XGrammar, estos principios son cruciales: el diseño incremental permite validar cada característica aisladamente, la separación de precedencia evita ambigüedad, y el diseño modular facilita extender la gramática cuando necesites añadir nuevas construcciones sintácticas a tu lenguaje de kernels GPU.
```

```{admonition} Resumen
:class: important
**Conceptos clave:**
- Diseña gramáticas incrementalmente: empieza simple y añade características una por una
- Codifica precedencia mediante niveles jerárquicos en la estructura de la gramática
- Evita ambigüedad mediante refactorización o usando precedencia explícita
- Aplica left-factoring para mejorar la eficiencia de parsers LL
- Organiza gramáticas en capas (léxico → sintaxis básica → construcciones complejas)

**Para la siguiente lectura:**
Exploraremos técnicas avanzadas de compilación de gramáticas, incluyendo normalización profunda, construcción de tablas de parsing y optimizaciones específicas.
```

## Ejercicios

1. **Diseño Incremental**: Especifica una gramática para direcciones de email:
   - Iteración 1: user@domain
   - Iteración 2: user.name@domain
   - Iteración 3: user+tag@domain.co.uk

2. **Precedencia**: Reescribe para expresar precedencia clara:
   ```
   expr = expr "||" expr | expr "&&" expr | expr "==" expr | ...
   ```

3. **Ambigüedad**: ¿Cuál es el problema aquí?
   ```
   stmt = "if" expr ":" stmt
        | "if" expr ":" stmt "else" stmt
   ```

4. **Left-Factoring**: Factoriza:
   ```
   stmt = "print" expr ";" | "println" expr ";"
   ```

5. **Debugging**: Diseña test cases para esta gramática:
   ```
   list = "[" [item ("," item)*] "]"
   item = number | string
   number = digit+
   string = '"' letter* '"'
   ```

## Preguntas de Reflexión

- ¿Cuál es el verdadero costo de la ambigüedad? ¿Solo teoría o afecta en la práctica?
- ¿Cómo se diseñaría una gramática para lenguaje natural (inglés) vs lenguaje de programación?
- En el contexto de GPU kernels, ¿hay características inherentemente ambiguas que no se pueden resolver solo con gramática?
- ¿Cómo el diseño de la gramática afecta la capacidad de dar mensajes de error útiles?

---

## Referencias

- Fowler, M. (2010). [Domain-Specific Languages](https://martinfowler.com/books/dsl.html). Addison-Wesley.
- Parr, T. (2013). [The Definitive ANTLR 4 Reference](https://pragprog.com/titles/tpantlr2/the-definitive-antlr-4-reference/). Pragmatic Bookshelf.
- Aho, A., Lam, M., Sethi, R., & Ullman, J. (2006). Compilers: Principles, Techniques, and Tools (2nd ed.). Pearson.
