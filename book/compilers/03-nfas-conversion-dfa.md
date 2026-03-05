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

# NFAs y Conversión a DFA: Determinismo a través de la Construcción de Subconjuntos

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/03-nfas-conversion-dfa.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn triton xgrammar
print('Dependencies installed!')
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Distinguir entre NFAs y DFAs y explicar sus ventajas/desventajas
- Calcular la epsilon-clausura de un conjunto de estados en un NFA
- Aplicar el algoritmo de construcción de subconjuntos para convertir NFA a DFA
- Explicar por qué la minimización de DFA es importante para la eficiencia
- Relacionar la conversión NFA→DFA con el pipeline de compilación de XGrammar
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[NFA to DFA Conversion (Neso Academy)](https://www.youtube.com/watch?v=--TqwBvW9v8)** - Conceptos extremadamente visuales sobre las máquinas de estados no deterministas y su conversión a DFA.
```

```{admonition} 🔧 Herramienta Interactiva
:class: seealso

**[AutomataVerse](https://automataverse.com/)** - Visualiza la conversión de NFA a DFA paso a paso, mostrando cómo se construyen los subconjuntos de estados.
```

## El Problema del Determinismo

En la lectura anterior, trabajamos con **DFAs deterministas**: desde cada estado, para cada carácter, hay exactamente una transición. Esto es restrictivo cuando diseñamos autómatas manualmente.

Imagina que quieres reconocer dos palabras: "gato" y "gatos". Con un DFA, necesitarías:

```
q₀ --g--> q₁ --a--> q₂ --t--> q₃ --o--> q₄ --s?--> q₅
```

Pero aquí hay un problema: desde q₄, cuando leemos 's', ¿es el final (aceptamos) o no? En un DFA, debe haber exactamente una opción.

Sería mucho más fácil si pudiéramos decir: "desde q₄, el string termina aquí (aceptado), **y** también si vemos 's', seguimos a q₅ (también aceptado)".

Esto es posible con **NFAs** (Nondeterministic Finite Automata).

## Autómatas Finitos No-Deterministas (NFA)

Un **NFA** es como un DFA, pero con superpoderes:

1. **Múltiples transiciones**: Desde un estado, el mismo carácter puede llevar a varios estados
2. **Transiciones épsilon (ε)**: Cambios de estado sin consumir carácter

### Definición Formal

Un NFA es una tupla: **M = (Q, Σ, δ, q₀, F)**

- **Q**: Conjunto de estados
- **Σ**: Alfabeto
- **δ: Q × (Σ ∪ {ε}) → 2^Q** (nota: potencia - múltiples estados posibles)
- **q₀**: Estado inicial
- **F**: Estados de aceptación

### Ejemplo 1: NFA para "gato" | "gatos"

```
          a    t    o
     ┌─g─→q₁─→q₂─→q₃─→q₄(accept)
     │                     ↑
q₀───┤                      └─s─┘
     │
     └─ (otras opciones posibles)

Estados de aceptación: {q₄, q₅}

Especialmente: δ(q₄, 's') = {q₅}
              δ(q₄, ε) = {} (implícitamente, q₄ es aceptación)

O más directo:
q₀ --g--> q₁ --a--> q₂ --t--> q₃ --o--> q₄ --ε--> q₅
                                 └──s──┘

Aquí q₄ y q₅ son aceptación.
```

Para reconocer "gatos":
```
q₀ -g-> q₁ -a-> q₂ -t-> q₃ -o-> q₄ -s-> q₅ (aceptado)
```

Para reconocer "gato":
```
q₀ -g-> q₁ -a-> q₂ -t-> q₃ -o-> q₄ (aceptado)
```

### Ejemplo 2: Transiciones Épsilon

Las transiciones épsilon permiten "saltos gratis". Considera un NFA que acepta strings que empiezan con 'a' o terminan con 'b':

```
       ┌─ a ─┐
       │      v
q₀─ε──┤     q₁ ─ε─→ q₃ (accept)
       │              ↑
       └─ b ─┐      (cualquier)
              v
             q₂ ─────→ q₃

Dicho de otro modo:
- Desde q₀, puedo saltar (ε) a q₁ o a q₂
- q₁ espera 'a' y salta a q₃
- q₂ espera cualquier carácter y salta a q₃
```

Entrada "a":
```
q₀ -ε-> q₁ -a-> q₃ (aceptado)
```

Entrada "xa" (donde x es cualquier carácter):
```
q₀ -ε-> q₂ -x-> q₃ -a-> ... (¡espera, necesitaría otro estado!)
```

Este NFA no es correcto. Mejor:

```
q₀ -ε-> q₁ -ε-> q₂
q₁ -a-> q₃ (accept)
q₂ -cualquier- -> q₂
q₂ -b-> q₃ (accept)
```

### Ventajas de NFAs

```
NFA = "Optimista"
DFA = "Pesimista"

En un NFA, cuando hay ambigüedad, exploramos TODOS los caminos posibles.
En un DFA, debemos decidir exactamente.
```

NFAs son:
- **Más fáciles de diseñar**: Puedes escribir múltiples transiciones sin pensar
- **Más compactos**: Número de estados generalmente menor
- **Teóricos**: Base para algoritmos de compilación

## Simulación de un NFA

Reconocer un string en un NFA es más complejo que en un DFA:

```
Algoritmo: Simular NFA
1. estados_actuales = ε-clausura(q₀)  // todos los estados alcanzables desde q₀ con ε
2. Para cada carácter en entrada:
      nuevos_estados = {}
      Para cada estado en estados_actuales:
          Para cada estado_siguiente en δ(estado, carácter):
              nuevos_estados = nuevos_estados ∪ ε-clausura(estado_siguiente)
      estados_actuales = nuevos_estados
      Si estados_actuales está vacío:
          RECHAZAR
3. Si algún estado en estados_actuales está en F:
      ACEPTAR
   Si no:
      RECHAZAR
```

La clave es **ε-clausura**: el conjunto de todos los estados alcanzables usando solo transiciones ε.

### Ejemplo: ε-clausura

```
Estados de un NFA:
q₀ -ε-> q₁
q₀ -ε-> q₂
q₁ -ε-> q₃

ε-clausura(q₀) = {q₀, q₁, q₂, q₃}  (todos alcanzables sin consumir entrada)
ε-clausura(q₁) = {q₁, q₃}
ε-clausura(q₂) = {q₂}
ε-clausura(q₃) = {q₃}
```

## Conversión de NFA a DFA: Construcción de Subconjuntos

Aquí viene la magia: **Cualquier NFA puede convertirse a un DFA equivalente**.

La técnica se llama **"Subset Construction"** o "construcción de potencia".

La idea: cada estado del DFA representa un **conjunto de estados del NFA**.

### Algoritmo de Conversión

```
Entrada: NFA M = (Q, Σ, δ, q₀, F)
Salida: DFA M' = (Q', Σ, δ', q₀', F')

q₀' = ε-clausura(q₀)
Q' = {q₀'}  (empezar con un estado)
cola = [q₀']

Mientras cola no esté vacía:
    S = cola.pop()  // S es un subconjunto de estados del NFA
    Q' = Q' ∪ {S}

    Para cada carácter a en Σ:
        S' = {}
        Para cada estado q en S:
            S' = S' ∪ δ(q, a)
        S' = ε-clausura(S')  // aplicar ε-clausura

        δ'(S, a) = S'

        Si S' no está en Q':
            cola = cola + [S']

F' = {S ⊆ Q | S ∩ F ≠ ∅}  // estados que contienen estados de aceptación del NFA
```

### Ejemplo Paso a Paso

**NFA Original**:
```
Queremos reconocer a*b (cero o más 'a', seguida de 'b')

q₀ --a--> q₁
q₀ --ε--> q₂
q₁ --a--> q₁
q₁ --ε--> q₂
q₂ --b--> q₃ (accept)

Estados de aceptación: {q₃}
```

**Conversión**:

Paso 1: q₀' = ε-clausura(q₀) = {q₀, q₂}

```
  δ_NFA(q₀, a) = {q₁}    → ε-clausura = {q₁, q₂}
  δ_NFA(q₂, a) = {}
  δ'({q₀, q₂}, a) = {q₁, q₂}

  δ_NFA(q₀, b) = {}
  δ_NFA(q₂, b) = {q₃}
  δ'({q₀, q₂}, b) = {q₃}
```

Paso 2: Procesar {q₁, q₂}
```
  δ_NFA(q₁, a) = {q₁}    → ε-clausura = {q₁, q₂}
  δ_NFA(q₂, a) = {}
  δ'({q₁, q₂}, a) = {q₁, q₂}

  δ_NFA(q₁, b) = {}
  δ_NFA(q₂, b) = {q₃}
  δ'({q₁, q₂}, b) = {q₃}
```

Paso 3: Procesar {q₃}
```
  δ_NFA(q₃, a) = {}
  δ'({q₃}, a) = {}

  δ_NFA(q₃, b) = {}
  δ'({q₃}, b) = {}
```

**DFA Resultante**:
```
Estados:
  Q' = {{q₀,q₂}, {q₁,q₂}, {q₃}, {}}

Transiciones (nota: renombramos para claridad):
  S₀ = {q₀, q₂}  (inicial)
  S₁ = {q₁, q₂}
  S₂ = {q₃}      (aceptación)
  S₃ = {}        (dead state)

Tabla:
        a    b
S₀ → S₁  S₂
S₁ → S₁  S₂
S₂ → S₃  S₃
S₃ → S₃  S₃
```

Este DFA acepta exactamente: a*b

Entradas:
- `"b"` → S₀ -b-> S₂ (aceptado)
- `"ab"` → S₀ -a-> S₁ -b-> S₂ (aceptado)
- `"aab"` → S₀ -a-> S₁ -a-> S₁ -b-> S₂ (aceptado)
- `"aa"` → S₀ -a-> S₁ -a-> S₁ (rechazado, en S₁)

:::{figure} diagrams/nfa_to_dfa_conversion.png
:name: fig-nfa-dfa-conversion
:alt: Proceso de conversión de NFA a DFA mediante construcción de subconjuntos
:align: center
:width: 90%

**Figura 1:** Construcción de subconjuntos: cómo los conjuntos de estados del NFA se convierten en estados individuales del DFA. Se muestran las transiciones ε-clausura en cada paso.
:::

## Minimización de DFA: Algoritmo de Hopcroft

Una vez que tenemos un DFA (conversión de NFA), podemos optimizarlo **eliminando estados redundantes**.

Dos estados son **equivalentes** si aceptan exactamente los mismos sufijos.

### Algoritmo Simplificado

```
Entrada: DFA M = (Q, Σ, δ, q₀, F)
Salida: DFA minimizado

Paso 1: Particionar Q en dos grupos:
  - F (aceptación)
  - Q - F (rechazo)

Paso 2: Mientras la partición cambie:
  Para cada grupo G en la partición:
    Para cada símbolo a en Σ:
      Si los estados en G transicionan a diferentes grupos con a:
        Dividir G en subgrupos según a-transición

Paso 3: Crear nuevo DFA con grupos como estados
```

### Ejemplo Minificación

Tomemos nuestro DFA anterior:

```
Estados originales: S₀, S₁, S₂, S₃

Partición inicial:
  Aceptación: {S₂}
  Rechazo: {S₀, S₁, S₃}

Análisis:
  Grupo {S₀, S₁, S₃}:
    - Transición 'a': S₀→S₁, S₁→S₁, S₃→S₃
      Diferentes grupos: S₁ (en el grupo), S₃ (en el grupo)
      Pero tanto S₀ como S₁ van a S₁ (mismo grupo), mientras S₃ va a S₃
      Dividir: {S₀, S₁} vs {S₃}

Partición después de división:
  {S₂}, {S₀, S₁}, {S₃}

Continuar análisis hasta estabilidad...
```

En este caso, muchos estados finales son necesarios y no se pueden eliminar.

## Relación NFA-DFA en la Práctica

```
HERRAMIENTA              ENTRADA    PROCESO                      SALIDA
────────────────────────────────────────────────────────────
Expresión Regular    →  NFA (auto)  →  Conversión subset constr.  →  DFA eficiente
                                        Minimización Hopcroft

flex (lexer gen)         regex      →  NFA                    →  DFA
                                        Subset construction
                                        Optimizaciones

XGrammar                 CFG        →  (diferente proceso, entrada Earley)
```

## Por qué Importa en XGrammar

1. **Tokenización**: El léxer usa DFAs eficientes generados de regexes (vía NFA)
2. **Compilación**: Entender NFA/DFA ayuda a entender transformaciones de gramáticas
3. **Optimización**: Técnicas similares (epsilon-elimination, minimización) se usan en la pipeline de XGrammar

## Implementación de NFA en Python

Vamos a implementar un NFA completo con soporte para transiciones épsilon:

```{code-cell} ipython3
from typing import Set, Dict, List, Tuple
from collections import defaultdict, deque

class NFA:
    """Autómata Finito No-Determinista con transiciones épsilon"""

    def __init__(self):
        self.states: Set[str] = set()
        self.alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.start_state: str = None
        self.accept_states: Set[str] = set()

    def add_state(self, state: str, is_accept: bool = False):
        """Añade un estado al NFA"""
        self.states.add(state)
        if is_accept:
            self.accept_states.add(state)

    def add_transition(self, from_state: str, symbol: str, to_state: str):
        """Añade una transición (símbolo puede ser 'ε' para épsilon)"""
        self.transitions[(from_state, symbol)].add(to_state)
        if symbol != 'ε':
            self.alphabet.add(symbol)

    def set_start_state(self, state: str):
        """Establece el estado inicial"""
        self.start_state = state
        self.states.add(state)

    def epsilon_closure(self, states: Set[str]) -> Set[str]:
        """Calcula la ε-clausura de un conjunto de estados"""
        closure = set(states)
        stack = list(states)

        while stack:
            current = stack.pop()
            # Buscar transiciones épsilon desde el estado actual
            epsilon_transitions = self.transitions.get((current, 'ε'), set())
            for next_state in epsilon_transitions:
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)

        return closure

    def accepts(self, input_string: str) -> bool:
        """Verifica si el NFA acepta el string de entrada"""
        # Empezar con la ε-clausura del estado inicial
        current_states = self.epsilon_closure({self.start_state})

        # Procesar cada carácter
        for symbol in input_string:
            next_states = set()

            # Para cada estado actual, encontrar transiciones con el símbolo
            for state in current_states:
                transitions = self.transitions.get((state, symbol), set())
                next_states.update(transitions)

            # Calcular ε-clausura del nuevo conjunto de estados
            current_states = self.epsilon_closure(next_states)

            # Si no hay estados, rechazar
            if not current_states:
                return False

        # Aceptar si algún estado actual es de aceptación
        return bool(current_states & self.accept_states)

    def __str__(self):
        result = "NFA:\n"
        result += f"  Estados: {self.states}\n"
        result += f"  Alfabeto: {self.alphabet}\n"
        result += f"  Inicial: {self.start_state}\n"
        result += f"  Aceptación: {self.accept_states}\n"
        result += "  Transiciones:\n"
        for (from_state, symbol), to_states in sorted(self.transitions.items()):
            for to_state in to_states:
                result += f"    {from_state} --{symbol}--> {to_state}\n"
        return result


# Ejemplo: NFA que reconoce "gato" | "gatos"
nfa = NFA()
nfa.set_start_state('q0')
nfa.add_state('q1')
nfa.add_state('q2')
nfa.add_state('q3')
nfa.add_state('q4', is_accept=True)
nfa.add_state('q5', is_accept=True)

nfa.add_transition('q0', 'g', 'q1')
nfa.add_transition('q1', 'a', 'q2')
nfa.add_transition('q2', 't', 'q3')
nfa.add_transition('q3', 'o', 'q4')
nfa.add_transition('q4', 's', 'q5')

print(nfa)

# Probar el NFA
test_words = ['gato', 'gatos', 'gat', 'gatoss', 'cat']
print("\nPruebas:")
for word in test_words:
    result = nfa.accepts(word)
    print(f"  '{word}' -> {'✓ aceptado' if result else '✗ rechazado'}")
```

```{code-cell} ipython3
# Ejemplo: NFA con transiciones épsilon para (a|b)*c

nfa_epsilon = NFA()
nfa_epsilon.set_start_state('q0')
nfa_epsilon.add_state('q1')
nfa_epsilon.add_state('q2')
nfa_epsilon.add_state('q3')
nfa_epsilon.add_state('q4', is_accept=True)

# Opción de leer 'a' o 'b' múltiples veces
nfa_epsilon.add_transition('q0', 'ε', 'q1')
nfa_epsilon.add_transition('q1', 'a', 'q2')
nfa_epsilon.add_transition('q1', 'b', 'q2')
nfa_epsilon.add_transition('q2', 'ε', 'q1')  # Loop para *

# Transición a 'c' final
nfa_epsilon.add_transition('q1', 'ε', 'q3')
nfa_epsilon.add_transition('q3', 'c', 'q4')

print(nfa_epsilon)

# Calcular ε-clausura del estado inicial
initial_closure = nfa_epsilon.epsilon_closure({'q0'})
print(f"\nε-clausura(q0) = {initial_closure}")

# Probar el NFA
test_words = ['c', 'ac', 'bc', 'abc', 'aabc', 'abbc', 'ababc', 'ab', 'a']
print("\nPruebas para (a|b)*c:")
for word in test_words:
    result = nfa_epsilon.accepts(word)
    print(f"  '{word}' -> {'✓ aceptado' if result else '✗ rechazado'}")
```

## Conversión de NFA a DFA: Subset Construction

```{code-cell} ipython3
class DFA:
    """Autómata Finito Determinista"""

    def __init__(self):
        self.states: Set[frozenset] = set()
        self.alphabet: Set[str] = set()
        self.transitions: Dict[Tuple[frozenset, str], frozenset] = {}
        self.start_state: frozenset = None
        self.accept_states: Set[frozenset] = set()

    def accepts(self, input_string: str) -> bool:
        """Verifica si el DFA acepta el string de entrada"""
        current_state = self.start_state

        for symbol in input_string:
            if (current_state, symbol) not in self.transitions:
                return False
            current_state = self.transitions[(current_state, symbol)]

        return current_state in self.accept_states

    def __str__(self):
        # Crear nombres cortos para los estados
        state_names = {}
        for i, state in enumerate(sorted(self.states, key=lambda x: len(x))):
            state_names[state] = f"S{i}"

        result = "DFA:\n"
        result += f"  Estados: {len(self.states)}\n"
        result += f"  Inicial: {state_names.get(self.start_state, 'N/A')}\n"
        result += f"  Aceptación: {[state_names[s] for s in self.accept_states]}\n"
        result += "  Transiciones:\n"

        for (from_state, symbol), to_state in sorted(self.transitions.items()):
            result += f"    {state_names[from_state]} --{symbol}--> {state_names[to_state]}\n"

        return result


def nfa_to_dfa(nfa: NFA) -> DFA:
    """Convierte un NFA a DFA usando construcción de subconjuntos"""
    dfa = DFA()
    dfa.alphabet = nfa.alphabet.copy()

    # Estado inicial del DFA es la ε-clausura del estado inicial del NFA
    initial_closure = frozenset(nfa.epsilon_closure({nfa.start_state}))
    dfa.start_state = initial_closure
    dfa.states.add(initial_closure)

    # Si contiene un estado de aceptación del NFA, es estado de aceptación del DFA
    if initial_closure & nfa.accept_states:
        dfa.accept_states.add(initial_closure)

    # Cola de estados por procesar
    unprocessed = deque([initial_closure])
    processed = set()

    while unprocessed:
        current_state_set = unprocessed.popleft()

        if current_state_set in processed:
            continue
        processed.add(current_state_set)

        # Para cada símbolo del alfabeto
        for symbol in nfa.alphabet:
            # Encontrar todos los estados alcanzables
            next_states = set()
            for state in current_state_set:
                transitions = nfa.transitions.get((state, symbol), set())
                next_states.update(transitions)

            # Calcular ε-clausura
            next_state_set = frozenset(nfa.epsilon_closure(next_states))

            if next_state_set:  # Si hay estados alcanzables
                # Añadir transición al DFA
                dfa.transitions[(current_state_set, symbol)] = next_state_set
                dfa.states.add(next_state_set)

                # Si es estado de aceptación
                if next_state_set & nfa.accept_states:
                    dfa.accept_states.add(next_state_set)

                # Añadir a la cola si no ha sido procesado
                if next_state_set not in processed:
                    unprocessed.append(next_state_set)

    return dfa


# Convertir el NFA anterior a DFA
print("="*60)
print("CONVERSIÓN NFA → DFA")
print("="*60)
print("\nNFA original:")
print(nfa_epsilon)

dfa = nfa_to_dfa(nfa_epsilon)
print("\nDFA resultante:")
print(dfa)

# Verificar que ambos aceptan los mismos strings
print("\nVerificación de equivalencia:")
test_words = ['c', 'ac', 'bc', 'abc', 'aabc', 'abbc', 'ababc', 'ab', 'a']
all_match = True

for word in test_words:
    nfa_result = nfa_epsilon.accepts(word)
    dfa_result = dfa.accepts(word)
    match = "✓" if nfa_result == dfa_result else "✗"
    print(f"  '{word}': NFA={nfa_result}, DFA={dfa_result} {match}")
    if nfa_result != dfa_result:
        all_match = False

print(f"\n{'✓ Todos los resultados coinciden' if all_match else '✗ Hay discrepancias'}")
```

```{code-cell} ipython3
# Ejemplo más complejo: NFA para a*b (cero o más 'a', seguida de 'b')

nfa_ab = NFA()
nfa_ab.set_start_state('q0')
nfa_ab.add_state('q1')
nfa_ab.add_state('q2')
nfa_ab.add_state('q3', is_accept=True)

# Transiciones
nfa_ab.add_transition('q0', 'a', 'q1')
nfa_ab.add_transition('q0', 'ε', 'q2')
nfa_ab.add_transition('q1', 'a', 'q1')
nfa_ab.add_transition('q1', 'ε', 'q2')
nfa_ab.add_transition('q2', 'b', 'q3')

print("="*60)
print("EJEMPLO: a*b")
print("="*60)
print("\nNFA para a*b:")
print(nfa_ab)

# Convertir a DFA
dfa_ab = nfa_to_dfa(nfa_ab)
print("\nDFA para a*b:")
print(dfa_ab)

# Probar
test_words = ['b', 'ab', 'aab', 'aaab', 'a', 'aa', 'bb', 'aba']
print("\nPruebas:")
for word in test_words:
    nfa_result = nfa_ab.accepts(word)
    dfa_result = dfa_ab.accepts(word)
    match = "✓" if nfa_result == dfa_result else "✗"
    print(f"  '{word}': NFA={nfa_result}, DFA={dfa_result} {match}")
```

## Visualización de la Construcción de Subconjuntos

```{code-cell} ipython3
# Visualización del algoritmo de subset construction
import plotly.graph_objects as go

# Estados del proceso NFA→DFA
pasos = [
    {"estado": "{q0}", "entrada": "ε-closure", "nuevo": "{q0}"},
    {"estado": "{q0}", "entrada": "a", "nuevo": "{q0, q1}"},
    {"estado": "{q0}", "entrada": "b", "nuevo": "{q0}"},
    {"estado": "{q0, q1}", "entrada": "a", "nuevo": "{q0, q1}"},
    {"estado": "{q0, q1}", "entrada": "b", "nuevo": "{q0, q2}"},
]

fig = go.Figure(data=[go.Table(
    header=dict(values=['Estado DFA', 'Símbolo', 'Nuevo Estado'],
                fill_color='steelblue', font=dict(color='white', size=14)),
    cells=dict(values=[
        [p["estado"] for p in pasos],
        [p["entrada"] for p in pasos],
        [p["nuevo"] for p in pasos]
    ], fill_color='lavender', font=dict(size=12))
)])

fig.update_layout(title="Tabla de Transición: NFA → DFA (Subset Construction)", height=300)
fig.show()
```

```{admonition} 🎮 Conversor NFA → DFA Online
:class: tip

Prueba la conversión automática:

<iframe src="https://www.automataverse.com/" width="100%" height="500px" style="border:1px solid #ddd; border-radius:8px;"></iframe>
```

```{code-cell} ipython3
def visualize_subset_construction(nfa: NFA):
    """Muestra paso a paso la construcción de subconjuntos"""
    print("="*60)
    print("CONSTRUCCIÓN DE SUBCONJUNTOS - PASO A PASO")
    print("="*60)

    # Estado inicial
    initial_closure = frozenset(nfa.epsilon_closure({nfa.start_state}))
    print(f"\n1. Estado inicial del DFA:")
    print(f"   ε-clausura({nfa.start_state}) = {set(initial_closure)}")

    # Procesar estados
    unprocessed = deque([initial_closure])
    processed = set()
    state_number = 0
    state_names = {initial_closure: f"S{state_number}"}
    state_number += 1

    all_transitions = {}

    while unprocessed:
        current_state_set = unprocessed.popleft()

        if current_state_set in processed:
            continue
        processed.add(current_state_set)

        print(f"\n{len(processed)+1}. Procesando {state_names[current_state_set]} = {set(current_state_set)}:")

        for symbol in sorted(nfa.alphabet):
            next_states = set()
            for state in current_state_set:
                transitions = nfa.transitions.get((state, symbol), set())
                next_states.update(transitions)

            next_state_set = frozenset(nfa.epsilon_closure(next_states))

            if next_state_set:
                if next_state_set not in state_names:
                    state_names[next_state_set] = f"S{state_number}"
                    state_number += 1

                all_transitions[(current_state_set, symbol)] = next_state_set
                print(f"   Con '{symbol}': {set(next_states)} → ε-clausura = {set(next_state_set)} ({state_names[next_state_set]})")

                if next_state_set not in processed:
                    unprocessed.append(next_state_set)

    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DEL DFA CONSTRUIDO")
    print(f"{'='*60}")
    print(f"\nEstados del DFA: {len(state_names)}")
    for state_set, name in sorted(state_names.items(), key=lambda x: x[1]):
        is_accept = bool(state_set & nfa.accept_states)
        accept_mark = "(ACEPTACIÓN)" if is_accept else ""
        is_initial = "(INICIAL)" if state_set == initial_closure else ""
        print(f"  {name} = {set(state_set)} {accept_mark}{is_initial}")

    print("\nTransiciones:")
    for (from_state, symbol), to_state in sorted(all_transitions.items(), key=lambda x: state_names[x[0][0]]):
        print(f"  {state_names[from_state]} --{symbol}--> {state_names[to_state]}")

# Visualizar la construcción
visualize_subset_construction(nfa_ab)
```

```{admonition} 🎯 Conexión con el Proyecto
:class: important
En el pipeline de XGrammar, las expresiones regulares de la gramática se convierten primero a NFAs (fáciles de construir) y luego a DFAs (eficientes de ejecutar). La construcción de subconjuntos es la técnica clave que permite esta transformación, generando autómatas optimizados para el análisis léxico de alto rendimiento.
```

```{admonition} ⚠️ Error Común
:class: warning
Al calcular la epsilon-clausura, los estudiantes frecuentemente olvidan incluir el estado inicial mismo. Recuerda: epsilon-clausura(q) siempre incluye q, más todos los estados alcanzables solo con transiciones ε.
```

```{admonition} Resumen
:class: important
**Conceptos clave:**
- NFAs permiten múltiples transiciones y transiciones epsilon (ε), facilitando el diseño
- La epsilon-clausura calcula todos los estados alcanzables sin consumir entrada
- El algoritmo de construcción de subconjuntos convierte NFAs a DFAs equivalentes
- La minimización reduce el DFA a su forma más compacta sin cambiar el lenguaje aceptado

**Para la siguiente lectura:**
Estudiaremos las Gramáticas Libres de Contexto (CFGs), el siguiente nivel en la jerarquía de Chomsky que permite expresar estructuras anidadas y recursivas.
```

## Ejercicios

1. **NFA para alternancia**: Diseña un NFA para (a|b)*c

2. **ε-clausura**: Dado este NFA, calcula ε-clausura para cada estado:
   ```
   q₀ -ε-> q₁ -a-> q₂
   q₀ -b-> q₃
   q₁ -ε-> q₄
   q₄ -ε-> q₁
   ```

3. **Simulación NFA**: Simula el procesamiento de "ba" en el NFA anterior.

4. **Subset construction**: Convierte este NFA a DFA:
   ```
   q₀ -a-> q₁
   q₀ -a-> q₂
   q₁ -b-> q₃ (accept)
   q₂ -c-> q₃ (accept)
   ```

5. **Minimización**: ¿Cuál es el DFA mínimo para (a|b)*?

## Preguntas de Reflexión

- ¿Cuál es el peor caso de tamaño cuando convertimos un NFA a DFA (número de estados)?
- ¿Por qué es importante la ε-clausura en la conversión?
- En XGrammar, cuando compilamos una gramática, ¿dónde aparecen procesos similares a NFA→DFA?

---

## Referencias

- Hopcroft, J. (1971). [An n log n Algorithm for Minimizing States in a Finite Automaton](https://doi.org/10.1016/B978-0-12-417750-5.50022-1). Theory of Machines and Computations.
- Thompson, K. (1968). [Programming Techniques: Regular Expression Search Algorithm](https://doi.org/10.1145/363347.363387). Communications of the ACM.
- Sipser, M. (2012). Introduction to the Theory of Computation (3rd ed.). Cengage Learning.
