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

# Fundamentos de Probabilidad

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/01-fundamentos-probabilidad.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q plotly
print('Dependencies installed!')
```
## Semana 1 - Estadística para Generación de Kernels GPU

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Definir y trabajar con espacios muestrales y eventos
- Calcular probabilidades usando los axiomas de Kolmogorov
- Aplicar el Teorema de Bayes para invertir probabilidades condicionales
- Interpretar variables aleatorias discretas y sus distribuciones
- Calcular valor esperado y varianza de distribuciones
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[StatQuest: Probability Concepts](https://www.youtube.com/watch?v=uzkc-qNVoOk)** - Josh Starmer explica de forma visual y amigable los conceptos fundamentales de probabilidad, incluyendo espacios muestrales y distribuciones.
```

```{admonition} 🔧 Herramienta Interactiva
:class: seealso

**[Seeing Theory (Brown University)](https://seeing-theory.brown.edu/)** - Posiblemente el mejor libro de texto interactivo de estadística en internet. Usa animaciones y minijuegos para hacer conceptos como probabilidad, Teorema del Límite Central e Inferencia Bayesiana completamente intuitivos.
```
:::{figure} ../images/seeing_theory.png
:name: seeing-theory
:alt: Captura de pantalla de la herramienta Seeing Theory
:align: center
:width: 100%

**Figura 1:** Portada visual de *Seeing Theory* (Brown University).
:::

Bienvenida a este módulo de estadística. Aquí aprenderemos los conceptos fundamentales que necesitas para analizar los resultados de tu investigación sobre generación de kernels GPU con restricciones gramaticales. Imagina que necesitas comparar cuántas iteraciones toma el algoritmo baseline versus tu versión mejorada. La probabilidad es el lenguaje que usamos para hablar sobre esta incertidumbre.

## Espacio Muestral y Eventos

```{admonition} 💡 Intuición
:class: hint
Antes de definir formalmente, pensemos: si lanzas una moneda, ¿cuáles son todos los posibles resultados? Cara o cruz. Ese conjunto de posibilidades es tu **espacio muestral**. En tu proyecto, si ejecutas tu generador 100 veces, ¿cuántos kernels válidos puedes obtener? Desde 0 hasta 100. Ese es tu espacio muestral.
```

Comencemos con lo más básico: **¿qué es un espacio muestral?** En tu proyecto, podrías ejecutar tu generador de kernels 100 veces y contar cuántos kernels generados son válidos (compilables). El **espacio muestral** Ω es el conjunto de todos los posibles resultados. En este caso:

```
Ω = {0, 1, 2, ..., 100}
```

Representa todos los números posibles de kernels válidos en 100 intentos.

```{code-cell} ipython3
import json
import numpy as np
import pandas as pd

# ── DATOS REALES del proyecto ──────────────────────────────
# Fuente: notebook_test_report.json (ejecución nbclient en CPU local)
with open("../../notebook_test_report.json") as f:
    report = json.load(f)

# Clasificar resultados
categorias = {"Success": 0, "Env Limit": 0, "Error": 0}
for valor in report.values():
    if valor == "Success":
        categorias["Success"] += 1
    elif valor.startswith("Env Limit"):
        categorias["Env Limit"] += 1   # xgrammar/triton no disponibles en CPU
    else:
        categorias["Error"] += 1

total = sum(categorias.values())
print(f"Experimento real: {total} notebooks ejecutados")
print(f"Espacio muestral Ω = {{Success, Env Limit, Error}}")
print()

for cat, n in categorias.items():
    print(f"  P({cat}) = {n}/{total} = {n/total:.3f}  ({n/total:.1%})")
```

Habiendo calculado empíricamente las probabilidades de cada categoría en nuestro espacio muestral, podemos visualizarlas para tener una mejor perspetiva de la distribución de nuestros resultados.

```{code-cell} ipython3
import matplotlib.subplots as subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ──────────────────────────────────────────────────────────
# Visualización: distribución real de resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: frecuencias absolutas
colores = ['#2ca02c', '#ff7f0e', '#d62728']
barras = ax1.bar(categorias.keys(), categorias.values(),
                 color=colores, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Número de notebooks', fontsize=12)
ax1.set_title('Resultados Reales de Ejecución\n(48 notebooks del curso, CPU local)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

for barra, n in zip(barras, categorias.values()):
    ax1.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.3,
             f'{n} ({n/total:.0%})', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gráfico 2: probabilidades como espacio muestral
ax2.pie(categorias.values(),
        labels=[f"{k}\n{v/total:.1%}" for k, v in categorias.items()],
        colors=colores, autopct='', startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
ax2.set_title('Probabilidad de Cada Categoría\n(P(·) empírica del experimento real)', fontsize=12)

plt.tight_layout()
plt.show()
```

Como podemos observar en los gráficos, la mayoría de los errores provienen de limitaciones del entorno ("Env Limit") y no de bugs en el código per se. 

Con estos datos en mano, podemos definir **eventos** específicos de interés y verificar los axiomas de la probabilidad:

```{code-cell} ipython3
# ── Eventos derivados ──────────────────────────────────────
P_success = categorias['Success'] / total
P_env = categorias['Env Limit'] / total
P_error = categorias['Error'] / total

print(f"\nEventos de interés para tu investigación:")
print(f"P(notebook ejecuta exitosamente)         = {P_success:.3f}")
print(f"P(falla por entorno, no código)          = {P_env:.3f}")
print(f"P(falla por bug de código)               = {P_error:.3f}")
print(f"\nVerificación axioma de normalización:    {P_success + P_env + P_error:.3f} = 1 ✓")

print(f"""
💡 Interpretación:
  • El {P_success*100:.0f}% de los notebooks corren bien; los problemas son principalmente de ENTORNO (xgrammar/triton)
    no del código del curso → P(bug | falla) = {P_error / (P_env + P_error):.1%}
  • Para tu proyecto: mide qué % de kernels generados compilan.
    Ese número estimará tu P(kernel válido).
""")
```

Un **evento** es un subconjunto del espacio muestral. Por ejemplo:
- Evento A: "El notebook ejecuta sin errores de código" → A = {Success}
- Evento B: "El notebook falla por limitación de entorno" → B = {Env Limit}


### Operaciones entre Eventos

Así como combinamos conjuntos, combinamos eventos:
- **Unión (A ∪ B)**: "Menos del 50% O más del 80% son válidos"
- **Intersección (A ∩ B)**: "Menos del 50% Y más del 80% son válidos" (imposible, por lo que es vacío)
- **Complemento (A^c)**: "No más del 80% son válidos"

## Axiomas de Probabilidad

Ahora que tenemos espacios y eventos, necesitamos una forma consistente de asignar probabilidades. Los **axiomas de Kolmogorov** nos dan tres reglas fundamentales:

**Axioma 1: No negatividad**
Para cualquier evento A: P(A) ≥ 0

Las probabilidades no pueden ser negativas. Tiene sentido.

**Axioma 2: Certeza**
P(Ω) = 1

Algo debe ocurrir. La probabilidad de que ocurra alguno de los resultados posibles es 1.

**Axioma 3: Aditividad**
Si los eventos A₁, A₂, A₃, ... son mutuamente excluyentes (no pueden ocurrir simultáneamente):
```
P(A₁ ∪ A₂ ∪ A₃ ∪ ...) = P(A₁) + P(A₂) + P(A₃) + ...
```

Estas reglas garantizan que nuestras probabilidades sean coherentes y útiles.

### Consecuencias Útiles

De estos axiomas, derivamos:
- P(A) + P(A^c) = 1
- P(∅) = 0
- Si A ⊆ B, entonces P(A) ≤ P(B)

## Probabilidad Condicional y Bayes

Aquí es donde las cosas se ponen interesantes. **¿Qué pasa si sabemos algo más sobre la situación?**

En tu proyecto, podrías preguntarte: "Si el kernel compiló exitosamente, ¿cuál es la probabilidad de que haya usado decoding con restricciones gramaticales?" Esta es una **probabilidad condicional**.

La probabilidad condicional de A dado B se denota P(A|B) y se define como:

```
P(A|B) = P(A ∩ B) / P(B)    [si P(B) > 0]
```

Intuitivamente, nos enfocamos solo en los casos donde B ocurrió y preguntamos qué fracción de esos casos también tienen A.

### Ejemplo Práctico

Supongamos que tenemos 1000 ejecuciones de nuestro generador:

| | Compiló | No compiló | Total |
|---|---------|-----------|-------|
| Baseline | 750 | 250 | 1000 |
| Restricciones | 850 | 150 | 1000 |
| Total | 1600 | 400 | 2000 |

P(Compiló) = 1600/2000 = 0.8
P(Compiló | Baseline) = 750/1000 = 0.75
P(Compiló | Restricciones) = 850/1000 = 0.85

Notemos que P(Compiló | Restricciones) > P(Compiló). Las restricciones parecen mejorar la compilabilidad.

```{code-cell} ipython3
import pandas as pd

# Datos del experimento simulado
data = {
    'Método': ['Baseline', 'Baseline', 'Restricciones', 'Restricciones'],
    'Resultado': ['Compiló', 'No compiló', 'Compiló', 'No compiló'],
    'Frecuencia': [750, 250, 850, 150]
}

df = pd.DataFrame(data)

# Calcular probabilidades condicionales
prob_compilo = (750 + 850) / 2000
prob_compilo_baseline = 750 / 1000
prob_compilo_restricciones = 850 / 1000

print("Probabilidades:")
print(f"P(Compiló) = {prob_compilo:.3f}")
print(f"P(Compiló | Baseline) = {prob_compilo_baseline:.3f}")
print(f"P(Compiló | Restricciones) = {prob_compilo_restricciones:.3f}")
print(f"\nMejora con restricciones: {(prob_compilo_restricciones - prob_compilo_baseline):.3f} ({(prob_compilo_restricciones - prob_compilo_baseline)*100:.1f}%)")
```

Visualicemos estas diferencias para entender más claro el impacto de nuestro método de restricciones versus el baseline:

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras apiladas
pivot_df = df.pivot(index='Método', columns='Resultado', values='Frecuencia')
pivot_df.plot(kind='bar', stacked=True, ax=axes[0], color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Frecuencia')
axes[0].set_xlabel('Método')
axes[0].set_title('Distribución de Resultados por Método')
axes[0].legend(title='Resultado')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Gráfico de probabilidades condicionales
metodos = ['P(C)', 'P(C|B)', 'P(C|R)']
probabilidades = [prob_compilo, prob_compilo_baseline, prob_compilo_restricciones]
colores_bar = ['blue', 'orange', 'green']

bars = axes[1].bar(metodos, probabilidades, color=colores_bar, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Probabilidad')
axes[1].set_title('Probabilidades de Compilación\n(C=Compiló, B=Baseline, R=Restricciones)')
axes[1].set_ylim(0, 1)
axes[1].grid(axis='y', alpha=0.3)

# Agregar valores sobre las barras
for bar, prob in zip(bars, probabilidades):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Teorema de Bayes

```{admonition} 🎯 Aplicación en ML
:class: important
En el contexto de evaluar kernels GPU, el Teorema de Bayes te permite responder preguntas como: "Si mi kernel compila correctamente, ¿cuál es la probabilidad de que esté realmente optimizado?" Esto es crucial cuando usas detectores automáticos de calidad que pueden dar falsos positivos.
```

El **Teorema de Bayes** es una de las herramientas más poderosas en probabilidad:

:::{figure} diagrams/bayes_tree.png
:name: fig-bayes-tree
:alt: Diagrama de árbol de probabilidades demostrando el teorema de Bayes
:align: center
:width: 90%

**Figura 2:** Árbol de probabilidades ilustrando el teorema de Bayes y el cálculo de probabilidades posteriores.
:::

```
P(A|B) = P(B|A) × P(A) / P(B)
```

¿Por qué es tan importante? Porque con frecuencia queremos invertir el sentido de una probabilidad condicional. Sabemos qué tan probable es que veamos cierta evidencia si una hipótesis es verdadera, pero queremos saber cuán probable es la hipótesis dada la evidencia.

### Ejemplo: Diagnóstico

Imagina que tenemos un "detector de calidad" que identifica kernels optimizados:
- P(detector dice "optimizado" | kernel es realmente optimizado) = 0.95
- P(detector dice "optimizado" | kernel NO es optimizado) = 0.10
- P(kernel es realmente optimizado en la población) = 0.05

Si el detector dice que un kernel es optimizado, ¿cuál es la probabilidad de que realmente lo sea?

```
P(realmente optimizado | detector dice optimizado)
= P(detector dice optimizado | realmente optimizado) × P(realmente optimizado) / P(detector dice optimizado)
```

Primero calculamos P(detector dice optimizado):
```
P(detector dice optimizado)
= P(detector | optimizado) × P(optimizado) + P(detector | no optimizado) × P(no optimizado)
= 0.95 × 0.05 + 0.10 × 0.95
= 0.0475 + 0.095
= 0.1425
```

Entonces:
```
P(realmente optimizado | detector) = 0.95 × 0.05 / 0.1425 ≈ 0.333
```

¡Sorpresa! Incluso con un detector que tiene 95% de exactitud, cuando dice que un kernel es optimizado, solo hay 33% de probabilidad de que realmente lo sea. Esto es porque los kernels optimizados son raros en la población (solo 5%). Este es un hallazgo crucial para entender la precisión en tareas desbalanceadas.

```{code-cell} ipython3
import numpy as np

# Parámetros del problema de Bayes
P_detector_dado_opt = 0.95  # P(detector dice "opt" | realmente opt)
P_detector_dado_no_opt = 0.10  # P(detector dice "opt" | NO opt)
P_opt = 0.05  # P(realmente optimizado)
P_no_opt = 0.95  # P(NO optimizado)

# Teorema de Bayes
P_detector = P_detector_dado_opt * P_opt + P_detector_dado_no_opt * P_no_opt
P_opt_dado_detector = (P_detector_dado_opt * P_opt) / P_detector

print("Teorema de Bayes - Diagnóstico de Kernels Optimizados")
print("=" * 60)
print(f"P(detector dice 'opt' | realmente opt) = {P_detector_dado_opt:.2f}")
print(f"P(detector dice 'opt' | NO opt) = {P_detector_dado_no_opt:.2f}")
print(f"P(realmente opt en población) = {P_opt:.2f}")
print(f"\nP(detector dice 'opt') = {P_detector:.4f}")
print(f"P(realmente opt | detector dice 'opt') = {P_opt_dado_detector:.3f}")
print(f"\n¡Solo {P_opt_dado_detector*100:.1f}% de probabilidad de ser realmente optimizado!")
```

Esta dramática caída (del 95% de precisión aparente al 33% de probabilidad posterior) es un ejemplo clásico de la **Falacia de la Tasa Base**. Debido a que los kernels optimizados son muy raros (sólo un 5%), el gran volumen de kernels no optimizados generará más "falsos positivos" que el total de optimizados reales. Podemos ver este fenómeno visualmente mediante un árbol de probabilidades:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Visualización con diagrama de árbol de probabilidades
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Izquierda: Árbol de probabilidades
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Árbol de Probabilidades - Teorema de Bayes', fontsize=14, fontweight='bold')

# Nivel 1: Prior
ax.text(1, 9, 'Población', fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue'))

# Nivel 2: Optimizado o no
ax.arrow(1, 8.5, 1.5, -2, head_width=0.2, head_length=0.2, fc='green', ec='green')
ax.arrow(1, 8.5, 1.5, -1, head_width=0.2, head_length=0.2, fc='red', ec='red')

ax.text(3, 6.2, f'Opt\n({P_opt:.2f})', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(3, 7.2, f'NO Opt\n({P_no_opt:.2f})', fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral'))

# Nivel 3: Detector
ax.arrow(3, 6, 2, -1.5, head_width=0.15, head_length=0.15, fc='blue', ec='blue')
ax.arrow(3, 6, 2, -0.5, head_width=0.15, head_length=0.15, fc='gray', ec='gray')
ax.arrow(3, 7, 2, 0.5, head_width=0.15, head_length=0.15, fc='blue', ec='blue')
ax.arrow(3, 7, 2, -0.5, head_width=0.15, head_length=0.15, fc='gray', ec='gray')

# Resultados finales
ax.text(5.5, 4.3, f'Detecta\n({P_detector_dado_opt:.2f})', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(5.5, 5.3, f'NO detecta\n({1-P_detector_dado_opt:.2f})', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(5.5, 7.3, f'Detecta\n({P_detector_dado_no_opt:.2f})', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(5.5, 6.3, f'NO detecta\n({1-P_detector_dado_no_opt:.2f})', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Probabilidades conjuntas
prob1 = P_opt * P_detector_dado_opt
prob2 = P_opt * (1 - P_detector_dado_opt)
prob3 = P_no_opt * P_detector_dado_no_opt
prob4 = P_no_opt * (1 - P_detector_dado_no_opt)

ax.text(7.5, 4.3, f'P={prob1:.4f}', fontsize=9, ha='center', color='green', fontweight='bold')
ax.text(7.5, 5.3, f'P={prob2:.4f}', fontsize=9, ha='center')
ax.text(7.5, 7.3, f'P={prob3:.4f}', fontsize=9, ha='center', color='red', fontweight='bold')
ax.text(7.5, 6.3, f'P={prob4:.4f}', fontsize=9, ha='center')

# Derecha: Comparación de probabilidades
ax2 = axes[1]
categorias = ['P(Opt | Detector+)', 'P(NO Opt | Detector+)']
valores = [P_opt_dado_detector, 1 - P_opt_dado_detector]
colores = ['green', 'red']

bars = ax2.bar(categorias, valores, color=colores, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Probabilidad', fontsize=12)
ax2.set_title('Probabilidades Posteriores\n(dado que el detector dice "optimizado")', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, valores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.1%}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nInterpretación Visual:")
print("Aunque el detector tiene 95% de sensibilidad, la baja prevalencia (5%)")
print("hace que la mayoría de detecciones positivas sean falsos positivos.")
```

```{code-cell} ipython3
# Simulación del Teorema de Bayes
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# P(Enfermedad) = 1%, Sensibilidad = 99%, Especificidad = 95%
prior = 0.01
sensitivity = 0.99
specificity = 0.95

# Calcular P(Enfermedad | Test+)
p_positive = sensitivity * prior + (1 - specificity) * (1 - prior)
posterior = (sensitivity * prior) / p_positive

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                    subplot_titles=("Antes del Test (Prior)", "Después del Test+ (Posterior)"))

fig.add_trace(go.Pie(values=[prior, 1-prior], labels=['Enfermo', 'Sano'],
                     marker_colors=['#FF6B6B', '#4ECDC4']), row=1, col=1)
fig.add_trace(go.Pie(values=[posterior, 1-posterior], labels=['Enfermo', 'Sano'],
                     marker_colors=['#FF6B6B', '#4ECDC4']), row=1, col=2)

fig.update_layout(title=f"Teorema de Bayes: P(Enfermo|Test+) = {posterior:.1%}", height=350)
fig.show()
```

## Variables Aleatorias Discretas

Una **variable aleatoria** es una función que asigna números a resultados. Es "aleatoria" porque el resultado aún es incierto antes de que ocurra.

Formalmente, X : Ω → ℝ. Para nuestro ejemplo de kernels:
- X = "número de kernels válidos en 100 intentos" (puede ser 0-100)

Una variable aleatoria es **discreta** si toma un número finito o contable de valores.

### Función de Masa de Probabilidad (PMF)

La **PMF** describe la probabilidad de cada valor posible:

```
p(x) = P(X = x)
```

Por ejemplo, si generamos 5 kernels y X = "número de kernels válidos":

```
x    | 0    | 1    | 2    | 3    | 4    | 5    |
-----|------|------|------|------|------|------|
p(x) | 0.01 | 0.05 | 0.15 | 0.30 | 0.35 | 0.14 |
```

Una PMF válida debe cumplir:
- p(x) ≥ 0 para todo x
- Σ p(x) = 1

## Valor Esperado y Varianza

**El valor esperado** E[X] es el promedio ponderado de todos los resultados posibles:

```
E[X] = Σ x × p(x)
```

Es lo que esperamos "en promedio" si repitiéramos el experimento muchas veces.

Para nuestro ejemplo:
```
E[X] = 0(0.01) + 1(0.05) + 2(0.15) + 3(0.30) + 4(0.35) + 5(0.14)
     = 0 + 0.05 + 0.30 + 0.90 + 1.40 + 0.70
     = 3.35 kernels válidos (en promedio)
```

**La varianza** mide cuánta dispersión hay alrededor de la media:

```
Var(X) = E[(X - E[X])²]
       = E[X²] - (E[X])²
```

Es útil porque nos dice si nuestros resultados son consistentes (varianza baja) o erráticos (varianza alta).

```{code-cell} ipython3
import numpy as np

# Función de masa de probabilidad (PMF)
x = np.array([0, 1, 2, 3, 4, 5])
p_x = np.array([0.01, 0.05, 0.15, 0.30, 0.35, 0.14])

# Calcular valor esperado
E_X = np.sum(x * p_x)

# Calcular varianza y desviación estándar
E_X2 = np.sum(x**2 * p_x)
Var_X = E_X2 - E_X**2
SD_X = np.sqrt(Var_X)

print("Variable Aleatoria: Número de kernels válidos en 5 intentos")
print("=" * 60)
print("\nFunción de Masa de Probabilidad (PMF):")
for xi, pi in zip(x, p_x):
    print(f"P(X = {xi}) = {pi:.2f}")

print(f"\nValor Esperado: E[X] = {E_X:.2f} kernels")
print(f"Varianza: Var(X) = {Var_X:.2f}")
print(f"Desviación Estándar: SD(X) = {SD_X:.2f}")
```

La visualización de estas medidas te ayudará a desarrollar un sentido intuitivo del Valor Esperado y la Distribución Acumulativa (CDF):

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Visualizaciones de PMF, CDF y E[X]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: PMF
axes[0].bar(x, p_x, color='steelblue', alpha=0.7, edgecolor='black', width=0.6)
axes[0].set_xlabel('Número de kernels válidos (X)')
axes[0].set_ylabel('Probabilidad P(X)')
axes[0].set_title('Función de Masa de Probabilidad (PMF)')
axes[0].set_xticks(x)
axes[0].grid(axis='y', alpha=0.3)

for xi, pi in zip(x, p_x):
    axes[0].text(xi, pi + 0.01, f'{pi:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: CDF acumulativa
cdf = np.cumsum(p_x)
axes[1].step(x, cdf, where='mid', linewidth=2, color='darkgreen', label='CDF')
axes[1].scatter(x, cdf, s=100, color='darkgreen', zorder=3)
axes[1].set_xlabel('Número de kernels válidos (X)')
axes[1].set_ylabel('Probabilidad Acumulada P(X ≤ x)')
axes[1].set_title('Función de Distribución Acumulativa (CDF)')
axes[1].set_xticks(x)
axes[1].grid(alpha=0.3)
axes[1].legend()

for xi, ci in zip(x, cdf):
    axes[1].text(xi + 0.1, ci - 0.05, f'{ci:.2f}', fontsize=9)

# Plot 3: Visualización de E[X] y desviación estándar
axes[2].bar(x, p_x, color='lightblue', alpha=0.7, edgecolor='black', width=0.6)
axes[2].axvline(E_X, color='red', linewidth=3, linestyle='--', label=f'E[X] = {E_X:.2f}')
axes[2].axvline(E_X - SD_X, color='orange', linewidth=2, linestyle=':', label=f'E[X] ± SD')
axes[2].axvline(E_X + SD_X, color='orange', linewidth=2, linestyle=':')
axes[2].fill_betweenx([0, max(p_x)], E_X - SD_X, E_X + SD_X,
                       alpha=0.2, color='orange', label=f'SD = {SD_X:.2f}')
axes[2].set_xlabel('Número de kernels válidos (X)')
axes[2].set_ylabel('Probabilidad P(X)')
axes[2].set_title('Valor Esperado y Desviación Estándar')
axes[2].set_xticks(x)
axes[2].grid(axis='y', alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()
```

Por último, los conceptos de Variable Aleatoria y Valor Esperado rigen tanto a nivel teórico como experimental. Si tuviéramos que simular 10,000 intentos basados en nuestra PMF, los resultados convergirían exactamente con la teoría (Ley de los Grandes Números):

```{code-cell} ipython3
# Simulación empírica para verificar el cálculo teórico
np.random.seed(42)
num_simulaciones = 10000
simulaciones = np.random.choice(x, size=num_simulaciones, p=p_x)

print(f"\nVerificación con {num_simulaciones} simulaciones:")
print(f"Media empírica simulada:  {np.mean(simulaciones):.3f} (teórico: {E_X:.2f})")
print(f"Varianza muestral simulada: {np.var(simulaciones, ddof=1):.3f} (teórico: {Var_X:.2f})")
```

### Propiedades Importantes

```
E[aX + b] = aE[X] + b
Var(aX + b) = a² × Var(X)
```

Si multiplicas todos los resultados por 2, el valor esperado también se multiplica por 2, pero la varianza se multiplica por 4.

## Resumen de Conceptos

```{admonition} Resumen
:class: important
**Checklist de Conceptos Fundamentales:**
- [x] **Espacio muestral (Ω)**: Conjunto de todos los resultados posibles
- [x] **Evento**: Subconjunto del espacio muestral (ej. "más del 80% válidos")
- [x] **Axiomas de Probabilidad**: No negatividad, certeza, aditividad
- [x] **Probabilidad Condicional P(A|B)**: Probabilidad de A dado que B ocurrió
- [x] **Teorema de Bayes**: Permite invertir probabilidades condicionales
- [x] **Variable Aleatoria**: Función que asigna números a resultados
- [x] **PMF**: Probabilidad de cada valor discreto
- [x] **Valor Esperado E[X]**: Promedio ponderado de todos los resultados
- [x] **Varianza Var(X)**: Medida de dispersión alrededor de la media
```

| Concepto | Definición | En tu proyecto |
|----------|-----------|-----------------|
| Espacio muestral | Todos los resultados posibles | Todos los kernels posibles |
| Evento | Subconjunto del espacio muestral | "Kernel compiló" |
| Probabilidad condicional | P(A\|B) | ¿Validez dado que usamos restricciones? |
| Teorema de Bayes | Invertir probabilidades condicionales | Diagnosticar si un kernel es óptimo |
| Variable aleatoria | Función numérica de resultados | Número de kernels válidos |
| PMF | Probabilidad de cada valor discreto | Distribución de validez |
| Valor esperado | Promedio ponderado | Promedio de iteraciones |
| Varianza | Dispersión alrededor de la media | Consistencia del algoritmo |

## Ejercicios y Reflexión

```{admonition} ✅ Verifica tu comprensión
:class: note
1. **Conceptual**: ¿Cuál es la diferencia entre P(A|B) y P(B|A)? Da un ejemplo con kernels GPU.
2. **Aplicación**: Si P(kernel compilado) = 0.80 y P(kernel compilado | restricciones) = 0.85, ¿qué dice esto sobre la efectividad de las restricciones?
3. **Interpretación**: Si el valor esperado de iteraciones es 42, ¿significa que siempre obtendrás exactamente 42 iteraciones?
4. **Bayes**: ¿Por qué un detector con 95% de exactitud puede tener solo 33% de probabilidad de ser correcto cuando detecta algo?
```

```{admonition} 🧠 Preguntas de Opción Múltiple
:class: tip
**P1.** P(A|B) = 0.9 y P(B|A) = 0.4. ¿Cuál afirmación es verdadera?

- a) **P(A|B) ≠ P(B|A) en general; son probabilidades diferentes** ✓
- b) P(A) = P(B) porque sus probabilidades condicionales están relacionadas
- c) P(A|B) × P(B|A) = P(A ∩ B)
- d) No se pueden calcular sin conocer P(A) y P(B)

*Razón: La probabilidad condicional P(A|B) = P(A∩B)/P(B); invertir el orden cambia el denominador*

---

**P2.** En los datos reales del curso: 36 notebooks ejecutaron exitosamente de 48. Si se elige un notebook al azar, P(falla) es:

- a) 36/48 = 0.75
- b) **12/48 = 0.25** ✓
- c) 8/48 = 0.167
- d) 4/48 = 0.083

*Razón: P(falla) = 1 - P(éxito) = 1 - 0.75 = 0.25 (axioma de complemento)*

---

**P3.** Si X = número de kernels válidos en 10 intentos con P(válido) = 0.8, el Valor Esperado E[X] es:

- a) 0.8
- b) 8.0 pero con mucha variación
- c) **E[X] = n × p = 10 × 0.8 = 8** ✓
- d) E[X] = 0.8^10 ≈ 0.107

*Razón: Para distribución binomial B(n, p), E[X] = n·p. Esto no significa que siempre obtendrás exactamente 8.*
```


### Ejercicio 1: Espacio Muestral y Eventos
En tu experimento de generación de kernels, ejecutas el baseline 20 veces y cuentas cuántos son válidos.
- Define el espacio muestral Ω
- Define el evento A: "al menos 15 kernels son válidos"
- Define el evento B: "exactamente 10 kernels son válidos"
- ¿Son A y B mutuamente excluyentes? ¿Es B ⊆ A?

### Ejercicio 2: Probabilidad Condicional
Tienes datos:
- P(kernel válido) = 0.80
- P(kernel válido | método A) = 0.75
- P(kernel válido | método B) = 0.85

Calcula P(método A | kernel válido) y P(método B | kernel válido) usando Bayes.

### Ejercicio 3: Variables Aleatorias
Si X = número de kernels válidos en 10 intentos y p(X=k) = C(10,k) × 0.8^k × 0.2^(10-k):
- Calcula E[X]
- Calcula Var(X)
- ¿Cuál es la desviación estándar?

### Reflexión
1. En tu proyecto, ¿cuál es el espacio muestral natural? ¿Cuáles son los eventos de interés?
2. ¿Hay relaciones causales o solo probabilidades condicionales? ¿Cómo esto afecta tus conclusiones?
3. Si observas que P(compiló | restricciones) > P(compiló | baseline), ¿qué puedes concluir? ¿Qué información adicional necesitarías?

```{admonition} 📋 Rúbrica de Evaluación (Ejercicios y Reflexiones)
:class: note
- **Excelente (100%)**: Cálculos de probabilidad precisos. La reflexión responde claramente usando términos estadísticos correctos (eventos, independencia, causalidad) y contextualiza perfectamente con el problema de los kernels GPU.
- **Bueno (80%)**: Cálculos correctos pero la reflexión es superficial o confunde correlación con causalidad en sus explicaciones.
- **Requiere Mejora (<60%)**: Errores en los cálculos de Bayes o Valor Esperado. Reflexión que no aplica los conceptos al dominio del proyecto.
```


---

**Próxima semana**: Aprenderemos distribuciones específicas (binomial, normal, Poisson) y estadística descriptiva para resumir nuestros datos.

---

## Referencias

- Kolmogorov, A. (1933). Foundations of the Theory of Probability. Chelsea Publishing.
- Ross, S. (2014). [A First Course in Probability](https://www.pearson.com/us/higher-education/program/Ross-First-Course-in-Probability-A-9th-Edition/PGM110742.html) (9th ed.). Pearson.
- DeGroot, M. & Schervish, M. (2012). Probability and Statistics (4th ed.). Pearson.

---

## Lecturas Recomendadas

- **D2L: Probability and Statistics** - [Capítulo 2.6](https://d2l.ai/chapter_preliminaries/probability.html). Referencia sólida para los fundamentos de probabilidad aplicados a ML.
