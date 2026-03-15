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

# Distribuciones y Estadística Descriptiva

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/02-distribuciones-descriptiva.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton
print('Dependencies installed!')
```
## Semana 2 - Estadística para Generación de Kernels GPU

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Identificar cuándo usar distribuciones Bernoulli, Binomial, Poisson y Normal
- Calcular probabilidades usando estas distribuciones
- Aplicar el Teorema Central del Límite a promedios muestrales
- Interpretar estadísticas descriptivas (media, mediana, desviación estándar)
- Detectar valores atípicos usando cuartiles e IQR
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[StatQuest: The Normal Distribution](https://www.youtube.com/watch?v=rzFX5NWojp0)** - Explicación clara y visual de la distribución normal, incluyendo la regla empírica 68-95-99.7 y cómo estandarizar con Z-scores.
```

```{admonition} 🔧 Herramienta Interactiva
:class: seealso

**[Seeing Theory (Brown University)](https://seeing-theory.brown.edu/)** - Explora las distribuciones de probabilidad de forma interactiva, ajustando parámetros y viendo cómo cambian las curvas en tiempo real.

**[Probability Playground (Distributome)](https://www.distributome.org/V3/)** - Explora cómo se interrelacionan docenas de distribuciones de probabilidad diferentes alterando sus parámetros.
```

Ahora que entiendes probabilidad fundamental, es tiempo de aprender sobre distribuciones específicas. Una distribución describe cómo se comportan nuestros datos. En tu proyecto, necesitarás saber si tus conteos de iteraciones siguen una distribución normal, Poisson, o binomial. Esto determinará qué pruebas estadísticas puedes usar después.

## Distribución de Bernoulli

```{admonition} 💡 Intuición
:class: hint
Antes de ver fórmulas, pensemos: una prueba de Bernoulli es como lanzar una moneda (posiblemente cargada). Solo hay dos resultados: cara o cruz. En tu proyecto, cada intento de generar un kernel es así: o compila (éxito) o no compila (fracaso). Es el bloque básico de construcción para distribuciones más complejas.
```

La más simple de todas. Una **prueba de Bernoulli** es un experimento con solo dos resultados: éxito o fracaso.

```
X ~ Bernoulli(p)

P(X = 1) = p
P(X = 0) = 1 - p

E[X] = p
Var(X) = p(1-p)
```

```{admonition} 🎯 Aplicación en ML
:class: important
En tu proyecto de kernels GPU, cada generación es una prueba de Bernoulli: ¿Compiló (1) o no compiló (0)? Si p = 0.82, entonces esperas que 82% de tus kernels compilen exitosamente.
```

Por ejemplo, si p = 0.82, entonces:
- E[X] = 0.82 (esperamos éxito 82% de las veces)
- Var(X) = 0.82 × 0.18 = 0.1476

## Distribución Binomial

Si repites una prueba de Bernoulli n veces de forma independiente, el número total de éxitos sigue una **distribución binomial**.

```
X ~ Binomial(n, p)

P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

donde C(n,k) = n! / (k!(n-k)!)

E[X] = np
Var(X) = np(1-p)
SD(X) = √(np(1-p))
```

### Ejemplo Práctico

Ejecutas tu generador 50 veces. Cada intento tiene p = 0.82 de compilación exitosa. ¿Cuántos kernels válidos esperas?

```
X ~ Binomial(50, 0.82)
E[X] = 50 × 0.82 = 41 kernels válidos
Var(X) = 50 × 0.82 × 0.18 = 7.38
SD(X) = √7.38 ≈ 2.72
```

Entonces esperas alrededor de 41 kernels válidos, con una desviación estándar de 2.72. La mayoría de las veces estarás entre 38-44.

```{code-cell} ipython3
import numpy as np
from scipy.stats import binom, bernoulli

# Parámetros
n = 50
p = 0.82

# Distribución Binomial
x = np.arange(0, n+1)
pmf_binomial = binom.pmf(x, n, p)

# Estadísticas teóricas
mean_binomial = n * p
var_binomial = n * p * (1 - p)
std_binomial = np.sqrt(var_binomial)

print("Distribución Binomial")
print("=" * 60)
print(f"Parámetros: n = {n} intentos, p = {p} probabilidad de éxito")
print(f"E[X] = {mean_binomial:.2f} kernels válidos")
print(f"Var(X) = {var_binomial:.2f}")
print(f"SD(X) = {std_binomial:.2f}")
print(f"Rango esperado (±2 SD): [{mean_binomial - 2*std_binomial:.1f}, {mean_binomial + 2*std_binomial:.1f}]")
```

Para comprender de manera visual esta distribución y cómo se compara con un solo intento (Bernoulli), podemos graficarlas:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. PMF Binomial
axes[0].bar(x, pmf_binomial, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(mean_binomial, color='red', linewidth=2, linestyle='--',
                   label=f'Media = {mean_binomial:.1f}')
axes[0].axvline(mean_binomial - std_binomial, color='orange', linewidth=2,
                   linestyle=':', label=f'±1 SD')
axes[0].axvline(mean_binomial + std_binomial, color='orange', linewidth=2, linestyle=':')
axes[0].fill_betweenx([0, max(pmf_binomial)],
                          mean_binomial - std_binomial,
                          mean_binomial + std_binomial,
                          alpha=0.2, color='orange')
axes[0].set_xlabel('Número de kernels válidos')
axes[0].set_ylabel('Probabilidad')
axes[0].set_title(f'Distribución Binomial(n={n}, p={p})')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# 2. Comparación Bernoulli vs Binomial
axes[1].bar([0, 1], bernoulli.pmf([0, 1], p), color='lightcoral',
               alpha=0.7, edgecolor='black', width=0.3, label='Bernoulli(p=0.82)')
axes[1].set_xlabel('Resultado (0=Fallo, 1=Éxito)')
axes[1].set_ylabel('Probabilidad')
axes[1].set_title('Bernoulli: Una sola prueba')
axes[1].set_xticks([0, 1])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

for i, val in enumerate(bernoulli.pmf([0, 1], p)):
    axes[1].text(i, val + 0.02, f'{val:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

Ahora, demostremos esto estadísticamente realizando una simulación y calculando nuestra Función de Distribución Acumulativa (CDF):

```{code-cell} ipython3
# Simulación y CDF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3. Simulación
np.random.seed(42)
num_simulaciones = 1000
simulaciones = np.random.binomial(n, p, num_simulaciones)

axes[0].hist(simulaciones, bins=20, density=True, alpha=0.6,
                color='lightblue', edgecolor='black', label='Simulación')
axes[0].plot(x, pmf_binomial, 'ro-', linewidth=2, markersize=4,
                label='Teórica', alpha=0.7)
axes[0].axvline(np.mean(simulaciones), color='green', linewidth=2,
                   linestyle='--', label=f'Media sim = {np.mean(simulaciones):.1f}')
axes[0].set_xlabel('Número de kernels válidos')
axes[0].set_ylabel('Densidad de Probabilidad')
axes[0].set_title(f'Comparación: Teórica vs {num_simulaciones} Simulaciones')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 4. CDF
cdf_binomial = binom.cdf(x, n, p)
axes[1].step(x, cdf_binomial, where='mid', linewidth=2, color='darkgreen')
axes[1].fill_between(x, 0, cdf_binomial, alpha=0.3, color='green', step='mid')
axes[1].set_xlabel('Número de kernels válidos')
axes[1].set_ylabel('Probabilidad Acumulada')
axes[1].set_title('Función de Distribución Acumulativa (CDF)')
axes[1].grid(alpha=0.3)

# Marcar percentiles importantes
percentiles = [0.25, 0.5, 0.75]
for perc in percentiles:
    val = binom.ppf(perc, n, p)
    axes[1].axhline(perc, color='red', linestyle=':', alpha=0.5)
    axes[1].axvline(val, color='red', linestyle=':', alpha=0.5)
    axes[1].text(val + 1, perc - 0.05, f'Q{int(perc*100)}={val:.0f}', fontsize=9)

plt.tight_layout()
plt.show()

print(f"\nVerificación con simulación:")
print(f"Media simulada: {np.mean(simulaciones):.2f} (teórico: {mean_binomial:.2f})")
print(f"SD simulada: {np.std(simulaciones, ddof=1):.2f} (teórico: {std_binomial:.2f})")
```

```{admonition} 📊 ¿Cómo interpretar la CDF en la práctica?
:class: important

La **Función de Distribución Acumulativa (CDF)** responde preguntas del tipo *"¿Cuánto?"* y *"¿Qué tan probable?"*. En tu proyecto:

| Pregunta práctica | Cómo usarla |
|---|---|
| ¿Qué % de corridas genera ≤38 kernels válidos? | Lee F(38) en la CDF |
| ¿Cuántos kernels necesito para estar en el 90º percentil? | Encuentra x tal que F(x)=0.9 |
| ¿Es 35 kernels válidos un resultado inusual? | Si F(35) < 0.05, sí — está en el 5% inferior |
| ¿Cuál es el rango del 80% central? | Encuentra Q10 y Q90 con `binom.ppf(0.1, n, p)` y `binom.ppf(0.9, n, p)` |

**Ejemplo de lectura**: Si la CDF vale 0.25 en x=40, eso significa que el 25% de las veces obtendrás 40 o *menos* kernels válidos en 50 intentos con p=0.82. Dicho de otro modo, el 75% de las veces obtendrás más de 40.

```{code-cell} ipython3
from scipy.stats import binom
import numpy as np

n, p = 50, 0.82

# Responder preguntas concretas con la CDF
print("Consultas prácticas sobre la CDF Binomial(50, 0.82):")
print("-" * 55)

# ¿Qué tan probable es obtener ≤38 kernels?
prob_le38 = binom.cdf(38, n, p)
print(f"P(X ≤ 38) = {prob_le38:.3f}   → {prob_le38*100:.1f}% de las corridas")

# ¿Cuántos kernels para estar en el 90º percentil?
q90 = binom.ppf(0.90, n, p)
print(f"Percentil 90 = {q90:.0f}       → 90% de corridas producen ≤{q90:.0f} kernels")

# Rango del 80% central (P10 a P90)
q10 = binom.ppf(0.10, n, p)
print(f"Rango 80% central: [{q10:.0f}, {q90:.0f}]  → 80% de veces caerás aquí")

# ¿Es 35 un resultado inusual?
prob_le35 = binom.cdf(35, n, p)
print(f"\nP(X ≤ 35) = {prob_le35:.4f}   → {'❗ Inusual (< 5%)' if prob_le35 < 0.05 else 'Normal'}")
print(f"→ Si obtienes 35 kernels válidos en 50 intentos con p=0.82,")
print(f"  esto ocurriría solo el {prob_le35*100:.1f}% de las veces por azar.")
print(f"  Podría indicar un bug o que la tasa real es menor que 0.82.")
```


Cuando n es grande y p es pequeño, pero np es moderado, usamos la **distribución de Poisson**. Modela conteos de eventos raros en intervalos fijos.

```
X ~ Poisson(λ)

P(X = k) = (e^(-λ) × λ^k) / k!

E[X] = λ
Var(X) = λ
```

**Característica interesante**: En Poisson, la media y varianza son iguales.

### Cuándo usarla

- Número de errores de compilación en 1000 intentos
- Número de regiones de memoria inválidas detectadas en 100 kernels
- Número de fallos en una semana de ejecución

Si esperamos λ = 5 errores por semana:

```
P(X = 3) = (e^(-5) × 5^3) / 3! ≈ 0.1404

P(X ≤ 2) = P(X=0) + P(X=1) + P(X=2) ≈ 0.1247
```

```{code-cell} ipython3
import numpy as np
from scipy.stats import poisson

# Parámetro lambda
lambda_val = 5

# Valores de x
x = np.arange(0, 15)
pmf_poisson = poisson.pmf(x, lambda_val)

# Calcular probabilidades específicas
prob_3 = poisson.pmf(3, lambda_val)
prob_le_2 = poisson.cdf(2, lambda_val)

print("Distribución de Poisson")
print("=" * 60)
print(f"Parámetro: λ = {lambda_val} errores por semana")
print(f"E[X] = Var(X) = λ = {lambda_val}")
print(f"\nP(X = 3) = {prob_3:.4f}")
print(f"P(X ≤ 2) = {prob_le_2:.4f}")
print(f"P(X > 5) = {1 - poisson.cdf(5, lambda_val):.4f}")
```

La siguiente celda visualiza una distribución de Poisson con un $\lambda = 5$ y también la contrasta con distintos valores de $\lambda$.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Visualización de la PMF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. PMF de Poisson
axes[0].bar(x, pmf_poisson, color='purple', alpha=0.7, edgecolor='black')
axes[0].axvline(lambda_val, color='red', linewidth=2, linestyle='--',
                   label=f'λ = {lambda_val}')
axes[0].set_xlabel('Número de errores (X)')
axes[0].set_ylabel('Probabilidad')
axes[0].set_title(f'Distribución de Poisson(λ={lambda_val})')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Resaltar P(X=3)
axes[0].bar(3, pmf_poisson[3], color='red', alpha=0.8, edgecolor='black')
axes[0].text(3, pmf_poisson[3] + 0.01, f'P(3)={prob_3:.4f}',
                ha='center', fontweight='bold')

# 2. Comparación de diferentes λ
lambdas = [1, 3, 5, 10]
x_extended = np.arange(0, 20)

for lam in lambdas:
    pmf = poisson.pmf(x_extended, lam)
    axes[1].plot(x_extended, pmf, marker='o', label=f'λ={lam}', alpha=0.7)

axes[1].set_xlabel('Número de eventos (X)')
axes[1].set_ylabel('Probabilidad')
axes[1].set_title('Comparación de Distribuciones de Poisson')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

Análogamente al caso Binomial, podemos graficar la Función de Distribución Acumulatva (CDF) para entender la probabilidad acumulada de eventos que no superen una cantidad dada, así como realizar una simulación aleatoria de eventos:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3. CDF
cdf_poisson = poisson.cdf(x, lambda_val)
axes[0].step(x, cdf_poisson, where='mid', linewidth=2, color='darkgreen')
axes[0].fill_between(x, 0, cdf_poisson, alpha=0.3, color='green', step='mid')
axes[0].axhline(prob_le_2, color='red', linestyle=':', linewidth=2,
                   label=f'P(X≤2)={prob_le_2:.4f}')
axes[0].axvline(2, color='red', linestyle=':', linewidth=2)
axes[0].set_xlabel('Número de errores (X)')
axes[0].set_ylabel('Probabilidad Acumulada')
axes[0].set_title('CDF de Poisson')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 4. Simulación
np.random.seed(42)
num_simulaciones = 1000
simulaciones = np.random.poisson(lambda_val, num_simulaciones)

axes[1].hist(simulaciones, bins=range(0, 16), density=True, alpha=0.6,
                color='lightblue', edgecolor='black', label='Simulación')
axes[1].plot(x, pmf_poisson, 'ro-', linewidth=2, markersize=6,
                label='Teórica', alpha=0.7)
axes[1].set_xlabel('Número de errores')
axes[1].set_ylabel('Densidad de Probabilidad')
axes[1].set_title(f'Comparación: Teórica vs {num_simulaciones} Simulaciones')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nVerificación con simulación:")
print(f"Media simulada: {np.mean(simulaciones):.2f} (teórico: {lambda_val})")
print(f"Varianza simulada: {np.var(simulaciones, ddof=1):.2f} (teórico: {lambda_val})")
print(f"\nNota: En Poisson, media = varianza = λ")
```

```{code-cell} ipython3
# Explorador de distribuciones
import numpy as np
import plotly.graph_objects as go
from scipy import stats

x = np.linspace(-5, 10, 500)

fig = go.Figure()

# Normal
fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, 0, 1), name='Normal(0,1)', line=dict(width=2)))
fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, 2, 0.5), name='Normal(2,0.5)', line=dict(width=2)))

# T-student
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, df=3), name='t(df=3)', line=dict(width=2, dash='dash')))
fig.add_trace(go.Scatter(x=x, y=stats.t.pdf(x, df=30), name='t(df=30)', line=dict(width=2, dash='dash')))

# Exponencial
fig.add_trace(go.Scatter(x=x[x>=0], y=stats.expon.pdf(x[x>=0], scale=2), name='Exp(λ=0.5)', line=dict(width=2, dash='dot')))

fig.update_layout(
    title="Comparación de Distribuciones (hover para detalles)",
    xaxis_title="x", yaxis_title="Densidad",
    height=400, hovermode='x unified'
)
fig.show()
```

## Distribución Normal (Gaussiana)

La distribución más importante en estadística. **Muchos fenómenos naturales convergen a esta forma**.

:::{figure} diagrams/distributions_comparison.png
:name: fig-distributions-comparison
:alt: Comparación visual de Bernoulli, Binomial, Poisson y Distribuciones Normales
:align: center
:width: 90%

**Figura 1:** Comparación de las principales distribuciones de probabilidad usadas en estadística.
:::

```
X ~ N(μ, σ²)

PDF: f(x) = (1/(σ√(2π))) × e^(-(x-μ)²/(2σ²))

E[X] = μ
Var(X) = σ²
```

Propiedades:
- Simétrica alrededor de μ
- ~68% de datos dentro de μ ± σ
- ~95% de datos dentro de μ ± 2σ
- ~99.7% de datos dentro de μ ± 3σ

### Estandarización (Z-score)

Para trabajar con cualquier normal, la convertimos a **normal estándar** N(0,1):

```
Z = (X - μ) / σ
```

Si tu tiempo de compilación promedio es 5.2 segundos con σ = 1.1, y observas un kernel que toma 7.5 segundos:

```
Z = (7.5 - 5.2) / 1.1 = 2.09
```

Este kernel es 2.09 desviaciones estándar arriba de la media, lo que es bastante inusual (~2% de probabilidad).

```{code-cell} ipython3
import numpy as np
from scipy.stats import norm

# Parámetros de la distribución normal
mu = 5.2  # media
sigma = 1.1  # desviación estándar

# Rango de valores
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = norm.pdf(x, mu, sigma)

# Z-score del kernel específico
tiempo_kernel = 7.5
z_score = (tiempo_kernel - mu) / sigma
prob_mayor = 1 - norm.cdf(tiempo_kernel, mu, sigma)

print("Distribución Normal y Z-scores")
print("=" * 60)
print(f"Tiempo de compilación: N(μ={mu}, σ={sigma})")
print(f"Tiempo observado: {tiempo_kernel} segundos")
print(f"Z-score: {z_score:.2f}")
print(f"P(X > {tiempo_kernel}) = {prob_mayor:.4f} ({prob_mayor*100:.2f}%)")
print(f"\nInterpretación: Este kernel es {z_score:.2f} desviaciones estándar")
print(f"por encima de la media, lo cual es inusual.")
```

Visualmente, podemos observar la forma de la campana en la **Distribución Normal** y de manera relacionada existe la **Regla Empírica del 68-95-99.7%**:

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Visualización fundamental
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Distribución Normal con regiones
axes[0].plot(x, pdf, 'b-', linewidth=2, label='N(5.2, 1.1²)')
axes[0].fill_between(x, 0, pdf, alpha=0.3, color='blue')

# Marcar μ ± σ, μ ± 2σ, μ ± 3σ
for k in range(1, 4):
    axes[0].axvline(mu + k*sigma, color='red', linestyle='--', alpha=0.5)
    axes[0].axvline(mu - k*sigma, color='red', linestyle='--', alpha=0.5)

axes[0].axvline(mu, color='red', linewidth=2, label=f'μ = {mu}')
axes[0].axvline(tiempo_kernel, color='green', linewidth=2, linestyle=':',
                   label=f'Observado = {tiempo_kernel}s')

# Sombrear región extrema
x_extremo = x[x >= tiempo_kernel]
pdf_extremo = norm.pdf(x_extremo, mu, sigma)
axes[0].fill_between(x_extremo, 0, pdf_extremo, alpha=0.6, color='red',
                        label=f'P(X≥{tiempo_kernel}) = {prob_mayor:.4f}')

axes[0].set_xlabel('Tiempo de compilación (segundos)')
axes[0].set_ylabel('Densidad de Probabilidad')
axes[0].set_title('Distribución Normal del Tiempo de Compilación')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Regla empírica 68-95-99.7
regiones = ['μ±1σ\n(68%)', 'μ±2σ\n(95%)', 'μ±3σ\n(99.7%)']
porcentajes = [68, 95, 99.7]
colores = ['lightgreen', 'lightyellow', 'lightcoral']

bars = axes[1].bar(regiones, porcentajes, color=colores, alpha=0.7,
                      edgecolor='black', linewidth=2)
axes[1].set_ylabel('Porcentaje de datos')
axes[1].set_title('Regla Empírica: 68-95-99.7')
axes[1].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, porcentajes):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val}%', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()
```

Por su parte, cuando **estandarizamos los valores**, usando *z-scores*, transformamos nuestra normal a una distribución de medias de valor 0 y desviación de valor 1. Así podemos visualizar fácilmente dónde caen distintos puntos de datos para ver si ameritan atención.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3. Distribución Normal Estándar (Z-scores)
z_range = np.linspace(-4, 4, 1000)
pdf_standard = norm.pdf(z_range, 0, 1)

axes[0].plot(z_range, pdf_standard, 'b-', linewidth=2, label='N(0, 1)')
axes[0].fill_between(z_range, 0, pdf_standard, alpha=0.3, color='blue')

# Marcar el z-score del kernel
axes[0].axvline(z_score, color='red', linewidth=2, linestyle='--',
                   label=f'Z = {z_score:.2f}')

# Sombrear área extrema
z_extremo = z_range[z_range >= z_score]
pdf_z_extremo = norm.pdf(z_extremo, 0, 1)
axes[0].fill_between(z_extremo, 0, pdf_z_extremo, alpha=0.6, color='red')

axes[0].set_xlabel('Z-score')
axes[0].set_ylabel('Densidad de Probabilidad')
axes[0].set_title('Distribución Normal Estándar (Estandarizada)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Anotar regiones
axes[0].text(0, 0.2, '68%', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))
axes[0].text(0, 0.15, '95%', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# 4. Comparación de múltiples valores y sus Z-scores
tiempos_test = [3.0, 4.0, 5.2, 6.0, 7.5]
z_scores = [(t - mu) / sigma for t in tiempos_test]
probabilidades = [1 - norm.cdf(t, mu, sigma) for t in tiempos_test]

scatter = axes[1].scatter(tiempos_test, z_scores, s=200, c=probabilidades,
                   cmap='RdYlGn_r', edgecolor='black', linewidth=2, zorder=3)

for t, z, p in zip(tiempos_test, z_scores, probabilidades):
    axes[1].text(t, z + 0.15, f'{t}s\nZ={z:.2f}\nP={p:.3f}',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[1].axhline(0, color='gray', linestyle='-', linewidth=1)
axes[1].axhline(1.96, color='orange', linestyle='--', alpha=0.5, label='Z=±1.96 (95%)')
axes[1].axhline(-1.96, color='orange', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Tiempo de compilación (segundos)')
axes[1].set_ylabel('Z-score')
axes[1].set_title('Tiempos de Compilación y sus Z-scores')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.colorbar(scatter, ax=axes[1], label='P(X > valor)')
plt.tight_layout()
plt.show()

# Tabla de valores
print("\nTabla de Z-scores y Probabilidades:")
print("-" * 60)
print(f"{'Tiempo (s)':<12} {'Z-score':<12} {'P(X > tiempo)':<15} {'Interpretación'}")
print("-" * 60)
for t, z, p in zip(tiempos_test, z_scores, probabilidades):
    if abs(z) < 1:
        interp = "Normal"
    elif abs(z) < 2:
        interp = "Poco usual"
    else:
        interp = "Muy inusual"
    print(f"{t:<12.1f} {z:<12.2f} {p:<15.4f} {interp}")
```

## Teorema Central del Límite (TCL)

```{admonition} 💡 Intuición
:class: hint
Imagina que lanzas un dado (distribución uniforme, no normal). Si lo lanzas muchas veces y calculas el promedio, ese promedio se comportará como una distribución normal. ¡Esto funciona sin importar la forma de la distribución original! Es como magia estadística.
```

El **Teorema Central del Límite** es el corazón de la estadística práctica:

> Si tomas muestras de cualquier distribución (normal o no) y calculas la media muestral, esa media seguirá aproximadamente una distribución normal, siempre que la muestra sea suficientemente grande.

```
Si X₁, X₂, ..., Xₙ son independientes e idénticamente distribuidas (i.i.d.) con E[X] = μ y Var(X) = σ²:

Entonces: X̄ ~ N(μ, σ²/n)    aproximadamente para n grande
```

```{admonition} 🎯 Aplicación en ML
:class: important
En tu proyecto de evaluación de kernels GPU, esto te permite: ejecutar tu generador 100 veces y calcular el promedio de iteraciones. Incluso si cada ejecución individual sigue una distribución asimétrica, el **promedio de 100 ejecuciones seguirá aproximadamente una distribución normal**. Esto justifica el uso de pruebas t para comparar métodos.
```

### Implicación Práctica

Ejecutas tu generador 100 veces y calculas el promedio de iteraciones. Incluso si cada ejecución individual sigue una distribución rara o asimétrica, el **promedio de 100 ejecuciones seguirá aproximadamente una distribución normal**.

Esto es por qué el TCL es tan poderoso: nos permite usar estadística basada en distribuciones normales incluso cuando nuestros datos individuales no lo son.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform

# Demostración del Teorema Central del Límite
np.random.seed(42)

# Función para generar muestras y calcular medias
def demostrar_tcl(distribucion, nombre, params, n_muestras=[1, 5, 10, 30]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    num_experimentos = 10000

    for idx, n in enumerate(n_muestras):
        # Generar experimentos: cada uno toma n muestras y calcula la media
        medias = []
        for _ in range(num_experimentos):
            if distribucion == 'exponencial':
                muestra = expon.rvs(scale=params['scale'], size=n)
            elif distribucion == 'uniforme':
                muestra = uniform.rvs(loc=params['loc'], scale=params['scale'], size=n)
            elif distribucion == 'bimodal':
                # Mezcla de dos normales
                if np.random.rand() < 0.5:
                    muestra = np.random.normal(0, 1, n)
                else:
                    muestra = np.random.normal(5, 1, n)

            medias.append(np.mean(muestra))

        medias = np.array(medias)

        # Histograma de las medias
        axes[idx].hist(medias, bins=50, density=True, alpha=0.6,
                      color='lightblue', edgecolor='black', label='Medias muestrales')

        # Superponer normal teórica
        mu_teorica = np.mean(medias)
        sigma_teorica = np.std(medias)
        x = np.linspace(medias.min(), medias.max(), 100)
        pdf_normal = norm.pdf(x, mu_teorica, sigma_teorica)
        axes[idx].plot(x, pdf_normal, 'r-', linewidth=2,
                      label=f'N({mu_teorica:.2f}, {sigma_teorica:.2f}²)')

        axes[idx].axvline(mu_teorica, color='red', linestyle='--', linewidth=2)
        axes[idx].set_xlabel('Media muestral')
        axes[idx].set_ylabel('Densidad')
        axes[idx].set_title(f'{nombre}: Distribución de Medias (n={n})')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    plt.suptitle(f'Teorema Central del Límite - Distribución Original: {nombre}',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

print("Demostración del Teorema Central del Límite")
print("=" * 60)
print("Veremos cómo las medias muestrales se vuelven normales")
print("independientemente de la distribución original.\n")

# 1. Distribución Exponencial (muy asimétrica)
print("1. Distribución Exponencial (muy asimétrica)")
demostrar_tcl('exponencial', 'Exponencial', {'scale': 2})

# 2. Distribución Uniforme
print("\n2. Distribución Uniforme")
demostrar_tcl('uniforme', 'Uniforme', {'loc': 0, 'scale': 10})
```

Como vemos, aumentar el tamaño de la muestra nos acerca cada vez más a una distribución normal, que es predecible, simétrica y bien conocida. Esto es extremadamente útil porque nos permite usar estadística basada en normales (como pruebas Z o T) **incluso cuando los datos muestrales no lo son**. Veámoslo con nuestro caso práctico de los Kernels GPU:

```{code-cell} ipython3
# Ejemplo práctico: iteraciones del generador
print("\nEjemplo Práctico: Iteraciones del Generador de Kernels")
print("-" * 60)

# Simulamos que cada ejecución individual sigue una distribución asimétrica
# (muchas convergen rápido, pocas toman mucho tiempo)
np.random.seed(42)
num_ejecuciones = 100
tamanios_muestra = [1, 5, 10, 30, 100]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, n in enumerate(tamanios_muestra):
    medias_experimento = []

    for _ in range(1000):
        # Distribución asimétrica (exponencial)
        iteraciones = np.random.exponential(scale=40, size=n)
        medias_experimento.append(np.mean(iteraciones))

    medias_experimento = np.array(medias_experimento)

    axes[idx].hist(medias_experimento, bins=40, density=True, alpha=0.6,
                   color='steelblue', edgecolor='black')

    # Ajustar normal
    mu = np.mean(medias_experimento)
    sigma = np.std(medias_experimento)
    x = np.linspace(medias_experimento.min(), medias_experimento.max(), 100)
    axes[idx].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                   label=f'N({mu:.1f}, {sigma:.1f}²)')

    axes[idx].set_xlabel('Media de iteraciones')
    axes[idx].set_ylabel('Densidad')
    axes[idx].set_title(f'n = {n} ejecuciones promediadas')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

    print(f"n={n:3d}: μ={mu:.2f}, σ={sigma:.2f}, σ_teórico={40/np.sqrt(n):.2f}")

# Mostrar distribución original
axes[5].hist(np.random.exponential(scale=40, size=10000), bins=50,
             density=True, alpha=0.6, color='red', edgecolor='black')
axes[5].set_xlabel('Iteraciones (individual)')
axes[5].set_ylabel('Densidad')
axes[5].set_title('Distribución Original (Exponencial, asimétrica)')
axes[5].grid(alpha=0.3)

plt.suptitle('TCL Aplicado: Promedios de Iteraciones se Vuelven Normales',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nConclusión: Incluso si cada ejecución individual sigue una")
print("distribución asimétrica, el PROMEDIO de múltiples ejecuciones")
print("sigue aproximadamente una distribución normal.")
```

## Estadística Descriptiva

Ahora aprendemos a **resumir datos** con números simples.

### Media (Promedio)

```
x̄ = (Σ xᵢ) / n
```

La suma de todos los valores dividida por cuántos hay. Fácil de calcular, pero sensible a valores atípicos.

Ejemplo: Tiempos de compilación (segundos): 2.1, 2.3, 2.2, 2.0, 15.5
```
x̄ = (2.1 + 2.3 + 2.2 + 2.0 + 15.5) / 5 = 24.1 / 5 = 4.82
```

Nota que ese 15.5 tira el promedio hacia arriba. La media es 4.82, pero la mayoría de valores están alrededor de 2.

### Mediana

El valor del medio cuando los datos están ordenados. No se ve afectada por valores atípicos.

```
Datos ordenados: 2.0, 2.1, 2.2, 2.3, 15.5
Mediana = 2.2 (elemento del medio)
```

Para un número par de elementos, promedias los dos del medio.

Comparación:
- **Media = 4.82** (tirada por el atípico)
- **Mediana = 2.2** (más representativa del "típico")

### Desviación Estándar y Varianza

Miden cuán dispersos están tus datos.

```
Varianza: s² = Σ(xᵢ - x̄)² / (n-1)    [muestral]

Desviación Estándar: s = √(s²)
```

Para nuestros datos:
```
Desviaciones: (2.1-4.82)², (2.3-4.82)², ..., (15.5-4.82)²
            = 7.39, 6.33, 6.76, 8.07, 114.49

s² = 143.04 / 4 = 35.76
s = 5.98 segundos
```

Un σ alto indica datos muy dispersos; σ bajo indica datos agrupados.

### Percentiles e Intervalo Intercuartílico (IQR)

El **percentil p** es el valor bajo el cual está el p% de tus datos.

Cuartiles importantes:
- Q1 (25º percentil): el 25% de datos están por debajo
- Q2 (50º percentil): la mediana
- Q3 (75º percentil): el 75% de datos están por debajo

**IQR = Q3 - Q1** es el rango del 50% central de datos.

Para nuestros 5 valores ordenados (2.0, 2.1, 2.2, 2.3, 15.5):
```
Q1 = 2.05 (promedio de 2.0 y 2.1)
Q2 = 2.2
Q3 = 2.25 (promedio de 2.2 y 2.3)
IQR = 2.25 - 2.05 = 0.20
```

El IQR es robusto a valores atípicos: no importa cuánto sea 15.5, el IQR se mantiene pequeño.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Datos de ejemplo: tiempos de compilación con un atípico
datos = np.array([2.0, 2.1, 2.2, 2.3, 15.5])

# Estadísticas descriptivas
media = np.mean(datos)
mediana = np.median(datos)
desv_std = np.std(datos, ddof=1)
varianza = np.var(datos, ddof=1)

# Cuartiles
Q1 = np.percentile(datos, 25)
Q2 = np.percentile(datos, 50)  # mediana
Q3 = np.percentile(datos, 75)
IQR = Q3 - Q1

print("Estadística Descriptiva")
print("=" * 60)
print(f"Datos: {datos}")
print(f"\nMediada de Tendencia Central:")
print(f"  Media    = {media:.2f} segundos")
print(f"  Mediana  = {mediana:.2f} segundos")
print(f"\nMediada de Dispersión:")
print(f"  Varianza = {varianza:.2f}")
print(f"  Desv Std = {desv_std:.2f} segundos")
print(f"\nCuartiles:")
print(f"  Q1 (25%) = {Q1:.2f}")
print(f"  Q2 (50%) = {Q2:.2f} (mediana)")
print(f"  Q3 (75%) = {Q3:.2f}")
print(f"  IQR      = {IQR:.2f}")
```

La forma más intuitiva de observar estas medidas, los cuartiles, la mediana y los datos atípicos, es mediante los Diagramas de Caja (*Box Plots*) contrastados con histogramas:

```{code-cell} ipython3
# Visualización general de datos y sus atípicos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Histograma
axes[0].hist(datos, bins=5, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=2)
axes[0].axvline(media, color='red', linewidth=2, linestyle='--',
                   label=f'Media = {media:.2f}')
axes[0].axvline(mediana, color='green', linewidth=2, linestyle=':',
                   label=f'Mediana = {mediana:.2f}')
axes[0].set_xlabel('Tiempo de compilación (s)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Tiempos de Compilación')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# 2. Box Plot
bp = axes[1].boxplot(datos, vert=True, patch_artist=True,
                        notch=True, showmeans=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(marker='D', markerfacecolor='green',
                                      markersize=8))

axes[1].set_ylabel('Tiempo de compilación (s)')
axes[1].set_title('Box Plot: Visualización de Cuartiles y Atípicos')
axes[1].grid(axis='y', alpha=0.3)

# Anotar componentes del box plot
axes[1].text(1.15, Q1, f'Q1={Q1:.2f}', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
axes[1].text(1.15, Q2, f'Q2={Q2:.2f}', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
axes[1].text(1.15, Q3, f'Q3={Q3:.2f}', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

print("\nInterpretación:")
print(f"- La media ({media:.2f}) está fuertemente afectada por el atípico (15.5)")
print(f"- La mediana ({mediana:.2f}) representa mejor el valor 'típico'")
print(f"- El IQR ({IQR:.2f}) es robusto y muestra baja dispersión del 50% central")
```

Más allá de analizar un único conjunto de datos con un atípico, vale la pena contrastar la forma en que los atípicos y la varianza afectan las medidas en general. Haremos esto comparando un conjunto de datos muy disperso frente a otros más unificados.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3. Comparación Media vs Mediana en diferentes datasets
datasets = {
    'Sin atípicos': np.array([2.0, 2.1, 2.2, 2.3, 2.4]),
    'Con atípico': datos,
    'Muy sesgado': np.array([1, 2, 2, 3, 3, 3, 4, 4, 50])
}

x_pos = np.arange(len(datasets))
medias_list = [np.mean(d) for d in datasets.values()]
medianas_list = [np.median(d) for d in datasets.values()]

width = 0.35
axes[0].bar(x_pos - width/2, medias_list, width, label='Media',
               color='red', alpha=0.7, edgecolor='black')
axes[0].bar(x_pos + width/2, medianas_list, width, label='Mediana',
               color='green', alpha=0.7, edgecolor='black')

axes[0].set_ylabel('Valor')
axes[0].set_title('Comparación: Media vs Mediana\n(Robustez a Atípicos)')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(datasets.keys())
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Anotar valores
for i, (m, med) in enumerate(zip(medias_list, medianas_list)):
    axes[0].text(i - width/2, m + 0.5, f'{m:.1f}', ha='center', fontsize=9)
    axes[0].text(i + width/2, med + 0.5, f'{med:.1f}', ha='center', fontsize=9)

# 4. Visualización de dispersión (varianza)
np.random.seed(42)
consistente = np.random.normal(5, 0.5, 100)  # baja varianza
erratico = np.random.normal(5, 2.0, 100)     # alta varianza

axes[1].hist(consistente, bins=20, alpha=0.6, color='green',
                edgecolor='black', label=f'Consistente (σ={np.std(consistente):.2f})')
axes[1].hist(erratico, bins=20, alpha=0.6, color='red',
                edgecolor='black', label=f'Errático (σ={np.std(erratico):.2f})')

axes[1].axvline(np.mean(consistente), color='green', linewidth=2, linestyle='--')
axes[1].axvline(np.mean(erratico), color='red', linewidth=2, linestyle='--')

axes[1].set_xlabel('Valor')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Comparación de Varianza\n(Misma media, diferente dispersión)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Detección de Valores Atípicos (Outliers)

Una regla común:
```
Límite inferior = Q1 - 1.5 × IQR = 2.05 - 1.5(0.20) = 1.85
Límite superior = Q3 + 1.5 × IQR = 2.25 + 1.5(0.20) = 2.55
```

Cualquier valor fuera de [1.85, 2.55] es un atípico. En nuestro caso, 15.5 es definitivamente un atípico.

## Diagramas de Caja (Box Plots)

Un visualizador útil que muestra mediana, cuartiles y atípicos:

```
     *              ← atípico (15.5)

1.0  |----[Q1--Q2--Q3]----| 2.3

     Q1=2.05  Q2=2.2  Q3=2.25
```

Es especialmente útil para **comparar distribuciones**. En tu proyecto, podrías tener un box plot lado a lado: baseline vs. con restricciones.

## Comparación: Media vs. Mediana

¿Cuándo usas cada una?

| Situación | Usa |
|-----------|-----|
| Datos simétricos, sin atípicos | Media |
| Datos con algunos atípicos | Mediana |
| Datos muy asimétricos | Mediana |
| Necesitas para cálculos posteriores | Media |
| Quieres robusto y simple | Mediana |

En nuestro ejemplo de tiempos de compilación:
- **Media = 4.82**: engañosa, no representa el típico
- **Mediana = 2.2**: mejor representa la experiencia típica

**En contexto ML:** Usa mediana para tiempos de GPU (outliers por thermal throttling), throughput con warm-up variable. Usa media para métricas de training (loss, accuracy) que son estables.

## Resumen de Distribuciones

```{admonition} Resumen
:class: important
**Checklist para elegir distribución:**
- [ ] **Bernoulli**: ¿Una sola prueba con éxito/fracaso?
- [ ] **Binomial**: ¿Cuento éxitos en n intentos independientes?
- [ ] **Poisson**: ¿Cuento eventos raros en intervalo fijo?
- [ ] **Normal**: ¿Datos continuos y simétricos? ¿Promedio de muchas mediciones?

**Para aplicar correctamente:**
- [ ] Verificar que los supuestos se cumplan (independencia, parámetros constantes)
- [ ] Usar TCL cuando n ≥ 30 para justificar normalidad de promedios
- [ ] Reportar tanto media como mediana cuando hay outliers
```

| Distribución | Caso de uso | Parámetros | Media | Varianza |
|--------------|-------------|-----------|-------|----------|
| Bernoulli | Éxito/fracaso | p | p | p(1-p) |
| Binomial | n ensayos Bernoulli | n, p | np | np(1-p) |
| Poisson | Conteos de eventos raros | λ | λ | λ |
| Normal | Datos continuos "típicos" | μ, σ | μ | σ² |

## Ejercicios y Reflexión

```{admonition} ✅ Verifica tu comprensión
:class: note
1. **Distribuciones**: ¿Cuándo usarías Binomial vs Poisson para contar eventos?
2. **TCL**: Si tus datos individuales son muy asimétricos pero n=100, ¿puedes usar pruebas t? ¿Por qué?
3. **Media vs Mediana**: Si tienes el outlier 90 en [42, 45, 43, 41, 90, 44, 43], ¿cuál reportarías?
4. **Z-scores**: Si z = 2.5, ¿qué porcentaje de datos está más extremo que este valor?
```

### Ejercicio 1: Identifica la Distribución
Para cada escenario en tu proyecto, identifica qué distribución es apropiada:
1. Número de kernels válidos en 30 intentos
2. Tiempo de compilación de un kernel individual
3. Número de fallos de memoria en 1000 ejecuciones
4. Tasa de error promedio en 50 ejecuciones

### Ejercicio 2: Cálculos Binomiales
Si p = 0.85 (probabilidad de compilación) y n = 20:
- ¿Cuál es E[X]?
- ¿Cuál es SD[X]?
- ¿Aproximadamente qué rango esperas con 95% confianza?

### Ejercicio 3: Z-scores
Tiempos de compilación: μ = 3.2s, σ = 0.8s
- Un kernel toma 5.6s. ¿Cuál es su z-score?
- ¿Qué tan inusual es esto?
- ¿Un kernel de 2.0s es atípico?

### Ejercicio 4: Estadística Descriptiva
Datos de iteraciones para baseline: 42, 45, 43, 41, 90, 44, 43
Datos de iteraciones para restricciones: 38, 39, 37, 40, 38, 39, 36
- Calcula media, mediana, SD para ambos
- ¿Hay atípicos?
- ¿Cuál método es más consistente?

### Reflexión
1. **TCL en tu proyecto**: Cuando ejecutas tu algoritmo 100 veces, ¿por qué el TCL te permite asumir normalidad en el promedio?
2. **Elegir entre media y mediana**: En tu conjunto de datos de iteraciones, ¿cuál es más informativo? ¿Por qué?
3. **Asimetría**: Si tus datos de validez de kernels tienen q = 0.85, ¿esperas una distribución simétrica? ¿Cómo lo afecta esto?

---

**Próxima semana**: Aprenderemos cómo usar estas distribuciones para hacer pruebas de hipótesis sobre si nuestro método es realmente mejor que el baseline.

---

## Referencias

- Rice, J. (2006). Mathematical Statistics and Data Analysis (3rd ed.). Cengage Learning.
- Casella, G. & Berger, R. (2002). Statistical Inference (2nd ed.). Cengage Learning.
- Wasserman, L. (2004). [All of Statistics](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
