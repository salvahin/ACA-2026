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

# Pruebas de Hipótesis

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/03-pruebas-hipotesis.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton
print('Dependencies installed!')
```
## Semana 3 - Estadística para Generación de Kernels GPU

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Formular hipótesis nulas y alternativas correctamente
- Interpretar p-valores y niveles de significancia
- Calcular y reportar pruebas t (una muestra, dos muestras, pareadas)
- Distinguir entre errores Tipo I y Tipo II
- Tomar decisiones estadísticas fundamentadas sobre tus resultados
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[StatQuest: Hypothesis Testing](https://www.youtube.com/watch?v=0oc49DyA3hU)** - Josh Starmer explica paso a paso cómo funcionan las pruebas de hipótesis, p-valores, y cómo interpretar los resultados correctamente.
```

Finalmente llegamos a la pregunta central de tu investigación: **¿Es mi método realmente mejor, o fue solo suerte?** Las pruebas de hipótesis son el marco que usamos para responder esto de manera rigurosa.

## Estructura de una Prueba de Hipótesis

Toda prueba de hipótesis tiene dos competidores:

![**Figura 1:** Diagrama de flujo para pruebas de hipótesis estadísticas.](diagrams/hypothesis_testing.png)

***Figura 1:** Diagrama de flujo para pruebas de hipótesis estadísticas.*


1. **Hipótesis Nula (H₀)**: La afirmación "aburrida" que asumimos verdadera. Típicamente es "no hay diferencia".
2. **Hipótesis Alternativa (H₁)**: Lo que buscas demostrar. Típicamente es "hay una diferencia".

### Ejemplos en tu Proyecto

```
H₀: La tasa de compilación con restricciones = tasa sin restricciones (p₁ = p₂)
H₁: La tasa de compilación con restricciones ≠ tasa sin restricciones (p₁ ≠ p₂)
```

O si tienes razones para esperar una dirección:

```
H₀: iteraciones con restricciones ≥ iteraciones sin restricciones
H₁: iteraciones con restricciones < iteraciones sin restricciones
```

## Errores Tipo I y Tipo II

Al decidir si rechazamos H₀, podemos cometer dos tipos de errores:

:::{figure} diagrams/type_errors_matrix.png
:name: fig-type-errors-matrix
:alt: Matriz de confusión mostrando errores Tipo I y Tipo II
:align: center
:width: 80%

**Figura 5:** Matriz de errores en pruebas de hipótesis: Tipo I (falso positivo) y Tipo II (falso negativo).
:::

### Error Tipo I (False Positive)

Rechazamos H₀ cuando en realidad es verdadera. Declaramos que nuestro método es mejor cuando realmente no lo es.

**α (alfa)** = P(Error Tipo I) = P(rechazar H₀ | H₀ verdadera)

Típicamente establecemos α = 0.05, significando que aceptamos una tasa de falsos positivos del 5%.

### Error Tipo II (False Negative)

No rechazamos H₀ cuando en realidad es falsa. Perdemos detectar una mejora real.

**β (beta)** = P(Error Tipo II) = P(no rechazar H₀ | H₀ falsa)

Típicamente buscamos β = 0.20, lo que significa:
- **Poder = 1 - β = 0.80**: Tenemos 80% de probabilidad de detectar un efecto real

### El Balance

Hay un trade-off entre α y β. Si reduces α (más conservador), aumentas β (menos poder). Por eso el diseño experimental cuidadoso es crucial.

## P-valores y Niveles de Significancia

```{admonition} 🎮 Simulador Interactivo de P-valores
:class: tip

Explora qué significa realmente un p-valor:

<iframe src="https://rpsychologist.com/pvalue/" width="100%" height="600px" style="border:1px solid #ddd; border-radius:8px;"></iframe>

*Desarrollado por Kristoffer Magnusson. Ajusta los parámetros y observa la distribución.*
```

```{admonition} 💡 Intuición
:class: hint
Imagina que alguien afirma tener una moneda justa. Lanzas la moneda 10 veces y obtienes 10 caras. ¿Qué tan probable es esto si la moneda realmente es justa? Muy improbable (p = 0.001). El p-valor cuantifica exactamente esto: "¿Qué tan sorprendentes son mis datos si H₀ fuera cierta?"
```

El **p-valor** es probablemente el concepto más mal entendido en estadística.

:::{figure} diagrams/pvalue_regions.png
:name: fig-pvalue-regions
:alt: Visualización de regiones de p-valor y regiones críticas en una distribución de prueba
:align: center
:width: 90%

**Figura 4:** Regiones de p-valor y zonas críticas para la interpretación de pruebas de hipótesis.
:::

> El **p-valor** es la probabilidad de observar datos tan o más extremos que los que observaste, SI H₀ fuera verdadera.

**NO es** "la probabilidad de que H₀ sea verdadera". Es lo opuesto: asume H₀ y pregunta "¿qué tan sorprendentes son mis datos?"

```{admonition} 📊 Cómo interpretar
:class: note
**Si p < 0.05**, entonces:
- Rechazas H₀ (hay evidencia de diferencia)
- Hay menos de 5% de probabilidad de ver estos datos por azar si H₀ fuera cierta
- Reportas: "diferencia estadísticamente significativa (p = 0.03)"

**Si p ≥ 0.05**, entonces:
- No rechazas H₀ (no hay evidencia suficiente)
- Los datos son razonablemente compatibles con H₀
- Reportas: "no se encontró diferencia significativa (p = 0.12)"

**IMPORTANTE**: "No significante" NO significa "no hay efecto", solo que no detectamos uno.
```

### Interpretación Correcta

Si p = 0.03:
- "Si no hubiera diferencia real entre métodos, solo hay 3% de probabilidad de ver una diferencia tan grande (por casualidad)."
- Esto sugiere que probablemente hay una diferencia real.

Si p = 0.50:
- "Si no hubiera diferencia real, hay 50% de probabilidad de ver esto o algo más extremo."
- Esto es muy plausible bajo H₀. No hay evidencia de diferencia.

### Nivel de Significancia (α)

Establecemos una **línea de corte**: si p < α, rechazamos H₀.

```
Típicamente α = 0.05

Si p < 0.05: "Significante al nivel 0.05"
Si p < 0.01: "Significante al nivel 0.01" (más fuerte)
Si p ≥ 0.05: "No significante"
```

**Advertencia**: p = 0.051 es tan evidencia contra H₀ como p = 0.049 en un sentido real, pero uno es "significante" y el otro no según la regla de corte arbitraria.

## Prueba t de Una Muestra

Comparas el promedio de tu muestra contra un valor conocido. Por ejemplo, ¿es el tiempo promedio de compilación diferente de 5 segundos?

```
H₀: μ = 5
H₁: μ ≠ 5

Estadístico de prueba: t = (x̄ - μ₀) / (s / √n)

Donde:
- x̄ = promedio muestral
- μ₀ = valor hipotético (5 segundos)
- s = desviación estándar muestral
- n = tamaño muestral
```

### Ejemplo Práctico

Ejecutas 25 veces y obtienes:
- x̄ = 5.8 segundos
- s = 1.2 segundos
- n = 25

```
t = (5.8 - 5.0) / (1.2 / √25)
  = 0.8 / (1.2 / 5)
  = 0.8 / 0.24
  = 3.33
```

Con df = n - 1 = 24 grados de libertad, este t = 3.33 corresponde a p ≈ 0.003.

**Conclusión**: p < 0.05, rechazamos H₀. El tiempo promedio es significativamente diferente de 5 segundos.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Datos del experimento
np.random.seed(42)
n = 25
x_bar = 5.8
s = 1.2
mu_0 = 5.0  # valor hipotético

# Calcular t estadístico
t_stat = (x_bar - mu_0) / (s / np.sqrt(n))
df = n - 1

# Calcular p-valor (prueba de dos colas)
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print("Prueba t de Una Muestra")
print("=" * 60)
print(f"H₀: μ = {mu_0} segundos")
print(f"H₁: μ ≠ {mu_0} segundos (dos colas)")
print(f"\nDatos:")
print(f"  n = {n}")
print(f"  x̄ = {x_bar} segundos")
print(f"  s = {s} segundos")
print(f"\nEstadístico de prueba:")
print(f"  t = (x̄ - μ₀) / (s / √n)")
print(f"  t = ({x_bar} - {mu_0}) / ({s} / √{n})")
print(f"  t = {t_stat:.3f}")
print(f"\nGrados de libertad: df = {df}")
print(f"p-valor = {p_value:.4f}")
print(f"\nConclusión (α = 0.05):")
if p_value < 0.05:
    print(f"  p < 0.05 → Rechazamos H₀")
    print(f"  El tiempo promedio ES significativamente diferente de {mu_0}s")
else:
    print(f"  p ≥ 0.05 → No rechazamos H₀")
    print(f"  No hay evidencia suficiente de diferencia")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribución t con región crítica
x = np.linspace(-4, 4, 1000)
t_pdf = stats.t.pdf(x, df)

axes[0, 0].plot(x, t_pdf, 'b-', linewidth=2, label=f't({df})')
axes[0, 0].fill_between(x, 0, t_pdf, alpha=0.3, color='blue')

# Regiones críticas (α = 0.05, dos colas)
t_critical = stats.t.ppf(0.975, df)
x_left = x[x <= -t_critical]
x_right = x[x >= t_critical]
axes[0, 0].fill_between(x_left, 0, stats.t.pdf(x_left, df),
                        alpha=0.6, color='red', label=f'Región crítica (α=0.05)')
axes[0, 0].fill_between(x_right, 0, stats.t.pdf(x_right, df),
                        alpha=0.6, color='red')

# Marcar t observado
axes[0, 0].axvline(t_stat, color='green', linewidth=3, linestyle='--',
                   label=f't observado = {t_stat:.2f}')
axes[0, 0].axvline(-t_critical, color='red', linewidth=1, linestyle=':',
                   label=f't crítico = ±{t_critical:.2f}')
axes[0, 0].axvline(t_critical, color='red', linewidth=1, linestyle=':')

axes[0, 0].set_xlabel('Estadístico t')
axes[0, 0].set_ylabel('Densidad de Probabilidad')
axes[0, 0].set_title('Distribución t y Regiones Críticas')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. P-valor visualizado
axes[0, 1].plot(x, t_pdf, 'b-', linewidth=2)
x_pval_left = x[x <= -abs(t_stat)]
x_pval_right = x[x >= abs(t_stat)]
axes[0, 1].fill_between(x_pval_left, 0, stats.t.pdf(x_pval_left, df),
                        alpha=0.7, color='orange',
                        label=f'p-valor = {p_value:.4f}')
axes[0, 1].fill_between(x_pval_right, 0, stats.t.pdf(x_pval_right, df),
                        alpha=0.7, color='orange')

axes[0, 1].axvline(t_stat, color='green', linewidth=3, linestyle='--',
                   label=f't = {t_stat:.2f}')
axes[0, 1].axvline(-t_stat, color='green', linewidth=3, linestyle='--')

axes[0, 1].set_xlabel('Estadístico t')
axes[0, 1].set_ylabel('Densidad de Probabilidad')
axes[0, 1].set_title('Visualización del P-valor')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Simulación de datos que podrían haber generado este resultado
np.random.seed(42)
datos_simulados = np.random.normal(x_bar, s, n)

axes[1, 0].hist(datos_simulados, bins=10, color='steelblue', alpha=0.7,
                edgecolor='black', density=True)
axes[1, 0].axvline(x_bar, color='red', linewidth=2, linestyle='--',
                   label=f'x̄ = {x_bar}')
axes[1, 0].axvline(mu_0, color='green', linewidth=2, linestyle=':',
                   label=f'μ₀ = {mu_0}')

# Superponer distribución teórica
x_range = np.linspace(min(datos_simulados), max(datos_simulados), 100)
axes[1, 0].plot(x_range, stats.norm.pdf(x_range, x_bar, s),
                'r-', linewidth=2, alpha=0.7, label='N(x̄, s²)')

axes[1, 0].set_xlabel('Tiempo de compilación (s)')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].set_title('Datos Simulados y Distribución')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Intervalo de confianza 95%
se = s / np.sqrt(n)
ci_lower = x_bar - t_critical * se
ci_upper = x_bar + t_critical * se

axes[1, 1].errorbar([1], [x_bar], yerr=[[x_bar - ci_lower], [ci_upper - x_bar]],
                    fmt='o', markersize=12, capsize=10, capthick=2,
                    color='blue', label=f'x̄ = {x_bar}')
axes[1, 1].axhline(mu_0, color='green', linewidth=2, linestyle='--',
                   label=f'μ₀ = {mu_0}')
axes[1, 1].axhline(ci_lower, color='red', linewidth=1, linestyle=':',
                   alpha=0.5)
axes[1, 1].axhline(ci_upper, color='red', linewidth=1, linestyle=':',
                   alpha=0.5)

axes[1, 1].fill_between([0.5, 1.5], ci_lower, ci_upper,
                        alpha=0.3, color='blue',
                        label=f'IC 95%: [{ci_lower:.2f}, {ci_upper:.2f}]')

axes[1, 1].set_xlim(0.5, 1.5)
axes[1, 1].set_ylabel('Tiempo de compilación (s)')
axes[1, 1].set_title('Intervalo de Confianza 95%')
axes[1, 1].set_xticks([1])
axes[1, 1].set_xticklabels(['Muestra'])
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nIntervalo de Confianza 95%: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Nota: μ₀ = {mu_0} NO está dentro del IC, consistente con rechazar H₀")
```

## Prueba t de Dos Muestras: Muestras Independientes

Comparas promedios entre dos grupos diferentes (baseline vs. tu método).

```
H₀: μ₁ = μ₂ (no hay diferencia)
H₁: μ₁ ≠ μ₂ (hay diferencia)

Estadístico: t = (x̄₁ - x̄₂) / SE(x̄₁ - x̄₂)

Donde SE = error estándar de la diferencia
```

### Asunciones Importantes

- **Normalidad**: Datos aproximadamente normales (TCL hace esto menos crítico con n > 30)
- **Igualdad de varianzas**: Las dos muestras tienen varianzas similares (Welch's t si no)
- **Independencia**: Las observaciones no dependen unas de otras

### Ejemplo: Baseline vs. Con Restricciones

Baseline (30 ejecuciones): x̄₁ = 4.2s, s₁ = 0.9s
Restricciones (30 ejecuciones): x̄₂ = 3.8s, s₂ = 0.8s

```
t = (4.2 - 3.8) / SE ≈ 1.85

Con df ≈ 58, esto da p ≈ 0.068
```

Con α = 0.05, esto **no es estadísticamente significante**. No podemos rechazar H₀.

Pero nota: hay una diferencia numérica de 0.4 segundos. Podría ser prácticamente importante aunque no sea estadísticamente significante. Hablaremos de esto cuando cubramos tamaño del efecto.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Datos de dos muestras independientes
np.random.seed(42)
n1 = n2 = 30
x_bar1 = 4.2  # Baseline
s1 = 0.9
x_bar2 = 3.8  # Restricciones
s2 = 0.8

# Generar datos simulados
datos_baseline = np.random.normal(x_bar1, s1, n1)
datos_restricciones = np.random.normal(x_bar2, s2, n2)

# Prueba t de dos muestras independientes
t_stat, p_value = stats.ttest_ind(datos_baseline, datos_restricciones)

# Cálculo manual del error estándar
sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))  # pooled std
se = sp * np.sqrt(1/n1 + 1/n2)
df = n1 + n2 - 2

# Cohen's d (tamaño del efecto)
cohens_d = (x_bar1 - x_bar2) / sp

# Intervalo de confianza de la diferencia
t_critical = stats.t.ppf(0.975, df)
ci_lower = (x_bar1 - x_bar2) - t_critical * se
ci_upper = (x_bar1 - x_bar2) + t_critical * se

print("Prueba t de Dos Muestras Independientes")
print("=" * 60)
print(f"H₀: μ₁ = μ₂ (no hay diferencia entre métodos)")
print(f"H₁: μ₁ ≠ μ₂ (hay diferencia)")
print(f"\nGrupo 1 (Baseline):      n={n1}, x̄={x_bar1:.2f}, s={s1:.2f}")
print(f"Grupo 2 (Restricciones): n={n2}, x̄={x_bar2:.2f}, s={s2:.2f}")
print(f"\nDiferencia observada: {x_bar1 - x_bar2:.2f} segundos")
print(f"Error estándar: {se:.3f}")
print(f"Estadístico t: {t_stat:.3f}")
print(f"Grados de libertad: {df}")
print(f"p-valor: {p_value:.4f}")
print(f"\nTamaño del efecto (Cohen's d): {cohens_d:.3f}")
print(f"IC 95% de la diferencia: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"\nConclusión (α = 0.05):")
if p_value < 0.05:
    print(f"  p < 0.05 → Rechazamos H₀")
    print(f"  Hay diferencia significativa entre métodos")
else:
    print(f"  p ≥ 0.05 → No rechazamos H₀")
    print(f"  No hay evidencia suficiente de diferencia")
    print(f"  PERO la diferencia práctica ({x_bar1-x_bar2:.2f}s) puede ser importante")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribuciones de ambos grupos
axes[0, 0].hist(datos_baseline, bins=15, alpha=0.6, color='red',
                edgecolor='black', label=f'Baseline (x̄={np.mean(datos_baseline):.2f})',
                density=True)
axes[0, 0].hist(datos_restricciones, bins=15, alpha=0.6, color='blue',
                edgecolor='black', label=f'Restricciones (x̄={np.mean(datos_restricciones):.2f})',
                density=True)

axes[0, 0].axvline(x_bar1, color='red', linewidth=2, linestyle='--')
axes[0, 0].axvline(x_bar2, color='blue', linewidth=2, linestyle='--')

axes[0, 0].set_xlabel('Tiempo de compilación (s)')
axes[0, 0].set_ylabel('Densidad')
axes[0, 0].set_title('Distribuciones de Ambos Grupos')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Box plots comparativos
bp = axes[0, 1].boxplot([datos_baseline, datos_restricciones],
                        labels=['Baseline', 'Restricciones'],
                        patch_artist=True, notch=True, showmeans=True)

bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('blue')
for box in bp['boxes']:
    box.set_alpha(0.6)

axes[0, 1].set_ylabel('Tiempo de compilación (s)')
axes[0, 1].set_title('Comparación de Distribuciones (Box Plots)')
axes[0, 1].grid(axis='y', alpha=0.3)

# Anotar diferencia
axes[0, 1].plot([1, 2], [x_bar1, x_bar2], 'go-', linewidth=2, markersize=10,
                label='Medias')
axes[0, 1].text(1.5, (x_bar1 + x_bar2)/2,
                f'Δ = {x_bar1-x_bar2:.2f}s\np = {p_value:.3f}',
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 3. Distribución t y región crítica
x = np.linspace(-4, 4, 1000)
t_pdf = stats.t.pdf(x, df)

axes[1, 0].plot(x, t_pdf, 'b-', linewidth=2, label=f't({df})')
axes[1, 0].fill_between(x, 0, t_pdf, alpha=0.3, color='blue')

# Regiones críticas
x_left = x[x <= -t_critical]
x_right = x[x >= t_critical]
axes[1, 0].fill_between(x_left, 0, stats.t.pdf(x_left, df),
                        alpha=0.6, color='red', label='Región crítica (α=0.05)')
axes[1, 0].fill_between(x_right, 0, stats.t.pdf(x_right, df),
                        alpha=0.6, color='red')

# Marcar t observado
axes[1, 0].axvline(t_stat, color='green', linewidth=3, linestyle='--',
                   label=f't obs = {t_stat:.2f}')
axes[1, 0].axvline(-t_critical, color='red', linewidth=1, linestyle=':')
axes[1, 0].axvline(t_critical, color='red', linewidth=1, linestyle=':',
                   label=f't crítico = ±{t_critical:.2f}')

axes[1, 0].set_xlabel('Estadístico t')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].set_title('Distribución t y Estadístico Observado')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Intervalo de confianza de la diferencia
axes[1, 1].errorbar([1], [x_bar1 - x_bar2],
                    yerr=[[x_bar1-x_bar2-ci_lower], [ci_upper-(x_bar1-x_bar2)]],
                    fmt='o', markersize=12, capsize=10, capthick=2,
                    color='purple', label='Diferencia observada')

axes[1, 1].axhline(0, color='red', linewidth=2, linestyle='--',
                   label='H₀: diferencia = 0')
axes[1, 1].fill_between([0.5, 1.5], ci_lower, ci_upper,
                        alpha=0.3, color='purple',
                        label=f'IC 95%: [{ci_lower:.3f}, {ci_upper:.3f}]')

axes[1, 1].set_xlim(0.5, 1.5)
axes[1, 1].set_ylabel('Diferencia en tiempo (s)')
axes[1, 1].set_title('Intervalo de Confianza de la Diferencia')
axes[1, 1].set_xticks([1])
axes[1, 1].set_xticklabels(['Baseline - Restricciones'])
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

if ci_lower < 0 < ci_upper:
    axes[1, 1].text(1, ci_upper + 0.05,
                    'IC incluye 0\n→ No significante',
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow'))

plt.tight_layout()
plt.show()

print("\nInterpretación del tamaño del efecto (Cohen's d):")
if abs(cohens_d) < 0.2:
    print("  d < 0.2: Efecto trivial/despreciable")
elif abs(cohens_d) < 0.5:
    print("  0.2 ≤ d < 0.5: Efecto pequeño")
elif abs(cohens_d) < 0.8:
    print("  0.5 ≤ d < 0.8: Efecto mediano")
else:
    print("  d ≥ 0.8: Efecto grande")
```

## Pruebas Pareadas (Paired t-test)

Cuando los mismos sujetos se prueban en dos condiciones, usas una **prueba pareada**.

Ejemplo: Ejecutas 20 kernels con ambos métodos (baseline y restricciones) y comparas.

```
Kernel | Baseline | Restricciones | Diferencia
-------|----------|---------------|-----------
1      | 4.1      | 3.9           | 0.2
2      | 4.3      | 3.8           | 0.5
3      | 3.9      | 3.7           | 0.2
...    | ...      | ...           | ...
```

En lugar de comparar dos muestras independientes, calculas las diferencias por par y pruebas si el promedio de diferencias es 0.

```
H₀: μ_diferencia = 0
H₁: μ_diferencia ≠ 0

t = d̄ / (s_d / √n)
```

### Ventaja de Diseños Pareados

Controlas la variabilidad entre kernels. El mismo kernel tiende a tomar tiempos similares en ambos métodos; solo importa la diferencia.

Esto reduce varianza y aumenta poder, permitiéndote detectar efectos más pequeños.

**Paired vs Unpaired:**
- Paired: mismos 50 kernels con método A y B (cada kernel es su propio control)
- Unpaired: 50 kernels método A vs 50 kernels diferentes método B
- Regla: si puedes emparejar, hazlo (más poder estadístico)

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Datos pareados: mismo kernel ejecutado con ambos métodos
np.random.seed(42)
n = 20

# Simulamos que cada kernel tiene su propio "dificultad base"
dificultad_base = np.random.uniform(3, 6, n)

# Baseline y Restricciones con algo de ruido
baseline = dificultad_base + np.random.normal(0.3, 0.2, n)
restricciones = dificultad_base + np.random.normal(-0.2, 0.2, n)

# Calcular diferencias
diferencias = baseline - restricciones

# Prueba t pareada
t_stat_paired, p_value_paired = stats.ttest_rel(baseline, restricciones)

# Estadísticas de las diferencias
d_bar = np.mean(diferencias)
s_d = np.std(diferencias, ddof=1)
se_d = s_d / np.sqrt(n)
df = n - 1

# Intervalo de confianza
t_critical = stats.t.ppf(0.975, df)
ci_lower = d_bar - t_critical * se_d
ci_upper = d_bar + t_critical * se_d

# Cohen's d para datos pareados
cohens_d = d_bar / s_d

print("Prueba t Pareada (Paired t-test)")
print("=" * 60)
print(f"H₀: μ_diferencia = 0 (no hay diferencia entre métodos)")
print(f"H₁: μ_diferencia ≠ 0 (hay diferencia)")
print(f"\nDatos: {n} kernels ejecutados con ambos métodos")
print(f"Baseline: x̄={np.mean(baseline):.3f}, s={np.std(baseline, ddof=1):.3f}")
print(f"Restricciones: x̄={np.mean(restricciones):.3f}, s={np.std(restricciones, ddof=1):.3f}")
print(f"\nDiferencias (Baseline - Restricciones):")
print(f"  Media: d̄ = {d_bar:.3f}")
print(f"  SD: s_d = {s_d:.3f}")
print(f"  SE: s_d/√n = {se_d:.3f}")
print(f"\nEstadístico t: {t_stat_paired:.3f}")
print(f"p-valor: {p_value_paired:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")
print(f"IC 95%: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"\nConclusión (α = 0.05):")
if p_value_paired < 0.05:
    print(f"  p < 0.05 → Rechazamos H₀")
    print(f"  Las restricciones REDUCEN significativamente el tiempo")
else:
    print(f"  p ≥ 0.05 → No rechazamos H₀")

# Comparación con prueba no pareada (para demostrar la diferencia)
t_stat_unpaired, p_value_unpaired = stats.ttest_ind(baseline, restricciones)

print(f"\nComparación con prueba NO pareada:")
print(f"  t pareada: {t_stat_paired:.3f}, p = {p_value_paired:.4f}")
print(f"  t NO pareada: {t_stat_unpaired:.3f}, p = {p_value_unpaired:.4f}")
print(f"  La prueba pareada tiene MÁS poder (p-valor más pequeño)")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Datos pareados - líneas conectando pares
axes[0, 0].plot([1]*n, baseline, 'ro', alpha=0.6, markersize=8, label='Baseline')
axes[0, 0].plot([2]*n, restricciones, 'bo', alpha=0.6, markersize=8, label='Restricciones')

for i in range(n):
    color = 'green' if baseline[i] > restricciones[i] else 'red'
    axes[0, 0].plot([1, 2], [baseline[i], restricciones[i]],
                    color=color, alpha=0.3, linewidth=1)

axes[0, 0].plot([1, 2], [np.mean(baseline), np.mean(restricciones)],
                'k-', linewidth=3, marker='D', markersize=12,
                label='Medias')

axes[0, 0].set_xlim(0.5, 2.5)
axes[0, 0].set_xticks([1, 2])
axes[0, 0].set_xticklabels(['Baseline', 'Restricciones'])
axes[0, 0].set_ylabel('Tiempo de compilación (s)')
axes[0, 0].set_title('Datos Pareados: Mismo Kernel con Ambos Métodos')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Histograma de diferencias
axes[0, 1].hist(diferencias, bins=10, color='purple', alpha=0.7,
                edgecolor='black', density=True)
axes[0, 1].axvline(d_bar, color='red', linewidth=2, linestyle='--',
                   label=f'd̄ = {d_bar:.3f}')
axes[0, 1].axvline(0, color='green', linewidth=2, linestyle=':',
                   label='H₀: diferencia = 0')

# Superponer distribución normal
x_range = np.linspace(min(diferencias), max(diferencias), 100)
axes[0, 1].plot(x_range, stats.norm.pdf(x_range, d_bar, s_d),
                'r-', linewidth=2, alpha=0.7, label=f'N({d_bar:.2f}, {s_d:.2f}²)')

axes[0, 1].set_xlabel('Diferencia (Baseline - Restricciones)')
axes[0, 1].set_ylabel('Densidad')
axes[0, 1].set_title('Distribución de las Diferencias')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Distribución t y estadístico
x = np.linspace(-4, 4, 1000)
t_pdf = stats.t.pdf(x, df)

axes[1, 0].plot(x, t_pdf, 'b-', linewidth=2, label=f't({df})')
axes[1, 0].fill_between(x, 0, t_pdf, alpha=0.3, color='blue')

# Regiones críticas
x_left = x[x <= -t_critical]
x_right = x[x >= t_critical]
axes[1, 0].fill_between(x_left, 0, stats.t.pdf(x_left, df),
                        alpha=0.6, color='red', label='Región crítica')
axes[1, 0].fill_between(x_right, 0, stats.t.pdf(x_right, df),
                        alpha=0.6, color='red')

axes[1, 0].axvline(t_stat_paired, color='green', linewidth=3, linestyle='--',
                   label=f't obs = {t_stat_paired:.2f}')

axes[1, 0].set_xlabel('Estadístico t')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].set_title('Distribución t y Estadístico Observado')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Tabla de datos
tabla_data = {
    'Kernel': range(1, min(11, n+1)),
    'Baseline': [f'{b:.2f}' for b in baseline[:min(10, n)]],
    'Restricciones': [f'{r:.2f}' for r in restricciones[:min(10, n)]],
    'Diferencia': [f'{d:.2f}' for d in diferencias[:min(10, n)]]
}

df_tabla = pd.DataFrame(tabla_data)

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=df_tabla.values,
                         colLabels=df_tabla.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Colorear header
for i in range(len(df_tabla.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

axes[1, 1].set_title(f'Primeros {min(10, n)} Pares de Datos\n(Mostrando dependencia par a par)',
                     fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("\nVentajas de Diseños Pareados:")
print("  1. Controla variabilidad individual (cada kernel es su propio control)")
print("  2. Mayor poder estadístico (menor varianza residual)")
print("  3. Necesita menos sujetos para detectar el mismo efecto")
```

## Decisiones e Interpretación

### Flujo de Decisión

```
1. Establece H₀, H₁, α antes del análisis
2. Recolecta datos
3. Calcula estadístico de prueba
4. Obtén p-valor
5. ¿p < α?
   - Sí: Rechaza H₀ (evidencia para H₁)
   - No: No rechaces H₀ (sin evidencia suficiente)
```

### Lo Que NO Dicen los P-valores

- **NO te dice la probabilidad de H₀**: p-valor asume H₀
- **NO te dice el tamaño del efecto**: p < 0.01 puede ser efecto minúsculo si n es muy grande
- **NO controla la tasa de falsos positivos globalmente**: solo para una prueba

## Resumen: Tipos de Pruebas

| Prueba | Compara | Asunciones | Uso |
|--------|---------|-----------|-----|
| t de una muestra | Muestra vs. valor | Normalidad | ¿Es diferente de un estándar? |
| t de dos muestras | Dos grupos | Normalidad, igualdad varianza | ¿Difieren dos grupos? |
| t pareada | Mismo sujeto x2 condiciones | Normalidad de diferencias | ¿Cambia tras intervención? |

Próximas semanas usaremos alternativas no paramétricas cuando no se cumplan asunciones.

## Ejercicios y Reflexión

### Ejercicio 1: Plantear Hipótesis
Para tu proyecto, escribe formalmente:
- H₀ y H₁ para comparar baseline vs. restricciones en términos de:
  - Tasa de validez
  - Número promedio de iteraciones
  - Tiempo de compilación
- ¿Son pruebas de una cola o dos colas? ¿Por qué?

### Ejercicio 2: Interpretación de P-valores
Para cada p-valor, decide qué conclusión sacar (α = 0.05):
1. p = 0.002: Evidencia ___ para H₁
2. p = 0.045: Evidencia ___ para H₁
3. p = 0.051: Evidencia ___ para H₁
4. p = 0.15: Evidencia ___ para H₁

(Llena con: "fuerte", "moderada", "débil", "ninguna")

### Ejercicio 3: Errores Tipo I y II
En tu proyecto:
- ¿Cuál es peor: decir que tus restricciones mejoran cuando no lo hacen, o perder detectar una mejora real?
- ¿Qué α y β propondrías? ¿Por qué?

### Ejercicio 4: Analizar un Escenario
Datos hipotéticos:
- Baseline: n=25, x̄=5.2, s=1.1
- Restricciones: n=25, x̄=4.8, s=1.3

a) Escribe H₀ y H₁
b) ¿Cuál es el error estándar de la diferencia?
c) Calcula el estadístico t aproximadamente
d) Con df≈48, ¿es esto significante a α=0.05?

### Reflexión
1. En tu proyecto, ¿cuál es la consecuencia de un Error Tipo I? ¿Y de un Tipo II?
2. ¿Cómo afecta el tamaño muestral tu capacidad de detectar diferencias?
3. Si observas p=0.06, ¿deberías concluir que no hay efecto? ¿Por qué o por qué no?

---

**Próxima semana**: Aprenderemos a diseñar experimentos cuidadosamente para asegurar que tenemos suficiente poder para detectar efectos reales.

---

## Referencias

- Fisher, R.A. (1925). [Statistical Methods for Research Workers](https://psychclassics.yorku.ca/Fisher/Methods/). Oliver & Boyd.
- Neyman, J. & Pearson, E. (1933). [On the Problem of the Most Efficient Tests of Statistical Hypotheses](https://doi.org/10.1098/rsta.1933.0009). Philosophical Transactions A.
- Student (Gosset, W.S.) (1908). [The Probable Error of a Mean](https://doi.org/10.1093/biomet/6.1.1). Biometrika.
