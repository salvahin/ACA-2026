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

# Comparaciones Múltiples
## Semana 8 - Estadística para Generación de Kernels GPU

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Entender el problema de inflación del error Tipo I con múltiples pruebas
- Aplicar correcciones de Bonferroni y Holm para controlar FWER
- Ejecutar e interpretar ANOVA para comparar 3+ grupos
- Realizar pruebas post-hoc (Tukey HSD) después de ANOVA significante
- Distinguir entre FWER y FDR y cuándo usar cada uno
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[StatQuest: False Discovery Rates (FDR)](https://www.youtube.com/watch?v=K8LQSvtjcEo)** - Josh Starmer explica claramente el problema de las comparaciones múltiples, la tasa de falsos descubrimientos, y cómo las correcciones como Benjamini-Hochberg ayudan a controlarlo.
```

Aquí llegamos a un peligro estadístico: **si haces suficientes pruebas, alguna será "significante" solo por suerte**. Aprendemos a controlar esto.

## El Problema de Comparaciones Múltiples

```{admonition} 💡 Intuición
:class: hint
Imagina lanzar 20 monedas justas. Esperas ~10 caras. Pero si SOLO reportas la moneda que salió cara 8 de 10 veces y dices "¡esta moneda está cargada!", estás haciendo trampa. Cuando haces 20 pruebas estadísticas, es muy probable que AL MENOS UNA dé p<0.05 solo por azar, aunque no haya efecto real.
```

Imagina que tienes 20 variables y comparas baseline vs. restricciones. Sin correcciones:

```
Pruebas independientes: 20
Nivel de significancia por prueba: α = 0.05
Probabilidad de al menos un falso positivo (Tipo I):
P(al menos un falso positivo) = 1 - (1 - 0.05)²⁰
                               = 1 - 0.95²⁰
                               = 1 - 0.358
                               = 0.642

¡64% de probabilidad de un falso positivo!
```

```{admonition} 🎯 Aplicación en ML
:class: important
En tu proyecto de evaluación de kernels GPU, esto te afecta cuando:
- Comparas múltiples métricas: validez, iteraciones, tiempo, memoria
- Pruebas múltiples configuraciones: temperatura 0.0, 0.3, 0.5, 0.7, 1.0
- Evalúas en múltiples datasets: kernels simples, medianos, complejos

**Sin corrección**: Muy probable encontrar "significancia" falsa
**Con corrección**: Proteges contra declarar efectos inexistentes
```

Acabas reportando un resultado "significante" que fue pura suerte.

## Tasa de Error Familia-wise (FWER)

**FWER** = Probabilidad de al menos un Error Tipo I en toda la familia de pruebas.

Sin corrección: FWER ≈ 0.64 (arriba)
Con corrección Bonferroni: FWER ≤ 0.05

:::{figure} diagrams/fwer_inflation.png
:name: fig-fwer-inflation
:alt: Gráfico mostrando inflación de FWER con número de pruebas múltiples
:align: center
:width: 90%

**Figura 7:** Inflación de la tasa de error familia-wise (FWER) según el número de pruebas realizadas.
:::

### Corrección de Bonferroni

Simple: divide α por número de pruebas.

```
α_ajustado = α_original / número de pruebas
          = 0.05 / 20
          = 0.0025

Reportas como significante solo si p < 0.0025
```

**Desventaja**: Muy conservador. Pierde poder. Con 20 pruebas, casi nada será significante.

## Corrección de Holm

Menos conservador que Bonferroni, pero controla FWER.

```
1. Ordena p-valores de menor a mayor: p₁ ≤ p₂ ≤ ... ≤ pₖ
2. Para i-ésima prueba, usa umbral: α / (k - i + 1)

Ejemplo con k=5 pruebas:
p₁ = 0.001, umbral: 0.05/5 = 0.010  → Significante (0.001 < 0.010)
p₂ = 0.008, umbral: 0.05/4 = 0.0125 → Significante (0.008 < 0.0125)
p₃ = 0.024, umbral: 0.05/3 = 0.0167 → No significante (0.024 > 0.0167)
p₄ = 0.037, umbral: 0.05/2 = 0.025  → No significante
p₅ = 0.053, umbral: 0.05/1 = 0.050  → No significante
```

**Ventaja**: Menos poder perdido que Bonferroni.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Simulación del problema de comparaciones múltiples
np.random.seed(42)

num_pruebas = 20
alpha = 0.05

# Generar p-valores bajo H₀ (sin efecto real)
# Estos deberían ser uniformes entre 0 y 1
p_valores_nulos = np.random.uniform(0, 1, num_pruebas)

# Algunos con efecto real (p-valores pequeños)
p_valores_reales = np.concatenate([
    np.random.beta(2, 20, 5),  # 5 con efecto real
    np.random.uniform(0, 1, 15)  # 15 sin efecto
])

np.random.shuffle(p_valores_reales)

print("Problema de Comparaciones Múltiples")
print("=" * 60)
print(f"Número de pruebas: {num_pruebas}")
print(f"Nivel de significancia por prueba: α = {alpha}")
print(f"\nProbabilidad de al menos un falso positivo (FWER):")
print(f"  FWER = 1 - (1 - α)^k")
print(f"  FWER = 1 - (1 - {alpha})^{num_pruebas}")
fwer_sin_correccion = 1 - (1 - alpha)**num_pruebas
print(f"  FWER = {fwer_sin_correccion:.3f} ({fwer_sin_correccion*100:.1f}%)")
print(f"\n¡{fwer_sin_correccion*100:.0f}% de probabilidad de al menos un falso positivo!")

# Correcciones
print("\n" + "=" * 60)
print("CORRECCIONES")
print("=" * 60)

# 1. Sin corrección
significantes_sin = sum(p_valores_reales < alpha)
print(f"\n1. Sin corrección (α = {alpha}):")
print(f"   Pruebas significantes: {significantes_sin}/{num_pruebas}")

# 2. Bonferroni
alpha_bonferroni = alpha / num_pruebas
significantes_bonf = sum(p_valores_reales < alpha_bonferroni)
print(f"\n2. Bonferroni (α_ajustado = {alpha_bonferroni:.4f}):")
print(f"   Pruebas significantes: {significantes_bonf}/{num_pruebas}")

# 3. Holm
p_ordenados = np.sort(p_valores_reales)
indices_ordenados = np.argsort(p_valores_reales)
significantes_holm = 0

print(f"\n3. Holm (procedimiento secuencial):")
print(f"   {'i':<4} {'p-valor':<10} {'umbral':<12} {'Significante'}")
print(f"   {'-'*40}")

for i, p in enumerate(p_ordenados[:5]):  # mostrar primeros 5
    umbral = alpha / (num_pruebas - i)
    es_sig = p <= umbral
    if es_sig:
        significantes_holm = i + 1
    print(f"   {i+1:<4} {p:<10.4f} {umbral:<12.4f} {'✓' if es_sig else '✗'}")

print(f"   ...")
print(f"   Total significantes: {significantes_holm}/{num_pruebas}")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. FWER vs Número de Pruebas
num_pruebas_range = np.arange(1, 51)
fwer_values = 1 - (1 - alpha)**num_pruebas_range

axes[0, 0].plot(num_pruebas_range, fwer_values, 'b-', linewidth=2)
axes[0, 0].axhline(alpha, color='red', linestyle='--', linewidth=2,
                   label=f'α nominal = {alpha}')
axes[0, 0].axvline(num_pruebas, color='green', linestyle=':', linewidth=2,
                   label=f'k = {num_pruebas}')
axes[0, 0].fill_between(num_pruebas_range, 0, fwer_values,
                        where=(fwer_values > alpha),
                        alpha=0.3, color='red',
                        label='FWER inflado')

axes[0, 0].set_xlabel('Número de pruebas (k)')
axes[0, 0].set_ylabel('FWER (Tasa de Error Family-wise)')
axes[0, 0].set_title('Inflación del Error Tipo I con Múltiples Pruebas')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Anotar punto específico
axes[0, 0].plot(num_pruebas, fwer_sin_correccion, 'ro', markersize=10)
axes[0, 0].text(num_pruebas + 2, fwer_sin_correccion,
                f'({num_pruebas}, {fwer_sin_correccion:.2f})',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow'))

# 2. P-valores ordenados con umbrales de corrección
axes[0, 1].scatter(range(1, num_pruebas+1), p_ordenados,
                   s=100, alpha=0.7, edgecolor='black', linewidth=2,
                   c=p_ordenados, cmap='RdYlGn_r')

# Líneas de umbral
x_range = np.arange(1, num_pruebas+1)
axes[0, 1].axhline(alpha, color='red', linestyle='-', linewidth=2,
                   label=f'Sin corrección (α={alpha})')

# Bonferroni
axes[0, 1].axhline(alpha_bonferroni, color='blue', linestyle='--', linewidth=2,
                   label=f'Bonferroni (α/{num_pruebas})')

# Holm (línea descendente)
umbrales_holm = [alpha / (num_pruebas - i) for i in range(num_pruebas)]
axes[0, 1].plot(x_range, umbrales_holm, 'g--', linewidth=2, label='Holm')

axes[0, 1].set_xlabel('Prueba (ordenada por p-valor)')
axes[0, 1].set_ylabel('P-valor')
axes[0, 1].set_title('P-valores y Umbrales de Corrección')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# 3. Comparación de métodos
metodos = ['Sin\ncorrección', 'Bonferroni', 'Holm']
num_sig = [significantes_sin, significantes_bonf, significantes_holm]
colores = ['red', 'blue', 'green']

bars = axes[1, 0].bar(metodos, num_sig, color=colores, alpha=0.7,
                      edgecolor='black', linewidth=2)

axes[1, 0].set_ylabel('Número de pruebas significantes')
axes[1, 0].set_title('Comparación de Métodos de Corrección')
axes[1, 0].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, num_sig):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)

# 4. Simulación: Tasa de descubrimientos falsos
num_simulaciones = 1000
num_pruebas_sim = 20
num_verdaderas_h1 = 5  # 5 efectos reales de 20

tasas_fp_sin = []
tasas_fp_bonf = []
tasas_fp_holm = []

for _ in range(num_simulaciones):
    # Generar p-valores: 5 con efecto, 15 sin efecto
    p_vals = np.concatenate([
        np.random.beta(2, 20, num_verdaderas_h1),  # con efecto
        np.random.uniform(0, 1, num_pruebas_sim - num_verdaderas_h1)  # sin efecto
    ])
    np.random.shuffle(p_vals)

    # Identificar cuáles son realmente H₀
    indices_h0 = np.arange(num_verdaderas_h1, num_pruebas_sim)

    # Sin corrección
    sig_sin = p_vals < alpha
    fp_sin = sum(sig_sin[indices_h0])
    if sum(sig_sin) > 0:
        tasas_fp_sin.append(fp_sin / sum(sig_sin))

    # Bonferroni
    sig_bonf = p_vals < (alpha / num_pruebas_sim)
    fp_bonf = sum(sig_bonf[indices_h0])
    if sum(sig_bonf) > 0:
        tasas_fp_bonf.append(fp_bonf / sum(sig_bonf))

axes[1, 1].hist(tasas_fp_sin, bins=20, alpha=0.6, color='red',
                edgecolor='black', label='Sin corrección')
axes[1, 1].hist(tasas_fp_bonf, bins=20, alpha=0.6, color='blue',
                edgecolor='black', label='Bonferroni')

axes[1, 1].axvline(np.mean(tasas_fp_sin), color='red', linestyle='--',
                   linewidth=2, label=f'Media sin corr = {np.mean(tasas_fp_sin):.2f}')
if len(tasas_fp_bonf) > 0:
    axes[1, 1].axvline(np.mean(tasas_fp_bonf), color='blue', linestyle='--',
                       linewidth=2, label=f'Media Bonf = {np.mean(tasas_fp_bonf):.2f}')

axes[1, 1].set_xlabel('Tasa de Descubrimientos Falsos (FDR)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].set_title(f'Simulación: FDR en {num_simulaciones} experimentos')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nConclusión:")
print("  - Sin corrección: Alto riesgo de falsos positivos")
print("  - Bonferroni: Muy conservador, pierde poder")
print("  - Holm: Mejor balance entre control de error y poder")
```

## ANOVA: Comparar 3+ Grupos

Cuando comparas más de 2 grupos, uses **ANOVA** (Analysis of Variance) en lugar de múltiples t-tests.

```
H₀: μ₁ = μ₂ = μ₃ = ... = μₖ (todos los grupos son iguales)
H₁: Al menos un grupo difiere
```

### Lógica de ANOVA

ANOVA particiona varianza total en:
- **Varianza entre grupos** (SSB): ¿Qué tan diferentes son los promedios?
- **Varianza dentro de grupos** (SSW): ¿Cuánta variación hay internamente?

```
F = (SSB / df_between) / (SSW / df_within)
  = Mean Square Between / Mean Square Within

Si F es grande: grupos muy diferentes (significante)
Si F es pequeño: grupos similares (no significante)
```

### Ejemplo: Tres Métodos de Decoding

```
Método A (Baseline):     42, 43, 41, 45, 43  (M=42.8, SD=1.6)
Método B (Restricciones): 38, 39, 37, 40, 38  (M=38.4, SD=1.1)
Método C (Restricciones v2): 35, 37, 36, 38, 34  (M=36.0, SD=1.4)

ANOVA:
F = 47.2, df=(2,12), p < 0.001

Conclusión: Métodos difieren significativamente.
```

Pero, ¿cuál difiere de cuál?

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Datos de tres métodos de decoding
np.random.seed(42)

metodo_A = np.array([42, 43, 41, 45, 43])
metodo_B = np.array([38, 39, 37, 40, 38])
metodo_C = np.array([35, 37, 36, 38, 34])

# Combinar todos los datos
todos_datos = np.concatenate([metodo_A, metodo_B, metodo_C])
grupos = ['A'] * len(metodo_A) + ['B'] * len(metodo_B) + ['C'] * len(metodo_C)

# ANOVA de una vía
F_stat, p_anova = stats.f_oneway(metodo_A, metodo_B, metodo_C)

# Grados de libertad
k = 3  # número de grupos
n_total = len(todos_datos)
df_between = k - 1
df_within = n_total - k

print("ANOVA: Comparación de 3+ Grupos")
print("=" * 60)
print(f"H₀: μ_A = μ_B = μ_C (todos los grupos son iguales)")
print(f"H₁: Al menos un grupo difiere")
print(f"\nDatos:")
print(f"  Método A (Baseline):     {metodo_A} (M={np.mean(metodo_A):.1f}, SD={np.std(metodo_A, ddof=1):.1f})")
print(f"  Método B (Restricciones): {metodo_B} (M={np.mean(metodo_B):.1f}, SD={np.std(metodo_B, ddof=1):.1f})")
print(f"  Método C (Restricciones v2): {metodo_C} (M={np.mean(metodo_C):.1f}, SD={np.std(metodo_C, ddof=1):.1f})")
print(f"\nResultados ANOVA:")
print(f"  F({df_between}, {df_within}) = {F_stat:.2f}")
print(f"  p-valor = {p_anova:.4f}")
print(f"\nConclusión (α = 0.05):")
if p_anova < 0.05:
    print(f"  p < 0.05 → Rechazamos H₀")
    print(f"  Los métodos difieren significativamente")
    print(f"  Necesitamos pruebas post-hoc para saber CUÁLES difieren")
else:
    print(f"  p ≥ 0.05 → No rechazamos H₀")
    print(f"  No hay evidencia de diferencias entre métodos")

# Cálculos manuales de ANOVA
# Between-group variance (SSB)
media_total = np.mean(todos_datos)
media_A = np.mean(metodo_A)
media_B = np.mean(metodo_B)
media_C = np.mean(metodo_C)

SSB = len(metodo_A) * (media_A - media_total)**2 + \
      len(metodo_B) * (media_B - media_total)**2 + \
      len(metodo_C) * (media_C - media_total)**2

# Within-group variance (SSW)
SSW = np.sum((metodo_A - media_A)**2) + \
      np.sum((metodo_B - media_B)**2) + \
      np.sum((metodo_C - media_C)**2)

# Total variance (SST)
SST = np.sum((todos_datos - media_total)**2)

# Mean squares
MSB = SSB / df_between
MSW = SSW / df_within

print(f"\nDesglose de Varianza:")
print(f"  SST (Total)        = {SST:.2f}")
print(f"  SSB (Between)      = {SSB:.2f}")
print(f"  SSW (Within)       = {SSW:.2f}")
print(f"  MSB = SSB/df_b     = {MSB:.2f}")
print(f"  MSW = SSW/df_w     = {MSW:.2f}")
print(f"  F = MSB/MSW        = {MSB/MSW:.2f}")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Box plots de los tres grupos
datos_plot = [metodo_A, metodo_B, metodo_C]
bp = axes[0, 0].boxplot(datos_plot, labels=['Método A', 'Método B', 'Método C'],
                        patch_artist=True, notch=True, showmeans=True)

colores = ['red', 'blue', 'green']
for patch, color in zip(bp['boxes'], colores):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axes[0, 0].set_ylabel('Iteraciones')
axes[0, 0].set_title('Comparación de Tres Métodos de Decoding')
axes[0, 0].grid(axis='y', alpha=0.3)

# Anotar medias
for i, (datos, color) in enumerate(zip(datos_plot, colores)):
    axes[0, 0].plot(i+1, np.mean(datos), 'D', markersize=10,
                    color=color, markeredgecolor='black', markeredgewidth=2)

# 2. Distribución F y región crítica
x = np.linspace(0, 10, 1000)
f_pdf = stats.f.pdf(x, df_between, df_within)

axes[0, 1].plot(x, f_pdf, 'b-', linewidth=2, label=f'F({df_between}, {df_within})')
axes[0, 1].fill_between(x, 0, f_pdf, alpha=0.3, color='blue')

# Región crítica
f_critical = stats.f.ppf(0.95, df_between, df_within)
x_crit = x[x >= f_critical]
axes[0, 1].fill_between(x_crit, 0, stats.f.pdf(x_crit, df_between, df_within),
                        alpha=0.6, color='red', label='Región crítica (α=0.05)')

# Marcar F observado
axes[0, 1].axvline(F_stat, color='green', linewidth=3, linestyle='--',
                   label=f'F observado = {F_stat:.2f}')
axes[0, 1].axvline(f_critical, color='red', linewidth=2, linestyle=':',
                   label=f'F crítico = {f_critical:.2f}')

axes[0, 1].set_xlabel('Estadístico F')
axes[0, 1].set_ylabel('Densidad de Probabilidad')
axes[0, 1].set_title('Distribución F y Prueba ANOVA')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Visualización de varianza Between vs Within
axes[1, 0].scatter([1]*len(metodo_A), metodo_A, s=100, alpha=0.6,
                   color='red', edgecolor='black', label='Método A')
axes[1, 0].scatter([2]*len(metodo_B), metodo_B, s=100, alpha=0.6,
                   color='blue', edgecolor='black', label='Método B')
axes[1, 0].scatter([3]*len(metodo_C), metodo_C, s=100, alpha=0.6,
                   color='green', edgecolor='black', label='Método C')

# Medias de grupo
axes[1, 0].plot([1, 2, 3], [media_A, media_B, media_C],
                'ko-', markersize=15, linewidth=3, label='Medias de grupo')

# Media total
axes[1, 0].axhline(media_total, color='purple', linestyle='--',
                   linewidth=2, label=f'Media total = {media_total:.1f}')

# Anotar varianza between
axes[1, 0].annotate('', xy=(1, media_total), xytext=(1, media_A),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
axes[1, 0].text(1.2, (media_total + media_A)/2, 'Between\nvariance',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow'))

axes[1, 0].set_xlim(0.5, 3.5)
axes[1, 0].set_xticks([1, 2, 3])
axes[1, 0].set_xticklabels(['A', 'B', 'C'])
axes[1, 0].set_ylabel('Iteraciones')
axes[1, 0].set_title('Varianza Between (entre grupos) vs Within (dentro de grupos)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Tabla ANOVA
tabla_anova = [
    ['Between Groups', f'{SSB:.2f}', str(df_between), f'{MSB:.2f}', f'{F_stat:.2f}', f'{p_anova:.4f}'],
    ['Within Groups', f'{SSW:.2f}', str(df_within), f'{MSW:.2f}', '', ''],
    ['Total', f'{SST:.2f}', str(n_total-1), '', '', '']
]

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=tabla_anova,
                         colLabels=['Fuente', 'SS', 'df', 'MS', 'F', 'p-valor'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.15, 0.1, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Colorear header
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Colorear filas
table[(1, 0)].set_facecolor('lightblue')
table[(2, 0)].set_facecolor('lightcoral')
table[(3, 0)].set_facecolor('lightgray')

axes[1, 1].set_title('Tabla ANOVA (Analysis of Variance)',
                     fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("\nInterpretación:")
print("  F grande → Varianza between > within → grupos diferentes")
print("  F pequeño → Varianza between ≈ within → grupos similares")
```

## Pruebas Post-Hoc: Desglosa Comparaciones

Después de ANOVA significante, necesitas saber qué grupos difieren.

### Prueba de Tukey HSD

Compara todos los pares, controlando FWER.

```
Comparaciones:
A vs. B: diferencia = 42.8 - 38.4 = 4.4 → p < 0.001 ✓
A vs. C: diferencia = 42.8 - 36.0 = 6.8 → p < 0.001 ✓
B vs. C: diferencia = 38.4 - 36.0 = 2.4 → p = 0.021 ✓

Conclusión: Todos los pares difieren.
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# Datos de los tres métodos (ampliados para más realismo)
np.random.seed(42)

metodo_A = np.random.normal(42.8, 1.6, 20)
metodo_B = np.random.normal(38.4, 1.1, 20)
metodo_C = np.random.normal(36.0, 1.4, 20)

# ANOVA primero
F_stat, p_anova = stats.f_oneway(metodo_A, metodo_B, metodo_C)

print("Pruebas Post-Hoc: Tukey HSD")
print("=" * 60)
print(f"Paso 1: ANOVA")
print(f"  F = {F_stat:.2f}, p < 0.001")
print(f"  Conclusión: Al menos un grupo difiere")
print(f"\nPaso 2: Pruebas Post-Hoc (Tukey HSD)")
print(f"  Comparamos todos los pares controlando FWER")

# Implementación simplificada de Tukey HSD
grupos = [metodo_A, metodo_B, metodo_C]
nombres = ['A (Baseline)', 'B (Restricciones)', 'C (Restricciones v2)']

n = len(metodo_A)
k = 3
df_within = n * k - k

# Calcular MSW (Mean Square Within)
SSW = np.sum((metodo_A - np.mean(metodo_A))**2) + \
      np.sum((metodo_B - np.mean(metodo_B))**2) + \
      np.sum((metodo_C - np.mean(metodo_C))**2)
MSW = SSW / df_within

# Comparaciones pareadas
comparaciones = list(combinations(range(k), 2))
resultados = []

print(f"\nComparaciones Pareadas:")
print(f"  {'Par':<20} {'Diferencia':<12} {'SE':<8} {'q':<8} {'p-valor'}")
print(f"  {'-'*60}")

for i, j in comparaciones:
    media_i = np.mean(grupos[i])
    media_j = np.mean(grupos[j])
    diff = abs(media_i - media_j)

    # Error estándar
    se = np.sqrt(MSW / n)

    # Estadístico q de Tukey
    q = diff / se

    # Aproximación del p-valor usando distribución studentized range
    # (simplificado - en práctica usar scipy.stats.studentized_range cuando disponible)
    from scipy.stats import t
    # Aproximación conservadora con t-distribution
    p_val = 2 * (1 - t.cdf(q/np.sqrt(2), df_within))

    resultados.append({
        'par': f"{nombres[i]} vs {nombres[j]}",
        'diff': diff,
        'se': se,
        'q': q,
        'p': p_val,
        'sig': '✓' if p_val < 0.05 else '✗'
    })

    print(f"  {nombres[i]:<10} vs {nombres[j]:<10} {diff:>6.2f}       "
          f"{se:>6.3f}  {q:>6.2f}  {p_val:>6.4f} {resultados[-1]['sig']}")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribuciones con comparaciones
axes[0, 0].hist(metodo_A, bins=10, alpha=0.5, color='red',
                edgecolor='black', label='Método A', density=True)
axes[0, 0].hist(metodo_B, bins=10, alpha=0.5, color='blue',
                edgecolor='black', label='Método B', density=True)
axes[0, 0].hist(metodo_C, bins=10, alpha=0.5, color='green',
                edgecolor='black', label='Método C', density=True)

axes[0, 0].axvline(np.mean(metodo_A), color='red', linewidth=2, linestyle='--')
axes[0, 0].axvline(np.mean(metodo_B), color='blue', linewidth=2, linestyle='--')
axes[0, 0].axvline(np.mean(metodo_C), color='green', linewidth=2, linestyle='--')

axes[0, 0].set_xlabel('Iteraciones')
axes[0, 0].set_ylabel('Densidad')
axes[0, 0].set_title('Distribuciones de los Tres Métodos')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Diagrama de medias con intervalos de confianza
medias = [np.mean(g) for g in grupos]
errores = [stats.sem(g) * stats.t.ppf(0.975, len(g)-1) for g in grupos]

x_pos = [1, 2, 3]
colores_bars = ['red', 'blue', 'green']

axes[0, 1].errorbar(x_pos, medias, yerr=errores,
                    fmt='o', markersize=12, capsize=10, capthick=2,
                    color='black', elinewidth=2)

for i, (x, m, c) in enumerate(zip(x_pos, medias, colores_bars)):
    axes[0, 1].scatter(x, m, s=200, color=c, alpha=0.6,
                      edgecolor='black', linewidth=2, zorder=3)

# Anotar diferencias significativas
for res in resultados:
    if res['sig'] == '✓':
        # Extraer índices de los métodos
        nombres_split = res['par'].split(' vs ')
        idx_i = nombres.index(nombres_split[0])
        idx_j = nombres.index(nombres_split[1])

        y_max = max(medias[idx_i], medias[idx_j]) + max(errores[idx_i], errores[idx_j]) + 1
        axes[0, 1].plot([x_pos[idx_i], x_pos[idx_j]], [y_max, y_max],
                       'k-', linewidth=1.5)
        axes[0, 1].text((x_pos[idx_i] + x_pos[idx_j])/2, y_max + 0.3,
                       f'p={res["p"]:.3f}*',
                       ha='center', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

axes[0, 1].set_xlim(0.5, 3.5)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(['A', 'B', 'C'])
axes[0, 1].set_ylabel('Iteraciones (media ± IC 95%)')
axes[0, 1].set_title('Comparaciones Post-Hoc con Intervalos de Confianza')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Matriz de p-valores (heatmap)
p_matrix = np.ones((k, k))
for res in resultados:
    nombres_split = res['par'].split(' vs ')
    idx_i = nombres.index(nombres_split[0])
    idx_j = nombres.index(nombres_split[1])
    p_matrix[idx_i, idx_j] = res['p']
    p_matrix[idx_j, idx_i] = res['p']

im = axes[1, 0].imshow(p_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.10)
axes[1, 0].set_xticks([0, 1, 2])
axes[1, 0].set_yticks([0, 1, 2])
axes[1, 0].set_xticklabels(['A', 'B', 'C'])
axes[1, 0].set_yticklabels(['A', 'B', 'C'])

# Anotar p-valores
for i in range(k):
    for j in range(k):
        if i != j:
            text = axes[1, 0].text(j, i, f'{p_matrix[i, j]:.4f}',
                                  ha='center', va='center',
                                  color='white' if p_matrix[i, j] < 0.03 else 'black',
                                  fontweight='bold')
        else:
            axes[1, 0].text(j, i, '-', ha='center', va='center',
                          fontsize=16, fontweight='bold')

axes[1, 0].set_title('Matriz de P-valores (Comparaciones Pareadas)')
plt.colorbar(im, ax=axes[1, 0], label='p-valor')

# 4. Resumen de significancia
pares = [r['par'].split(' vs ') for r in resultados]
significantes = [r['sig'] == '✓' for r in resultados]

axes[1, 1].axis('tight')
axes[1, 1].axis('off')

tabla_data = [[r['par'], f"{r['diff']:.2f}", f"{r['q']:.2f}",
               f"{r['p']:.4f}", r['sig']] for r in resultados]

table = axes[1, 1].table(cellText=tabla_data,
                         colLabels=['Comparación', 'Diferencia', 'q', 'p-valor', 'Sig'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.15, 0.15, 0.15, 0.1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Colorear header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Colorear filas según significancia
for i, sig in enumerate(significantes):
    color = 'lightgreen' if sig else 'lightcoral'
    for j in range(5):
        table[(i+1, j)].set_facecolor(color)

axes[1, 1].set_title('Resumen de Pruebas Post-Hoc (Tukey HSD)',
                     fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("\nConclusión:")
for res in resultados:
    if res['sig'] == '✓':
        print(f"  {res['par']}: DIFIEREN significativamente (p={res['p']:.4f})")
    else:
        print(f"  {res['par']}: NO difieren (p={res['p']:.4f})")
```

### Prueba de Dunn (no paramétrica)

Alternativa a Tukey cuando no tienes normalidad. Basada en Kruskal-Wallis.

```
Compara rangos medios entre grupos.
Controla FWER como Tukey.
```

## Decisión: ANOVA vs. Kruskal-Wallis

| Criterio | ANOVA | Kruskal-Wallis |
|----------|-------|----------------|
| Supuesto | Normalidad | Ninguno |
| Datos | Continuos | Ordinales o continuos |
| Tamaño n | Cualquiera | Mejor > 5 por grupo |
| Poder | Mayor si normal | Menor pero robusto |
| Post-hoc | Tukey | Dunn |

En tu proyecto:
- Si iteraciones son normales: ANOVA
- Si no normales o n pequeño: Kruskal-Wallis

## Correcciones en Contexto

### Estrategia Conservadora (muchas pruebas exploratorias)

Especifica **una prueba primaria**:
```
Hipótesis primaria: Validez difiere entre baseline y restricciones
Prueba: t-test pareado, α = 0.05
```

Pruebas secundarias sin corrección, reportadas como exploratorias:
```
"Como análisis exploratorio (no ajustado por comparaciones múltiples):
- Iteraciones: p = 0.023
- Tiempo: p = 0.087
- Memoria: p = 0.301
```

Los lectores entienden que estos son menos confiables.

### Estrategia Balanceada (múltiples hipótesis pre-registradas)

Pre-registra todas las comparaciones planeadas, usa Holm:
```
Hipótesis primarias:
1. Validez
2. Iteraciones
3. Tiempo

Corrección: Holm para k=3
α_adjusted para primero: 0.05/3 = 0.0167
```

## Tasa de Descubrimiento Falso (FDR)

Alternativa a FWER. **Más liberal**, útil en exploración.

```
FDR = Proporción esperada de falsos positivos entre descubrimientos

Método Benjamini-Hochberg:
1. Ordena p-valores: p₁ ≤ p₂ ≤ ... ≤ pₖ
2. Encuentra mayor i donde: pᵢ ≤ (i/k) × α
3. Rechaza todas las pruebas ≤ i

Ejemplo k=10, α=0.05:
p₅ = 0.021 ≤ (5/10)×0.05 = 0.025 ✓
p₆ = 0.028 > (6/10)×0.05 = 0.030 ✗

Rechaza hipótesis 1-5, no 6-10
```

**Interpretación**: Esperas que ~5% de tus "significantes" sean falsos positivos.

Útil para descubrimiento exploratorio, menos para confirmación.

## Resumen: Cuándo Usar Qué

```{admonition} Resumen
:class: important
**Checklist para comparaciones múltiples:**
- [ ] **¿Cuántas pruebas hago?** Si k > 1, necesito corrección
- [ ] **¿Una hipótesis principal?** Sin corrección para ella, corrección para exploratorias
- [ ] **¿2-5 comparaciones planeadas?** Usar Holm (menos conservador que Bonferroni)
- [ ] **¿Muchas comparaciones (>10)?** Usar Bonferroni o FDR
- [ ] **¿Comparando 3+ grupos?** Primero ANOVA, luego post-hoc (Tukey HSD)
- [ ] **¿Exploratorio?** Usar FDR, etiquetar claramente como exploratorio
- [ ] **Pre-registro**: Especifica qué pruebas son confirmatorias vs exploratorias
```

```
¿Cuántos grupos comparo?
├─ Dos → t-test (o Mann-Whitney U)
├─ Tres+
│  ├─ ¿Normalidad? → ANOVA
│  └─ No-normal → Kruskal-Wallis
│
¿Cuántas comparaciones hago?
├─ Una (la que me importa) → Sin corrección
├─ Pocas (2-5) → Holm
├─ Muchas (>10) → Bonferroni, FDR, o pre-registro
└─ Exploratorio → FDR
```

## Ejemplo Completo

Proyecto comparando 4 métodos de generación (Baseline A, B, Restricciones C, D):

```
Paso 1: ANOVA
  H₀: μₐ = μᵦ = μc = μd
  F(3,76) = 8.4, p = 0.0002 ✓ Significante

Paso 2: Pruebas Post-Hoc (Tukey)
  A vs. B: p = 0.89 (no diferencia)
  A vs. C: p < 0.001 ✓ Restricciones mejor
  A vs. D: p < 0.001 ✓ Restricciones mejor
  B vs. C: p = 0.006 ✓ Restricciones mejor
  B vs. D: p = 0.004 ✓ Restricciones mejor
  C vs. D: p = 0.32 (ambas restricciones similares)

Conclusión:
- Métodos A y B (baseline) no difieren entre sí
- Métodos C y D (restricciones) no difieren entre sí
- Ambas variantes de restricciones mejoran baseline
```

## Ejercicios y Reflexión

```{admonition} ✅ Verifica tu comprensión
:class: note
1. **FWER**: Si haces 10 pruebas con α=0.05, ¿cuál es la probabilidad de AL MENOS un falso positivo sin corrección?
2. **Bonferroni vs Holm**: ¿Por qué Holm es menos conservador que Bonferroni pero igual de válido?
3. **ANOVA**: ¿Por qué no debes hacer múltiples t-tests cuando comparas 4 grupos?
4. **Post-hoc**: Si ANOVA no es significante (p=0.12), ¿deberías hacer Tukey HSD de todos modos?
```

### Ejercicio 1: FWER sin Corrección
Si haces 15 pruebas independientes con α=0.05:
- ¿Cuál es FWER sin corrección?
- ¿Cuántos falsos positivos esperas?
- ¿Cuál es α_ajustado con Bonferroni?

### Ejercicio 2: Elegir Corrección
Para cada escenario, elige Bonferroni, Holm, Tukey, o FDR:
1. Tienes 1 hipótesis principal, 5 secundarias exploratorias
2. Comparas 6 métodos diferentes (15 pares)
3. Exploras 100 variables potencialmente correlacionadas
4. Tienes 3 hipótesis pre-registradas

### Ejercicio 3: ANOVA
Datos de 3 métodos (10 réplicas cada uno):
Método A: 45, 44, 46, 43, 47, 44, 45, 46, 44, 45 (M=44.9)
Método B: 40, 41, 39, 42, 40, 41, 39, 40, 41, 40 (M=40.3)
Método C: 50, 51, 49, 52, 50, 51, 49, 50, 51, 50 (M=50.3)

Corre ANOVA manualmente o en Python. ¿Significante? Si sí, corre Tukey.

### Ejercicio 4: En Tu Proyecto
¿Cuántas comparaciones planeadas vas a hacer? Propón estrategia:
- ¿Una prueba primaria o múltiples?
- ¿Qué corrección usarías?
- ¿Qué pre-registrarías vs. explorarías?

### Reflexión
1. **Conservadurismo**: ¿Por qué es mejor ser conservador con comparaciones múltiples en primera instancia?
2. **Exploración**: ¿Cómo reportarías análisis exploratorio de forma honesta sin engañar?
3. **Diseño**: ¿Cómo evitas comparaciones múltiples diseñando mejor tu experimento desde el inicio?

---

**Próxima semana**: Aprenderemos a registrar y visualizar experimentos con herramientas modernas de MLOps.

---

## Referencias

- Bonferroni, C. (1936). Teoria Statistica delle Classi e Calcolo delle Probabilità. Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze.
- Holm, S. (1979). [A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733). Scandinavian Journal of Statistics.
- Benjamini, Y. & Hochberg, Y. (1995). [Controlling the False Discovery Rate](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). JRSS-B.
