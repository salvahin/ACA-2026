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

# Pruebas No Paramétricas


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/07-pruebas-noparametricas.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Determinar cuándo usar pruebas paramétricas vs no paramétricas
- Aplicar la prueba de Shapiro-Wilk para verificar normalidad
- Ejecutar e interpretar Mann-Whitney U, Wilcoxon y Kruskal-Wallis
- Calcular tamaños de efecto para pruebas no paramétricas
- Elegir la prueba correcta según las características de tus datos
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[StatQuest: Non-Parametric Tests](https://www.youtube.com/watch?v=GrLsW-gABm8)** - Explicación visual de cuándo y por qué usar pruebas no paramétricas como Mann-Whitney U y Wilcoxon, con ejemplos prácticos.
```

## Semana 6 - Estadística para Generación de Kernels GPU

Hasta ahora hemos asumido normalidad: que nuestros datos siguen aproximadamente una distribución normal. Pero, ¿qué pasa si no es así? Aquí es donde las **pruebas no paramétricas** entran al rescate.

## Cuándo Usar Pruebas No Paramétricas

:::{figure} diagrams/parametric_vs_nonparametric.png
:name: fig-parametric-vs-nonparametric
:alt: Comparación de pruebas paramétricas y no paramétricas
:align: center
:width: 90%

**Figura 8:** Comparación entre pruebas paramétricas (asumen distribución normal) y no paramétricas (robustas a distribuciones arbitrarias).
:::

Usa pruebas no paramétricas cuando:

1. **Datos claramente no normales**: Sesgados, multimodales, con colas pesadas
2. **Tamaño muestral pequeño (n < 30)** y no puedes asumir normalidad
3. **Datos ordinales**: Escalas como "muy pobre, pobre, neutral, bueno, muy bueno"
4. **Datos con outliers extremos**: Mediano-Wilcoxon es robusto a atípicos
5. **Violaciones de supuestos**: Varianzas muy diferentes entre grupos

### Ley de Conservadurismo

Si dudas, usa pruebas no paramétricas. Son casi tan poderosas como t-tests cuando normalidad se cumple, pero mucho más robustas cuando no.

En tu proyecto, podrías tener:
- Distribución sesgada de iteraciones (muchos casos rápidos, algunos muy lentos)
- Pequeño n de kernels únicos si ejecutas pocos
- Datos ordinales si clasificas kernels como "inválido, compilable, eficiente, altamente optimizado"

## Prueba de Normalidad: Shapiro-Wilk

Antes de decidir, **prueba si tus datos son normales**.

La **prueba de Shapiro-Wilk** testea:
```
H₀: Los datos provienen de una distribución normal
H₁: Los datos no provienen de una distribución normal
```

Devuelve un p-valor:
- Si p > 0.05: No hay evidencia contra normalidad (asume normalidad)
- Si p < 0.05: Significante evidencia de no-normalidad (usa no-paramétrica)

### Ejemplo en Python

```{code-cell} ipython3
from scipy.stats import shapiro

# Tu data
iteraciones = [42, 43, 41, 45, 43, 42, 88, 44, 41, 43]

statistic, p_value = shapiro(iteraciones)
print(f"p-valor: {p_value}")

if p_value > 0.05:
    print("Asume normalidad, usa t-test")
else:
    print("No es normal, usa Wilcoxon")
```

## Alternativas No Paramétricas

### Mann-Whitney U (para dos grupos independientes)

```{admonition} 💡 Intuición
:class: hint
En lugar de comparar medias (que son sensibles a outliers), Mann-Whitney U compara **rangos**. Piensa en dos equipos: ordenas todos los tiempos del más rápido al más lento. Si un equipo domina los primeros lugares, sus rangos serán menores. Es como una competencia de carreras donde importa la posición, no el tiempo exacto.
```

**Alternativa a**: t-test de dos muestras
**Supuestos**: Solo que los datos sean ordinales/continuos

En lugar de comparar **medias**, compara **rangos**.

```
Datos:
Baseline:      4.1, 4.3, 3.9, 4.2, 4.4
Restricciones: 3.2, 3.1, 3.5, 3.0, 3.3

Rangos combinados (de menor a mayor):
3.0(1) 3.1(2) 3.2(3) 3.3(4) 3.5(5) 4.1(6) 4.2(7) 4.3(8) 4.4(9)
                ↑ Restricciones         ↑ Baseline

Suma de rangos Restricciones: 1+2+3+4+5 = 15
Suma de rangos Baseline: 6+7+8+9 = 30

Si los grupos fueran idénticos, esperaríamos rangos iguales.
```

```{admonition} 📊 Cómo interpretar
:class: note
**Si p < 0.05**, entonces:
- Los dos grupos tienen distribuciones significativamente diferentes
- Uno tiende a tener valores mayores/menores que el otro
- Reporta: "Mann-Whitney U = 12, p = 0.001, r = 0.76 (efecto grande)"

**Si p ≥ 0.05**, entonces:
- No hay evidencia suficiente de diferencia entre grupos
- Las distribuciones son razonablemente similares
- Reporta: "No se encontró diferencia significativa (U = 45, p = 0.23)"

**Reporta también**: Medianas de cada grupo, no medias
```

Devuelve un **U-statistic** y p-valor.

Intuitivamente: ¿Las distribuciones son diferentes (uno típicamente mayor)?

### Prueba de Wilcoxon de Rangos Pareados

**Alternativa a**: t-test pareado
**Supuestos**: Datos pareados ordinales/continuos

Útil cuando tienes **pares** (ej. antes-después, baseline vs. restricción para el mismo kernel).

```
Kernel | Baseline | Restricciones | Diferencia | Rango
-------|----------|---------------|-----------|---------
1      | 4.1      | 3.9           | 0.2       | 1
2      | 4.3      | 3.8           | 0.5       | 3
3      | 3.9      | 3.7           | 0.2       | 1
4      | 4.2      | 4.1           | 0.1       | 0.5
5      | 4.4      | 3.5           | 0.9       | 5
...    | ...      | ...           | ...       | ...

Suma de rangos (positivos): 1+3+1+0.5+5+... = 25.5
Suma de rangos (negativos): 0

Si no hay diferencia, esperamos suma similar.
```

### Prueba de Kruskal-Wallis (para múltiples grupos)

**Alternativa a**: ANOVA de una vía
**Supuestos**: Solo ordinalidad

Compara k > 2 grupos usando rangos.

```
Grupo A (Baseline):       4.1, 4.3, 4.2
Grupo B (Restricciones):  3.8, 3.9, 3.7
Grupo C (Temperatura 0.5): 4.5, 4.6, 4.4

Rangos combinados, calcula H-statistic.
Si H es grande, grupos difieren significativamente.
```

## Efecto de Tamaño en Pruebas No Paramétricas

Así como reportas Cohen's d para t-tests, reporta medidas de efecto para pruebas no paramétricas:

### Rank-Biserial Correlation (para Mann-Whitney U)

```
r = 1 - (2U) / (n₁ × n₂)

Rango: -1 a +1
Interpretación: -0.3 a 0.3 = pequeño, 0.3 a 0.5 = mediano, > 0.5 = grande
```

### Efecto Tamaño para Wilcoxon

```
r = Z / √N

Donde Z viene del test, N = total de pares
```

## Ventajas y Desventajas

### Ventajas de Pruebas No Paramétricas

- **Robustas**: No asumen distribución específica
- **Resisten outliers**: Basadas en rangos, no valores
- **Aplicables a ordinales**: Escalas de Likert, rankings
- **Válidas para pequeños n**: Aunque poder disminuye

### Desventajas

- **Menor poder**: Si normalidad se cumple, son menos sensibles
- **Menos interpretables**: Compara distribuciones, no medias específicamente
- **Múltiples comparaciones complejas**: ANOVA tiene muchas extensiones, Kruskal-Wallis tiene pocas

## Flujo de Decisión

```{admonition} Resumen
:class: important
**Checklist para elegir prueba estadística:**
- [ ] **Paso 1**: ¿Son datos ordinales (rankings)? → Usa no-paramétrica
- [ ] **Paso 2**: ¿n < 30 en algún grupo? → Considera no-paramétrica
- [ ] **Paso 3**: Ejecutar Shapiro-Wilk: ¿p > 0.05? → Normal, usa paramétrica
- [ ] **Paso 4**: ¿Hay outliers extremos? → Usa no-paramétrica (más robusta)
- [ ] **Paso 5**: ¿Varianzas muy desiguales? → Considera no-paramétrica
- [ ] **Dudas**: Reporta AMBAS y compara conclusiones

**Correspondencias:**
- t-test de dos muestras ↔ Mann-Whitney U
- t-test pareado ↔ Wilcoxon signed-rank
- ANOVA ↔ Kruskal-Wallis
```

```
¿Tienes datos ordinales (rankings, escalas)?
├─ Sí → Usa no-paramétrica
└─ No → ¿Distribución normal? (Shapiro-Wilk p > 0.05)
    ├─ Sí → t-test es aceptable
    ├─ No → Compara n vs. violaciones de supuestos
    │   ├─ Violaciones leves, n > 30 → t-test (robusto por TCL)
    │   └─ Violaciones severas o n < 30 → Usa no-paramétrica
    └─ Dudas → Reporta ambas (paramétrica + no-paramétrica)
```

## Ejemplo Completo: Comparar Métodos

Supongamos tienes iteraciones de dos métodos:

```
Baseline:      42, 43, 41, 45, 43, 42, 88, 44, 41, 43
Restricciones: 38, 39, 37, 40, 38, 39, 36, 38, 37, 39

Paso 1: Prueba Shapiro-Wilk
  Baseline p = 0.001 (no normal, ese 88 lo arruina)
  Restricciones p = 0.82 (normal)

Decisión: Usa Mann-Whitney U

Paso 2: Corre Mann-Whitney U
  U-statistic = 12, p = 0.001

Paso 3: Efecto tamaño
  r_biserial = 1 - (2×12)/(10×10) = 1 - 0.24 = 0.76 (grande)

Conclusión: Restricciones significativamente mejores (p=0.001),
efecto grande (r=0.76). Baseline tiene algunas iteraciones muy altas,
sugiriendo falta de estabilidad.
```

## Python: Ejemplo Práctico

```{code-cell} ipython3
from scipy.stats import shapiro, mannwhitneyu, wilcoxon
import numpy as np

baseline = np.array([42, 43, 41, 45, 43, 42, 88, 44, 41, 43])
restricciones = np.array([38, 39, 37, 40, 38, 39, 36, 38, 37, 39])

# Prueba normalidad
stat_b, p_b = shapiro(baseline)
stat_r, p_r = shapiro(restricciones)
print(f"Baseline normal? p = {p_b:.4f}")
print(f"Restricciones normal? p = {p_r:.4f}")

# Mann-Whitney U (independientes)
u_stat, u_p = mannwhitneyu(baseline, restricciones)
print(f"\nMann-Whitney U: U = {u_stat}, p = {u_p:.4f}")

# Efecto tamaño
n1, n2 = len(baseline), len(restricciones)
r_biserial = 1 - (2 * u_stat) / (n1 * n2)
print(f"Efecto tamaño: r = {r_biserial:.3f}")

# Si fueran pareados (mismo kernel ambos métodos)
diffs = baseline - restricciones
w_stat, w_p = wilcoxon(diffs)
print(f"\nWilcoxon (pareado): W = {w_stat}, p = {w_p:.4f}")
```

## Casos en tu Proyecto

### Caso 1: Distribución Sesgada

Si tus iteraciones tienen muchos valores pequeños (rápida convergencia) y pocos valores muy grandes (excepciones), entonces:
- Shapiro-Wilk probablemente significante
- Mann-Whitney U es más apropiado
- Reporta mediana, no media

### Caso 2: Datos Ordinales

Si clasificas kernels como:
```
1 = Inválido
2 = Válido pero lento
3 = Válido y razonable
4 = Válido y optimizado
```

Entonces debes usar:
- Kruskal-Wallis para comparar grupos
- Mediano, no media
- Rank-biserial para efecto tamaño

## Ejercicios y Reflexión

```{admonition} ✅ Verifica tu comprensión
:class: note
1. **Cuándo usar**: ¿Cuándo usarías Mann-Whitney U vs t-test de dos muestras?
2. **Interpretación**: Si Shapiro-Wilk da p = 0.03, ¿qué significa esto sobre tus datos?
3. **Robustez**: ¿Por qué Mann-Whitney U es más robusto a outliers que t-test?
4. **Reportar**: Si usas prueba de Wilcoxon, ¿reportas media o mediana?
```

### Ejercicio 1: Prueba de Normalidad
Para datos en tu proyecto:
- Ejecuta Shapiro-Wilk en: iteraciones, tiempos, tasa de validez
- ¿Cuáles son normales, cuáles no?
- ¿Por qué crees que unos sí y otros no?

### Ejercicio 2: Elegir Prueba Estadística
Para cada escenario, elige: t-test, Mann-Whitney U, Wilcoxon, o Kruskal-Wallis
1. Comparar tasa de validez (%) entre baseline y 3 métodos
2. Comparar iteraciones para el mismo kernel, baseline vs. restricciones
3. Comparar tiempos (segundos) entre 2 grupos de kernels grandes
4. Comparar ranking de calidad (1-5 escala) entre 4 métodos

### Ejercicio 3: Implementación
Corre código Python (arriba) con tus datos:
- Interpreta output Shapiro-Wilk
- Reporta p-valor y efecto tamaño de Mann-Whitney U
- Compara conclusión vs. si usaras t-test

### Ejercicio 4: Robustez
Toma datos con outliers extremos:
```
baseline = [40, 42, 41, 43, 1000]  # ese 1000 es error
```
Corre tanto t-test como Mann-Whitney U. ¿Cómo difieren conclusiones?

### Reflexión
1. **Robustez**: En tu proyecto, ¿esperas datos limpios o con outliers? ¿Por qué es importante robustecer?
2. **Poder vs. Robustez**: Si datos son normales pero casi siempre usas Mann-Whitney, ¿qué pierdes?
3. **Comunicación**: Si reportas Mann-Whitney U a un comité, ¿cómo explicarías por qué vs. t-test?

---

**Próxima semana**: Aprenderemos por qué simplemente reportar p-valores es insuficiente; necesitamos reportar tamaños de efecto.

## Errores Comunes

```{admonition} ⚠️ Trampas de pruebas no paramétricas
:class: warning

1. **Usar no paramétricas "por si acaso"**: Si tus datos son normales, t-test tiene más poder estadístico.
2. **Ignorar la prueba de normalidad**: Siempre ejecuta Shapiro-Wilk antes de decidir.
3. **Confundir mediana con media**: Mann-Whitney compara distribuciones, no específicamente medias.
```

## Ejercicio Práctico: Elegir Prueba Correcta

```{code-cell} ipython3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generar datos sesgados (tiempos de ejecución con outliers)
np.random.seed(42)
baseline = np.concatenate([
    np.random.gamma(2, 2, 18),  # La mayoría de datos
    [25, 30]  # Dos outliers extremos
])

restricciones = np.random.gamma(2, 1.5, 20)

print("=== Paso 1: Inspección Visual ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(baseline, bins=10, alpha=0.6, label='Baseline', color='steelblue')
ax1.hist(restricciones, bins=10, alpha=0.6, label='Restricciones', color='coral')
ax1.set_xlabel('Tiempo (ms)')
ax1.set_ylabel('Frecuencia')
ax1.legend()
ax1.set_title('Distribuciones (nótese outliers)')

ax2.boxplot([baseline, restricciones], labels=['Baseline', 'Restricciones'])
ax2.set_ylabel('Tiempo (ms)')
ax2.set_title('Box Plot')
plt.tight_layout()

# Paso 2: Prueba de normalidad
print("\n=== Paso 2: Prueba de Normalidad (Shapiro-Wilk) ===")
stat_b, p_b = stats.shapiro(baseline)
stat_r, p_r = stats.shapiro(restricciones)
print(f"Baseline: W={stat_b:.4f}, p={p_b:.4f}")
print(f"Restricciones: W={stat_r:.4f}, p={p_r:.4f}")

if p_b < 0.05:
    print("  ⚠ Baseline NO es normal (p<0.05)")
else:
    print("  ✓ Baseline es normal (p>0.05)")

if p_r < 0.05:
    print("  ⚠ Restricciones NO es normal (p<0.05)")
else:
    print("  ✓ Restricciones es normal (p>0.05)")

# Paso 3: Decisión
print("\n=== Paso 3: Decisión ===")
if p_b < 0.05 or p_r < 0.05:
    print("DECISIÓN: Usar Mann-Whitney U (no paramétrico)")
    print("RAZÓN: Al menos un grupo viola normalidad\n")

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(baseline, restricciones, alternative='two-sided')
    print(f"Mann-Whitney U: U={u_stat:.2f}, p={u_p:.4f}")

    # Efecto tamaño
    n1, n2 = len(baseline), len(restricciones)
    r_biserial = 1 - (2 * u_stat) / (n1 * n2)
    print(f"Efecto tamaño (r): {r_biserial:.3f}")

    # Comparar con t-test (para ilustración)
    t_stat, t_p = stats.ttest_ind(baseline, restricciones)
    print(f"\n[Solo comparación] t-test: t={t_stat:.2f}, p={t_p:.4f}")
    print(f"Diferencia en p-valor: {abs(u_p - t_p):.4f}")

else:
    print("DECISIÓN: t-test es apropiado")
    t_stat, t_p = stats.ttest_ind(baseline, restricciones)
    print(f"t-test: t={t_stat:.2f}, p={t_p:.4f}")

# Paso 4: Interpretación
print("\n=== Paso 4: Interpretación ===")
print(f"Mediana Baseline: {np.median(baseline):.2f} ms")
print(f"Mediana Restricciones: {np.median(restricciones):.2f} ms")
print(f"Diferencia: {np.median(baseline) - np.median(restricciones):.2f} ms")

if u_p < 0.05:
    print("\n✓ Diferencia estadísticamente significativa (p<0.05)")
    print(f"  Las restricciones REDUCEN el tiempo de ejecución")
else:
    print("\n✗ No hay evidencia suficiente de diferencia (p≥0.05)")
```

---

## Referencias

- Mann, H. & Whitney, D. (1947). [On a Test of Whether One of Two Random Variables is Stochastically Larger](https://doi.org/10.1214/aoms/1177730491). Annals of Mathematical Statistics.
- Wilcoxon, F. (1945). [Individual Comparisons by Ranking Methods](https://doi.org/10.2307/3001968). Biometrics Bulletin.
- Kruskal, W. & Wallis, W. (1952). [Use of Ranks in One-Criterion Variance Analysis](https://doi.org/10.1080/01621459.1952.10483441). JASA.
- Shapiro, S. & Wilk, M. (1965). [An Analysis of Variance Test for Normality](https://doi.org/10.1093/biomet/52.3-4.591). Biometrika.
