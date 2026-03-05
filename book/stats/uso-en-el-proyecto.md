# Guía de Integración: Estadística ↔ Proyecto ACA-2026

> **Propósito:** Esta guía te ayuda a saber *cuándo* y *cómo* usar cada herramienta estadística del curso en las diferentes etapas de tu proyecto de investigación sobre generación de kernels GPU.

---

## ¿Cuándo necesito estadística?

```
Etapa del Proyecto              → Herramienta Estadística
─────────────────────────────────────────────────────────
Primeros resultados del baseline  → L01: Probabilidad (PMF, E[X], Var)
Comparar baseline vs tu método   → L03: Prueba t o Wilcoxon (muestras ind.)
Múltiples configuraciones         → L06: ANOVA o Kruskal-Wallis
Reportar efecto práctico          → L08: Cohen's d, tamaño de efecto
Generar figuras para el reporte   → L09: Visualización y MLOps
Análisis exploratorio inicial     → L02: Distribuciones y estadística desc.
```

---

## Tabla de Integración Completa

| Cuando necesites… | Lección de Stats | Conceptos clave | Momento en el proyecto |
|---|---|---|---|
| Describir tus datos experimentales | **L02:** Estadística Descriptiva | Media, mediana, IQR, boxplot | Al tener los primeros datos del baseline |
| Reportar tasa de éxito / fracaso | **L01:** Probabilidad | PMF, E[X], intervalo de confianza | Al correr el baseline las primeras veces |
| Demostrar que tu método es mejor | **L03:** Pruebas de Hipótesis | t-test, valor-p, H₀ vs H₁ | Al completar experimento de comparación |
| Comparar >2 configuraciones | **L06:** Comparaciones Múltiples | ANOVA, Kruskal-Wallis, post-hoc | Al explorar hiperparámetros |
| Cuantificar cuánto mejor | **L08:** Tamaño de Efecto | Cohen's d, odds ratio | En la sección Resultados del reporte |
| Evaluar correlación entre variables | **L05:** Correlación y Regresión | Pearson, Spearman, regresión lineal | Al analizar qué afecta la calidad del kernel |
| Visualizar distribuciones | **L09:** Visualización | histograma, boxplot, violin | En cualquier etapa, especialmente en Resultados |

---

## ¿Qué prueba estadística debo usar?

```
¿Cuántos grupos estoy comparando?
│
├─ 2 grupos
│   ├─ ¿Datos normales? (prueba Shapiro-Wilk)
│   │   ├─ Sí → t-test de Student  (L03)
│   │   └─ No → Prueba de Wilcoxon (L03)
│   └─ ¿Muestras relacionadas (mismo baseline, distintas config)?
│       ├─ Normales   → t-test pareado   (L03)
│       └─ No normales → Wilcoxon pareado (L03)
│
└─ >2 grupos
    ├─ ¿Datos normales?
    │   ├─ Sí → ANOVA de una vía (L06)
    │   └─ No → Kruskal-Wallis   (L06)
    └─ ¿Post-hoc (cuáles grupos difieren)?
        ├─ Normal     → Tukey HSD (L06)
        └─ No normal  → Dunn + Bonferroni (L06)
```

---

## Aplicación Concreta al Proyecto

### Escenario 1: Comparar baseline vs método con restricciones

```python
from scipy import stats
import numpy as np

# Resultados: número de kernels válidos por ejecución
baseline     = np.array([72, 68, 75, 70, 73, 69, 71, 74, 70, 72])  # 10 corridas
con_restr    = np.array([85, 88, 82, 86, 89, 84, 87, 83, 88, 86])  # 10 corridas

# 1. Verificar normalidad
_, p_norm_b = stats.shapiro(baseline)
_, p_norm_r = stats.shapiro(con_restr)
print(f"Shapiro-Wilk baseline p={p_norm_b:.3f}, con_restr p={p_norm_r:.3f}")

# 2. Si normales → t-test; si no → Wilcoxon
if p_norm_b > 0.05 and p_norm_r > 0.05:
    stat, p_val = stats.ttest_ind(baseline, con_restr)
    test_name = "t-test"
else:
    stat, p_val = stats.mannwhitneyu(baseline, con_restr, alternative='less')
    test_name = "Mann-Whitney U"

print(f"\n{test_name}: estadístico={stat:.3f}, p-valor={p_val:.4f}")
print(f"¿Diferencia significativa (α=0.05)? {'Sí' if p_val < 0.05 else 'No'}")

# 3. Tamaño de efecto (Cohen's d)
pooled_std = np.sqrt((np.std(baseline, ddof=1)**2 + np.std(con_restr, ddof=1)**2) / 2)
cohens_d = (np.mean(con_restr) - np.mean(baseline)) / pooled_std
print(f"Cohen's d = {cohens_d:.3f} ({'grande' if abs(cohens_d)>0.8 else 'mediano' if abs(cohens_d)>0.5 else 'pequeño'})")
```

### Escenario 2: Comparar múltiples beam_widths (1, 3, 5, 10)

```python
# Usar ANOVA si normales, Kruskal-Wallis si no
bw1  = [72, 70, 73, 71, 72]
bw3  = [82, 84, 81, 83, 82]
bw5  = [85, 87, 84, 86, 85]
bw10 = [86, 88, 85, 87, 86]

# Kruskal-Wallis (no asume normalidad, recomendado para muestras pequeñas)
stat, p_val = stats.kruskal(bw1, bw3, bw5, bw10)
print(f"Kruskal-Wallis: H={stat:.3f}, p={p_val:.4f}")
```

---

## Checklist para la Sección de Resultados del Reporte

- [ ] Reportar media ± desviación estándar para cada condición
- [ ] Especificar cuántas corridas por condición
- [ ] Incluir la prueba de normalidad y el test estadístico elegido
- [ ] Reportar el valor-p y el tamaño de efecto
- [ ] Incluir al menos un gráfico (boxplot o violin) para cada comparación
- [ ] Interpretar los resultados en términos del dominio (¿qué significa que sea mejor en X%?)

---

## Referencias Cruzadas

- **L01** (Probabilidad) → Fundamentos teóricos para entender distribuciones
- **L03** (Hipótesis) → Pruebas t, Wilcoxon, intervalos de confianza
- **L06** (Múltiples) → ANOVA, Kruskal-Wallis para comparaciones múltiples
- **L08** (Efecto) → Cohen's d, odds ratio, relevancia práctica
- **Project_1 L07** → Experimentos de evaluación de gramáticas Triton
- **Project_1 L09** → Token economics y comparación de métodos

---

*Este documento es parte del módulo Stats del curso ACA-2026.*
