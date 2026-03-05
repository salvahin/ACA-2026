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

# Tamaño del Efecto

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/08-tamano-efecto.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Calcular e interpretar Cohen's d para diferencias de medias
- Usar intervalos de confianza para cuantificar incertidumbre
- Aplicar bootstrap para estimar ICs cuando no hay normalidad
- Distinguir entre significancia estadística y práctica
- Reportar resultados completos (estadístico, p-valor, efecto, IC)
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[StatQuest: Effect Size](https://www.youtube.com/watch?v=fJPZTdVk7lk)** - Explicación clara de por qué el tamaño del efecto es tan importante como el p-valor, y cómo calcular e interpretar Cohen's d correctamente.
```

## Semana 7 - Estadística para Generación de Kernels GPU

Aquí es donde pasamos de "¿hay diferencia?" a "¿qué tan grande es la diferencia y le importa?" Un p-valor pequeño significa que una diferencia es estadísticamente significante, pero no te dice si es **prácticamente importante**.

## Por Qué el Tamaño del Efecto Importa

Imagina dos escenarios con el mismo p-valor:

**Escenario A**: n = 100,000
- Baseline: x̄ = 5.0s
- Restricciones: x̄ = 5.01s
- t-test: p = 0.001 (¡significante!)
- **Diferencia real**: 0.01 segundos (imperceptible)

**Escenario B**: n = 20
- Baseline: x̄ = 5.0s
- Restricciones: x̄ = 5.5s
- t-test: p = 0.04 (¡significante!)
- **Diferencia real**: 0.5 segundos (noticeable)

Ambos tienen p < 0.05, pero las conclusiones prácticas son opuestas.

## p-valores vs. Tamaño del Efecto

| Aspecto | p-valor | Tamaño del efecto |
|---------|---------|------------------|
| **Mide** | Probabilidad de datos | Magnitud de diferencia |
| **Sensible a** | Tamaño muestral | Efecto real |
| **Determina por** | n y efecto | Solo el efecto |
| **Utilidad** | Rechaza H₀ | Cuán importante es |

**La práctica moderna**: Reporta AMBOS. Nunca solo p-valor.

## Cohen's d (Diferencias entre Medias)

```{admonition} 💡 Intuición
:class: hint
Cohen's d te dice "¿cuántas desviaciones estándar de diferencia hay entre los grupos?" Si d = 0.5, significa que los grupos difieren por media desviación estándar. Es como medir la distancia entre dos distribuciones usando su propia variabilidad como regla.
```

La métrica estándar para comparar dos grupos:

```
Cohen's d = (μ₁ - μ₂) / σ_pooled

Donde:
σ_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]
```

### Interpretación

- d = 0.2: Efecto **pequeño** (noticeable solo con atención)
- d = 0.5: Efecto **mediano** (fácilmente noticeable)
- d = 0.8: Efecto **grande** (muy importante)
- d > 1.2: Efecto **muy grande** (obvio a todos)

```{code-cell} ipython3
# Visualización de Cohen's d
import numpy as np
import plotly.graph_objects as go
from scipy import stats

x = np.linspace(-4, 6, 500)

fig = go.Figure()

# Grupo control (siempre en 0)
fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, 0, 1),
                         fill='tozeroy', name='Control', fillcolor='rgba(70,130,180,0.3)'))

# Diferentes tamaños de efecto
effects = [(0.2, 'Pequeño (d=0.2)', 'green'),
           (0.5, 'Mediano (d=0.5)', 'orange'),
           (0.8, 'Grande (d=0.8)', 'red')]

for d, label, color in effects:
    fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, d, 1),
                             name=label, line=dict(color=color, width=2)))

fig.update_layout(
    title="Visualización de Cohen's d: Overlap entre grupos",
    xaxis_title="Valor", yaxis_title="Densidad",
    height=400
)
fig.show()
```

```{admonition} 🎮 Visualizador Interactivo de Cohen's d
:class: tip

<iframe src="https://rpsychologist.com/cohend/" width="100%" height="600px" style="border:1px solid #ddd; border-radius:8px;"></iframe>
```

:::{figure} diagrams/cohens_d_visualization.png
:name: fig-cohens-d-visualization
:alt: Visualización de Cohen's d mostrando magnitudes de efectos
:align: center
:width: 90%

**Figura 9:** Visualización de Cohen's d: diferentes tamaños de efecto y su interpretación en términos de solapamiento entre distribuciones.
:::

```{admonition} 🎯 Aplicación en ML
:class: important
En el contexto de evaluar kernels GPU, Cohen's d te permite cuantificar mejoras:
- **d=0.2-0.3** (~5% mejora): Marginal, quizá no vale la pena implementar
- **d=0.5-0.7** (~15%): Optimización típica, justifica el esfuerzo
- **d=0.8-1.2** (~30%): Cambio arquitectónico significativo
- **d>1.5** (>50%): Innovación sustancial, publicable

**Ejemplo**: Si baseline tarda 5.2s y restricciones 4.8s con d=0.52, esto es mediano-importante.
```

### Ejemplo en Tu Proyecto

Baseline tiempos: n₁=30, x̄₁=5.2s, s₁=0.8s
Restricciones tiempos: n₂=30, x̄₂=4.8s, s₂=0.9s

```
σ_pooled = √[((29×0.64 + 29×0.81) / 58)]
         = √[(18.56 + 23.49) / 58]
         = √0.727
         = 0.853

d = (5.2 - 4.8) / 0.853
  = 0.4 / 0.853
  = 0.47

Interpretación: Efecto mediano. Las restricciones ahorran ~0.4s,
lo que es prácticamente importante.
```

### Cuándo el Efecto Importa Prácticamente

La importancia depende del contexto:

**Compilación de kernel (esperado < 1s)**:
- d = 0.4 es importante (40% del tiempo típico)

**Entrenar red neuronal (esperado 10 horas)**:
- d = 0.4 es trivial (solo ~50 minutos ahorrados)

Necesitas juicio para definir "prácticamente importante". En tu propuesta de tesis, especifica:

> "Consideramos un tamaño de efecto de d ≥ 0.5 como prácticamente importante, basado en [justificación]."

## Intervalos de Confianza (CI)

```{admonition} 💡 Intuición
:class: hint
Un intervalo de confianza te dice: "Si repitiera este experimento muchas veces, en 95% de esas repeticiones el verdadero valor estaría dentro de este rango." Es como un margen de error: no solo reportas un número, sino qué tan seguro estás de ese número.
```

En lugar de solo reportar un punto (media), reporta un rango: **Intervalo de Confianza de 95%**.

```
IC 95%: [μ_bajo, μ_alto]

Interpretación: "Confiamos 95% que el verdadero parámetro está en este rango."

No: "Hay 95% de probabilidad de que esté en este rango" (incorrecto)
```

### Cálculo

```
IC 95% = x̄ ± (1.96 × SE)

Donde SE = s / √n
```

```{admonition} 📊 Cómo interpretar
:class: note
**Ejemplo**: x̄ = 4.8s, IC 95%: [4.51, 5.09]

**Si los ICs de dos grupos NO se solapan**:
- Hay diferencia significativa
- Ejemplo: Grupo A [4.5, 5.0], Grupo B [5.2, 5.8] → Significante

**Si los ICs se solapan**:
- Puede o no haber diferencia (necesitas hacer la prueba formal)
- Ejemplo: Grupo A [4.5, 5.1], Grupo B [4.9, 5.5] → Hacer t-test

**IC que incluye 0 (cuando comparas diferencias)**:
- No hay diferencia significativa
- Ejemplo: Diferencia = 0.3, IC [-0.1, 0.7] → No significante
```

Ejemplo: x̄ = 4.8s, s = 0.8s, n = 30

```
SE = 0.8 / √30 = 0.146
IC = 4.8 ± 1.96×0.146
   = 4.8 ± 0.286
   = [4.51, 5.09]

Reporta: "Tiempo medio = 4.8s (IC 95%: [4.51, 5.09])"
```

### Cómo Usar IC para Comparar

Si los ICs de dos grupos **se solapan**, la diferencia no es significante.
Si los ICs **no se solapan**, la diferencia es significante.

```
Grupo A: [4.51, 5.09]
Grupo B: [4.20, 4.80]

Solapan en [4.51, 4.80], luego no significante.

Grupo C: [4.40, 4.70]
Grupo D: [4.80, 5.20]

No solapan, luego significante.
```

Esto da una visión intuitiva. Si reportas solo p-valor, no ves esto claramente.

## Bootstrap: Estimación No Paramétrica de CI

Cuando no puedes asumir normalidad, usa **bootstrap**:

```
1. Tienes datos originales: [4.1, 4.3, 4.2, 4.4, 3.9]

2. Resamplea con reemplazo, calcula estadístico:
   Bootstrap muestra 1: [4.1, 4.1, 4.3, 3.9, 4.2] → media = 4.12
   Bootstrap muestra 2: [4.4, 4.3, 4.1, 4.1, 4.2] → media = 4.22
   Bootstrap muestra 3: [3.9, 4.2, 4.2, 4.1, 4.4] → media = 4.16
   ... repite 10,000 veces

3. Distribución de las 10,000 medias bootstrap
   Percentil 2.5% = 4.05
   Percentil 97.5% = 4.35
   → IC 95% = [4.05, 4.35]
```

**Ventaja**: No asume normalidad. Funciona para cualquier estadístico.

```{code-cell} ipython3
from scipy.stats import bootstrap
import numpy as np

data = np.array([4.1, 4.3, 4.2, 4.4, 3.9])
def mean_func(x, axis):
    return np.mean(x, axis=axis)

res = bootstrap((data,), mean_func, n_resamples=10000)
print(f"IC 95%: [{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]")
```

## Reportando Resultados Completos

En lugar de esto:

> "Las restricciones mejoraron significativamente el tiempo (t(58)=2.1, p=0.04)."

Reporta:

> "Las restricciones redujeron el tiempo de compilación (M=4.8s, DE=0.9s vs. M=5.2s, DE=0.8s), t(58)=2.1, p=0.04, d=0.47, IC 95% de la diferencia: [0.05, 0.79]."

Esto proporciona:
- **Medias y DEs**: Contexto numérico
- **Estadístico t**: Cómo fue calculado
- **p-valor**: Significancia estadística
- **d**: Tamaño del efecto
- **IC**: Rango plausible de la diferencia

## Comparar Múltiples Efectos

En tu proyecto, medirás múltiples DVs (validez, iteraciones, tiempo).

Para cada una, reporta tamaño del efecto:

```
| DV | Baseline | Restricciones | d | IC 95% | Conclusión |
|----|----------|---------------|---|--------|------------|
| Validez (%) | 75 | 85 | 0.58 | [0.12, 1.04] | Mediano, significante |
| Iteraciones | 45 | 38 | 0.65 | [0.18, 1.12] | Mediano, significante |
| Tiempo (s) | 5.2 | 4.8 | 0.47 | [0.05, 0.79] | Pequeño-mediano, significante |
```

Ahora el lector ve: todas las métricas mejoran, pero validez es el mayor ganador.

## Efecto Tamaño para Proporciones

Si comparas porcentajes/proporciones:

```
p₁ = 0.75 (baseline válido)
p₂ = 0.85 (restricciones válido)

Diferencia relativa = (p₂ - p₁) / p₁ = 0.10 / 0.75 = 13.3%

O usa h de Cohen:
h = 2 × arcsin(√p₂) - 2 × arcsin(√p₁)
  = 2 × arcsin(√0.85) - 2 × arcsin(√0.75)
  ≈ 0.42 (mediano)
```

Reporta ambos:

> "Las restricciones aumentaron tasa de validez de 75% a 85% (aumento relativo 13%), h=0.42."

## Número Necesario a Tratar (NNT)

Para datos binarios, NNT dice cuántos casos necesitas tratar para ver un beneficio:

```
NNT = 1 / (p₂ - p₁)
    = 1 / (0.85 - 0.75)
    = 1 / 0.10
    = 10
```

**Interpretación**: Necesitas usar restricciones en 10 kernels para que 1 adicional compile exitosamente.

Útil para contexto: ¿Vale la pena el esfuerzo de implementar restricciones para 1 mejora por cada 10 intentos?

## Ejercicios y Reflexión

```{admonition} ✅ Verifica tu comprensión
:class: note
1. **P-valor vs Tamaño**: ¿Por qué p < 0.001 no significa automáticamente que el efecto sea importante?
2. **Cohen's d**: Si d = 0.8, ¿significa que un grupo es "80% mejor"? ¿Qué significa realmente?
3. **Intervalos de Confianza**: Si IC 95% de diferencia es [0.05, 0.79], ¿qué puedes concluir?
4. **Práctica**: ¿Cuándo un efecto puede ser estadísticamente significante pero prácticamente trivial?
```

### Ejercicio 1: Calcular Cohen's d
Datos:
- Método A: n=25, x̄=42, s=8
- Método B: n=25, x̄=38, s=9

Calcula:
a) Cohen's d (step-by-step)
b) ¿Cuál es la interpretación (pequeño/mediano/grande)?
c) ¿Es esto prácticamente importante en tu proyecto?

### Ejercicio 2: Intervalo de Confianza
Para datos arriba (Método B):
- Calcula error estándar (SE)
- Calcula IC 95%
- Reporta completo: "Media = ___ (IC 95%: [___, ___])"

### Ejercicio 3: Bootstrap
Para una variable en tu proyecto:
- Resamplea 1000 veces con reemplazo
- Calcula media para cada resample
- Grafica distribución bootstrap
- Reporta IC 95% bootstrap vs. paramétrico

### Ejercicio 4: Reportar Completo
Para comparación baseline vs. restricciones en tu proyecto, escribe párrafo que incluya:
- Medias (M) y desviaciones estándar (DE)
- Estadístico de prueba y p-valor
- Cohen's d o efecto tamaño apropiado
- IC 95% de diferencia
- Conclusión práctica

### Reflexión
1. **Muestras grandes vs. pequeñas**: ¿Por qué pequeño n pero gran efecto es mejor que gran n pero efecto tiny?
2. **Significancia vs. Importancia**: En tu proyecto, ¿qué efecto tamaño sería "prácticamente importante"?
3. **Reporte transparente**: ¿Por qué reportar efecto tamaño es mejor que "significancia" a secas?

---

**Próxima semana**: Aprenderemos qué pasa cuando comparas más de dos grupos (múltiples comparaciones y sus problemas).

## Errores Comunes

```{admonition} ⚠️ Trampas de tamaño del efecto
:class: warning

1. **Reportar solo p-valor**: Un resultado puede ser estadísticamente significativo pero prácticamente irrelevante.
2. **Ignorar intervalos de confianza**: El IC te dice qué tan precisa es tu estimación.
3. **Malinterpretar Cohen's d**: d=0.8 no significa "80% mejor", significa 0.8 desviaciones estándar de diferencia.
```

## Ejercicio Práctico: Decisión de Optimización

**Escenario**: Has desarrollado una optimización de kernel que reduce el tiempo de compilación. ¿Vale la pena implementarla en producción?

```{code-cell} ipython3
import numpy as np
from scipy import stats

# Datos: tiempo de compilación (segundos)
np.random.seed(42)
baseline = np.random.normal(5.2, 0.8, 30)  # n=30, media=5.2s, sd=0.8s
optimizado = np.random.normal(4.8, 0.9, 30)  # n=30, media=4.8s, sd=0.9s

print("=== Paso 1: Estadísticas Descriptivas ===")
print(f"Baseline:   M={np.mean(baseline):.3f}s, SD={np.std(baseline, ddof=1):.3f}s")
print(f"Optimizado: M={np.mean(optimizado):.3f}s, SD={np.std(optimizado, ddof=1):.3f}s")
print(f"Diferencia: {np.mean(baseline) - np.mean(optimizado):.3f}s ({(np.mean(baseline) - np.mean(optimizado)) / np.mean(baseline) * 100:.1f}%)")

# Paso 2: Prueba estadística
print("\n=== Paso 2: Significancia Estadística ===")
t_stat, p_value = stats.ttest_ind(baseline, optimizado)
print(f"t-test: t({len(baseline) + len(optimizado) - 2})={t_stat:.2f}, p={p_value:.4f}")

if p_value < 0.05:
    print("✓ Diferencia estadísticamente significativa (p<0.05)")
else:
    print("✗ No significativo (p≥0.05)")

# Paso 3: Tamaño del efecto (Cohen's d)
print("\n=== Paso 3: Tamaño del Efecto ===")

# Calcular pooled standard deviation
n1, n2 = len(baseline), len(optimizado)
s1, s2 = np.std(baseline, ddof=1), np.std(optimizado, ddof=1)
pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

# Cohen's d
cohens_d = (np.mean(baseline) - np.mean(optimizado)) / pooled_std

print(f"Pooled SD: {pooled_std:.3f}s")
print(f"Cohen's d: {cohens_d:.3f}")

# Interpretación
if abs(cohens_d) < 0.2:
    effect_interpretation = "Pequeño (d<0.2)"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "Pequeño-Mediano (0.2≤d<0.5)"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "Mediano (0.5≤d<0.8)"
else:
    effect_interpretation = "Grande (d≥0.8)"

print(f"Interpretación: {effect_interpretation}")
```

La siguiente fase consta de determinar nuestro grado de certeza al afirmar esto con un Intervalo de Confianza, y en consecuencia reflexionar respecto a la utilidad empírica de dicho rediseño.

```{code-cell} ipython3
# Paso 4: Intervalo de Confianza
print("\n=== Paso 4: Intervalo de Confianza (95%) ===")
se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
mean_diff = np.mean(baseline) - np.mean(optimizado)
ci_low = mean_diff - 1.96 * se_diff
ci_high = mean_diff + 1.96 * se_diff

print(f"Diferencia media: {mean_diff:.3f}s")
print(f"IC 95%: [{ci_low:.3f}s, {ci_high:.3f}s]")

if ci_low > 0:
    print("✓ IC no incluye 0 → mejora consistente")
else:
    print("⚠ IC incluye 0 → mejora no consistente")

# Paso 5: Decisión práctica
print("\n=== Paso 5: Decisión de Implementación ===")

# Criterios de decisión
tiempo_ahorrado = mean_diff
compilaciones_diarias = 1000  # Ejemplo: 1000 compilaciones/día
ahorro_diario = tiempo_ahorrado * compilaciones_diarias / 60  # minutos

print(f"Tiempo ahorrado por compilación: {tiempo_ahorrado:.3f}s")
print(f"Ahorro diario (1000 compilaciones): {ahorro_diario:.1f} minutos")

# Costo de implementación (ejemplo)
horas_implementacion = 8  # 1 día de trabajo
costo_hora = 50  # USD/hora
costo_total = horas_implementacion * costo_hora

print(f"\nCosto estimado de implementación: ${costo_total}")
print(f"Ahorro por día: {ahorro_diario:.1f} minutos de CPU")

# Decisión
print("\n--- DECISIÓN ---")
if cohens_d >= 0.5 and p_value < 0.05:
    print("✓ IMPLEMENTAR")
    print("  Razón: Efecto mediano-grande Y estadísticamente significativo")
    print(f"  Impacto: ~{ahorro_diario*60:.0f} segundos/día ahorrados")
elif cohens_d >= 0.3 and p_value < 0.05:
    print("? CONSIDERAR")
    print("  Razón: Efecto pequeño pero significativo")
    print(f"  Evaluar si {ahorro_diario:.1f} min/día justifica ${costo_total} de implementación")
else:
    print("✗ NO IMPLEMENTAR")
    print("  Razón: Efecto muy pequeño o no significativo")
    print("  Mejor invertir esfuerzo en otras optimizaciones")
```

De la mano, graficar nuestros resultados subraya dramáticamente lo concluyente - o no - de nuestros descubrimientos.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Paso 6: Visualización
print("\n=== Paso 6: Visualización ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Distribuciones
ax1.hist(baseline, bins=15, alpha=0.6, label='Baseline', color='steelblue')
ax1.hist(optimizado, bins=15, alpha=0.6, label='Optimizado', color='coral')
ax1.axvline(np.mean(baseline), color='steelblue', linestyle='--', linewidth=2)
ax1.axvline(np.mean(optimizado), color='coral', linestyle='--', linewidth=2)
ax1.set_xlabel('Tiempo de compilación (s)')
ax1.set_ylabel('Frecuencia')
ax1.set_title(f'Distribuciones (d={cohens_d:.2f})')
ax1.legend()

# Intervalos de confianza
methods = ['Baseline', 'Optimizado']
means = [np.mean(baseline), np.mean(optimizado)]
se_baseline = s1 / np.sqrt(n1)
se_optimizado = s2 / np.sqrt(n2)
ci_bars = [1.96 * se_baseline, 1.96 * se_optimizado]

ax2.bar(methods, means, yerr=ci_bars, capsize=10,
        color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Tiempo medio (s)')
ax2.set_title('Medias con IC 95%')
ax2.set_ylim([0, max(means) * 1.3])

# Añadir valores
for i, (m, ci) in enumerate(zip(means, ci_bars)):
    ax2.text(i, m + ci + 0.1, f'{m:.2f}s', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("Figura generada.")
```

---

## Referencias

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Routledge.
- Hedges, L. (1981). [Distribution Theory for Glass's Estimator of Effect Size](https://doi.org/10.3102/10769986006002107). Journal of Educational Statistics.
- Efron, B. & Tibshirani, R. (1993). [An Introduction to the Bootstrap](https://doi.org/10.1007/978-1-4899-4541-9). Chapman & Hall.
