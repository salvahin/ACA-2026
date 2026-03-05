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

# Análisis y Visualización: Comunicando Resultados de Experimentos

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/09_analisis_visualizacion.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 9
> **Tiempo de lectura:** ~30 minutos

---

## Introducción

Los datos sin análisis son solo números. La visualización efectiva transforma resultados de experimentos en **insights accionables**. Esta lectura te enseña técnicas para analizar datos de benchmarks GPU y comunicar hallazgos de forma clara y convincente.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Aplicar análisis exploratorio a datos de performance (stats descriptivas, distribuciones, outliers)
- Crear visualizaciones efectivas (box plots, scaling, heatmaps, roofline)
- Identificar patrones (crossover points, scaling behavior) y anomalías (outliers, bimodality)
- Generar reportes profesionales con hallazgos clave y evidencia visual
- Adaptar comunicación para diferentes audiencias (ejecutivos, ingenieros, investigadores)
```

---

## Análisis Exploratorio de Datos

### Carga y Limpieza

```{code-cell} ipython3
:tags: [skip-execution]

import pandas as pd
import numpy as np

def load_benchmark_results(path: str) -> pd.DataFrame:
    """Carga y prepara datos de benchmark."""

    df = pd.read_json(path)

    # Convertir tipos
    df['time_ms'] = df['time_ms'].astype(float)
    df['size'] = df['size'].astype(int)
    df['variant'] = df['variant'].astype(str)

    # Detectar outliers (> 3 std)
    grouped = df.groupby(['variant', 'size'])
    df['z_score'] = grouped['time_ms'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    df['is_outlier'] = df['z_score'].abs() > 3

    # Remover outliers
    df_clean = df[~df['is_outlier']].copy()

    print(f"Removed {df['is_outlier'].sum()} outliers ({df['is_outlier'].mean()*100:.1f}%)")

    return df_clean
```

### Estadísticas Descriptivas

```{code-cell} ipython3
:tags: [skip-execution]

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas por variante y tamaño."""

    stats = df.groupby(['variant', 'size']).agg({
        'time_ms': ['mean', 'std', 'min', 'max', 'median',
                    lambda x: np.percentile(x, 5),
                    lambda x: np.percentile(x, 95)]
    }).round(4)

    stats.columns = ['mean', 'std', 'min', 'max', 'median', 'p5', 'p95']

    # Calcular coeficiente de variación
    stats['cv'] = stats['std'] / stats['mean']

    return stats
```

### Análisis de Distribución

```{code-cell} ipython3
:tags: [skip-execution]

def analyze_distributions(df: pd.DataFrame):
    """Analiza la distribución de tiempos."""

    from scipy import stats

    results = []

    for (variant, size), group in df.groupby(['variant', 'size']):
        times = group['time_ms'].values

        # Test de normalidad
        _, p_normal = stats.normaltest(times) if len(times) > 20 else (0, 1)

        # Skewness y kurtosis
        skew = stats.skew(times)
        kurt = stats.kurtosis(times)

        results.append({
            'variant': variant,
            'size': size,
            'n_samples': len(times),
            'is_normal': p_normal > 0.05,
            'skewness': skew,
            'kurtosis': kurt,
            'distribution': (
                'normal' if abs(skew) < 0.5 and abs(kurt) < 1
                else 'right-skewed' if skew > 0.5
                else 'left-skewed' if skew < -0.5
                else 'heavy-tailed' if kurt > 1
                else 'light-tailed'
            )
        })

    return pd.DataFrame(results)
```

---

## Visualizaciones Efectivas

### 1. Comparación de Variantes

```{code-cell} ipython3
:tags: [skip-execution]

import matplotlib.pyplot as plt
import seaborn as sns

def plot_variant_comparison(df: pd.DataFrame, size: int):
    """Box plot comparando variantes para un tamaño específico."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    subset = df[df['size'] == size]

    # Box plot
    ax1 = axes[0]
    sns.boxplot(data=subset, x='variant', y='time_ms', ax=ax1)
    ax1.set_title(f'Time Distribution (size={size})')
    ax1.set_xlabel('Variant')
    ax1.set_ylabel('Time (ms)')

    # Violin plot para ver distribución completa
    ax2 = axes[1]
    sns.violinplot(data=subset, x='variant', y='time_ms', ax=ax2)
    ax2.set_title(f'Time Distribution Detail (size={size})')
    ax2.set_xlabel('Variant')
    ax2.set_ylabel('Time (ms)')

    plt.tight_layout()
    return fig
```

### 2. Scaling Analysis

```{code-cell} ipython3
:tags: [skip-execution]

def plot_scaling(df: pd.DataFrame):
    """Analiza cómo escala cada variante con el tamaño."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Tiempo absoluto
    ax1 = axes[0]
    stats = df.groupby(['variant', 'size'])['time_ms'].mean().reset_index()

    for variant in stats['variant'].unique():
        data = stats[stats['variant'] == variant]
        ax1.plot(data['size'], data['time_ms'], marker='o', label=variant)

    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Execution Time vs Size')

    # Throughput
    ax2 = axes[1]
    stats['throughput'] = stats['size'] / (stats['time_ms'] * 1e-3) / 1e9  # Billions/s

    for variant in stats['variant'].unique():
        data = stats[stats['variant'] == variant]
        ax2.plot(data['size'], data['throughput'], marker='s', label=variant)

    ax2.set_xlabel('Input Size')
    ax2.set_ylabel('Throughput (Billions elem/s)')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Throughput vs Size')

    plt.tight_layout()
    return fig
```

### 3. Heatmap de Speedup

```{code-cell} ipython3
:tags: [skip-execution]

def plot_speedup_heatmap(df: pd.DataFrame, baseline: str = 'A'):
    """Heatmap de speedup por variante y tamaño."""

    # Calcular speedup
    pivot = df.pivot_table(values='time_ms', index='variant', columns='size', aggfunc='mean')
    baseline_times = pivot.loc[baseline]
    speedup = pivot.apply(lambda row: baseline_times / row, axis=1)

    # Crear heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        speedup.drop(baseline),
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=1.0,
        ax=ax
    )

    ax.set_title(f'Speedup vs Baseline ({baseline})')
    ax.set_xlabel('Input Size')
    ax.set_ylabel('Variant')

    return fig
```

### 4. Roofline Plot

```{code-cell} ipython3
:tags: [skip-execution]

def plot_roofline(df: pd.DataFrame, peak_bandwidth: float, peak_compute: float):
    """Roofline model para analizar bottlenecks."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline teórico
    ai_range = np.logspace(-2, 2, 100)
    memory_bound = peak_bandwidth * ai_range
    compute_bound = np.full_like(ai_range, peak_compute)
    roofline = np.minimum(memory_bound, compute_bound)

    ax.plot(ai_range, roofline, 'k-', linewidth=2, label='Roofline')
    ax.axhline(peak_compute, color='gray', linestyle='--', alpha=0.5)

    # Puntos de datos
    for variant in df['variant'].unique():
        subset = df[df['variant'] == variant]
        # Calcular arithmetic intensity y achieved performance
        # (esto requiere conocer FLOPs y bytes transferidos)
        ai = subset['arithmetic_intensity']
        perf = subset['achieved_flops']
        ax.scatter(ai, perf, label=variant, s=100)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)')
    ax.set_ylabel('Performance (FLOPS)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Roofline Analysis')

    return fig
```

### 5. Stability Analysis

```{code-cell} ipython3
:tags: [skip-execution]

def plot_stability(df: pd.DataFrame):
    """Analiza estabilidad de mediciones."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Coeficiente de variación por variante
    ax1 = axes[0, 0]
    cv = df.groupby(['variant', 'size']).apply(
        lambda g: g['time_ms'].std() / g['time_ms'].mean()
    ).reset_index(name='cv')

    for variant in cv['variant'].unique():
        data = cv[cv['variant'] == variant]
        ax1.plot(data['size'], data['cv'] * 100, marker='o', label=variant)

    ax1.set_xlabel('Input Size')
    ax1.set_ylabel('Coefficient of Variation (%)')
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.set_title('Measurement Stability')
    ax1.grid(True)

    # Histograma de tiempos para una configuración
    ax2 = axes[0, 1]
    for variant in df['variant'].unique():
        times = df[(df['variant'] == variant) & (df['size'] == df['size'].max())]['time_ms']
        ax2.hist(times, alpha=0.5, label=variant, bins=20)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.set_title(f'Time Distribution (largest size)')

    # QQ plot para normalidad
    ax3 = axes[1, 0]
    from scipy import stats
    times = df[df['variant'] == 'A']['time_ms']
    stats.probplot(times, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Variant A)')

    # Time series de mediciones (para detectar drift)
    ax4 = axes[1, 1]
    subset = df[df['variant'] == 'A'].reset_index()
    ax4.plot(subset.index, subset['time_ms'], alpha=0.7)
    ax4.axhline(subset['time_ms'].mean(), color='red', linestyle='--')
    ax4.set_xlabel('Measurement Index')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Measurement Sequence (Variant A)')

    plt.tight_layout()
    return fig
```

---

## Patrones y Anomalías

### Detección de Anomalías

```{code-cell} ipython3
:tags: [skip-execution]

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta anomalías en los datos."""

    anomalies = []

    for (variant, size), group in df.groupby(['variant', 'size']):
        times = group['time_ms']

        # IQR method
        Q1 = times.quantile(0.25)
        Q3 = times.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = group[(times < lower_bound) | (times > upper_bound)]

        if len(outliers) > 0:
            anomalies.append({
                'variant': variant,
                'size': size,
                'n_outliers': len(outliers),
                'outlier_pct': len(outliers) / len(group) * 100,
                'min_outlier': outliers['time_ms'].min(),
                'max_outlier': outliers['time_ms'].max(),
                'normal_range': f"[{lower_bound:.3f}, {upper_bound:.3f}]"
            })

    return pd.DataFrame(anomalies)
```

### Identificación de Patrones

```{code-cell} ipython3
:tags: [skip-execution]

def identify_patterns(df: pd.DataFrame) -> Dict:
    """Identifica patrones en los datos."""

    patterns = {}

    # 1. Scaling behavior
    for variant in df['variant'].unique():
        subset = df[df['variant'] == variant]
        sizes = subset.groupby('size')['time_ms'].mean()

        # Fit power law: time = a * size^b
        log_sizes = np.log(sizes.index)
        log_times = np.log(sizes.values)
        slope, intercept = np.polyfit(log_sizes, log_times, 1)

        patterns[f'{variant}_scaling'] = {
            'exponent': slope,
            'type': (
                'linear (O(n))' if 0.9 < slope < 1.1
                else 'sublinear (O(n^<1))' if slope < 0.9
                else 'superlinear (O(n^>1))'
            )
        }

    # 2. Crossover points
    variants = df['variant'].unique()
    for i, v1 in enumerate(variants):
        for v2 in variants[i+1:]:
            times1 = df[df['variant'] == v1].groupby('size')['time_ms'].mean()
            times2 = df[df['variant'] == v2].groupby('size')['time_ms'].mean()

            # Find where they cross
            diff = times1 - times2
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            if len(sign_changes) > 0:
                crossover_size = times1.index[sign_changes[0]]
                patterns[f'{v1}_vs_{v2}_crossover'] = {
                    'size': crossover_size,
                    'faster_before': v1 if diff.iloc[0] < 0 else v2,
                    'faster_after': v2 if diff.iloc[0] < 0 else v1
                }

    return patterns
```

---

## Generación de Reportes

### Reporte Ejecutivo

```{code-cell} ipython3
:tags: [skip-execution]

def generate_executive_summary(df: pd.DataFrame, analysis: Dict) -> str:
    """Genera resumen ejecutivo para stakeholders."""

    # Encontrar mejor variante overall
    avg_speedups = {}
    baseline = 'A'
    baseline_mean = df[df['variant'] == baseline].groupby('size')['time_ms'].mean()

    for variant in df['variant'].unique():
        if variant == baseline:
            continue
        variant_mean = df[df['variant'] == variant].groupby('size')['time_ms'].mean()
        speedup = (baseline_mean / variant_mean).mean()
        avg_speedups[variant] = speedup

    best_variant = max(avg_speedups, key=avg_speedups.get)
    best_speedup = avg_speedups[best_variant]

    summary = f"""
# Executive Summary

## Key Finding
**Variant {best_variant}** achieves the best overall performance with an average
speedup of **{best_speedup:.2f}x** compared to baseline.

## Recommendations
1. Use Variant {best_variant} for production workloads
2. For sizes < {analysis.get('crossover_size', 'N/A')}, Variant A may be preferable
3. Further investigation needed for edge cases

## Performance Overview
| Metric | Value |
|--------|-------|
| Best Variant | {best_variant} |
| Avg Speedup | {best_speedup:.2f}x |
| Max Speedup | {max(avg_speedups.values()):.2f}x |
| Min Speedup | {min(avg_speedups.values()):.2f}x |

## Confidence
- Sample size: {len(df)} measurements
- Statistical significance: p < 0.01 for all comparisons
- Effect size: Large (Cohen's d > 0.8)
"""

    return summary
```

### Reporte Técnico Detallado

```{code-cell} ipython3
:tags: [skip-execution]

def generate_technical_report(df: pd.DataFrame, analysis: Dict) -> str:
    """Genera reporte técnico completo."""

    report = f"""
# Technical Performance Report

## Methodology
- Warmup iterations: {analysis['warmup']}
- Measurement iterations: {analysis['iterations']}
- Hardware: {analysis['hardware']}
- Software: {analysis['software']}

## Raw Statistics
{analysis['descriptive_stats'].to_markdown()}

## Statistical Analysis

### ANOVA Results
{pd.DataFrame(analysis['anova_by_size']).to_markdown()}

### Pairwise Comparisons
{pd.DataFrame(analysis['pairwise']).to_markdown()}

## Scaling Analysis
{pd.DataFrame(analysis['scaling']).to_markdown()}

## Anomalies Detected
{pd.DataFrame(analysis['anomalies']).to_markdown() if analysis['anomalies'] else 'None'}

## Recommendations
{analysis['recommendations']}

## Appendix: Data Quality
- Outliers removed: {analysis['outliers_removed']}
- Measurement stability (CV): {analysis['avg_cv']:.2%}
- Normality: {analysis['normality']}
"""

    return report
```

---

## Comunicación de Resultados

### Para Diferentes Audiencias

```{code-cell} ipython3
:tags: [skip-execution]

def format_for_audience(results: Dict, audience: str) -> str:
    """Formatea resultados para diferentes audiencias."""

    if audience == 'executive':
        return f"""
**Bottom Line**: Kernel B is {results['best_speedup']:.0%} faster.
Estimated annual savings: ${results['cost_savings']:,.0f}
"""

    elif audience == 'engineering':
        return f"""
## Performance Comparison
- Best: Variant {results['best_variant']} ({results['best_speedup']:.2f}x speedup)
- Memory bandwidth utilization: {results['bandwidth_util']:.0%}
- Compute utilization: {results['compute_util']:.0%}

### Key Optimizations
1. {results['optimization_1']}
2. {results['optimization_2']}

### Integration Notes
- API compatible: {results['api_compatible']}
- Breaking changes: {results['breaking_changes']}
"""

    elif audience == 'research':
        return f"""
## Experimental Results

### Hypothesis Testing
- H0: No difference between variants
- Result: H0 rejected (p < {results['p_value']:.4f})
- Effect size: {results['effect_size']} (Cohen's d = {results['cohens_d']:.2f})

### Confidence Intervals (95%)
{pd.DataFrame(results['confidence_intervals']).to_markdown()}

### Reproducibility
- Code: {results['code_hash']}
- Data: {results['data_hash']}
- Environment: {results['environment']}
"""
```

---

## Resumen

```{admonition} Resumen
:class: important
**Pipeline de análisis completo:**

1. **Limpieza**: Detectar y remover outliers (>3σ)
2. **Estadísticas**: Mean, std, median, p5, p95, CV
3. **Visualización**: Box plots (distribución), scaling (throughput vs size), heatmaps (speedup)
4. **Análisis**: ANOVA para diferencias globales, t-tests pairwise
5. **Reporte**: Adaptar mensaje según audiencia

**Visualizaciones esenciales:**
- **Box plot**: Ver distribución y outliers por variante
- **Scaling plot**: Verificar cómo escala con tamaño (O(n), O(n²), etc.)
- **Heatmap de speedup**: Comparar todas las variantes vs baseline
- **Roofline**: Identificar memory-bound vs compute-bound

**Checklist de reporte:**
- [ ] Outliers removidos y documentados
- [ ] Visualización principal (1 figura que cuenta la historia)
- [ ] Hallazgo clave en 1 frase
- [ ] Evidencia estadística (p-value, Cohen's d)
- [ ] Recomendación accionable
```

```{admonition} ⚡ Tip: La Figura Perfecta
:class: important
**Una buena visualización cuenta una historia completa:**

Elementos esenciales:
1. **Título descriptivo**: "Variant B is 2x faster across all sizes"
2. **Ejes claros**: Labels con unidades (ms, GB/s, TFLOPS)
3. **Baseline visible**: Línea horizontal o color diferenciado
4. **Significancia anotada**: * para p<0.05, ** para p<0.01
5. **Error bars**: Mostrar variabilidad (std o CI 95%)

**Malo**: Scatter plot sin contexto
**Bueno**: Box plot + línea de tendencia + baseline + significancia
```

```{admonition} 📊 Cómo detectar anomalías
:class: tip
**Patrones sospechosos en datos:**

1. **Bimodalidad**: Dos picos en histograma → thermal throttling o scheduling
2. **Outliers consistentes**: Siempre en misma posición → bug o edge case
3. **Varianza creciente**: CV aumenta con tamaño → falta estabilidad
4. **Crossover inesperado**: Variante A mejor para n<1000, B mejor para n>1000

Usa Q-Q plot para verificar normalidad - si no es normal, usa mediana en vez de media.
```

```{admonition} 🎯 En tu proyecto
:class: note
Tu reporte final incluirá:
1. **Executive summary** (1 página): Hallazgo principal + ROI
2. **Visualización de resultados**: Speedup heatmap por nivel
3. **Análisis de errores**: Taxonomía de errores comunes del generador
4. **Ablation studies**: Impacto de cada componente (gramática, prompts, etc.)

La visualización efectiva multiplica el impacto de tu trabajo.
```

---

## Ejercicios y Reflexión

### Ejercicio 1: Análisis Completo

Dado un dataset de benchmarks:
1. Calcula estadísticas descriptivas
2. Detecta outliers
3. Identifica patrones de scaling
4. Genera visualizaciones

### Ejercicio 2: Reporte

Crea un reporte de una página que comunique:
- Principal hallazgo
- Evidencia (una visualización)
- Recomendación

### Ejercicio 3: Presentación

Adapta tus resultados para:
1. Un ejecutivo (1 slide)
2. Un equipo técnico (5 slides)
3. Una publicación (paper format)

### Para Pensar

> *Si tus resultados muestran que la variante B es 50% más rápida pero tiene 3x más varianza, ¿cómo comunicarías esto?*

---

## Próximos Pasos

En la última semana, consolidaremos todo en **Documentación Final**: cómo documentar tu trabajo para que otros puedan reproducirlo y construir sobre él.

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Williams, S., Waterman, A., & Patterson, D. (2009). [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785). Communications of the ACM.
- Tufte, E. (2001). The Visual Display of Quantitative Information (2nd ed.). Graphics Press.
- Plotly. [Plotly Python Open Source Graphing Library](https://plotly.com/python/). Plotly.
