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

# Visualización de Datos para Experimentos de Performance

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/09-mlops-visualizacion.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Elegir el tipo de gráfico correcto para cada pregunta de investigación
- Crear visualizaciones efectivas con matplotlib y seaborn
- Diseñar dashboards informativos que comuniquen resultados claramente
- Aplicar principios de accesibilidad (colores, fuentes, contraste)
- Exportar figuras en formatos apropiados para publicación
```

## Semana 9 - Estadística para Generación de Kernels GPU

Los datos de performance son complejos: múltiples variables, distribuciones no normales, outliers, y relaciones no lineales. Una buena visualización transforma estos datos en **insights accionables**. Esta lectura te enseña a elegir y crear visualizaciones efectivas para experimentos de kernels GPU.

---

## Principios de Visualización Efectiva

```{admonition} 💡 Intuición
:class: hint
Antes de hacer cualquier gráfico, pregúntate: "¿Qué pregunta exacta quiero responder?" Un gráfico mal elegido puede oscurecer patrones importantes o exagerar diferencias triviales. El tipo de gráfico debe seguir naturalmente de tu pregunta de investigación.
```

### El Propósito Guía la Forma

```
Pregunta de investigación → Tipo de gráfico

"¿Cómo se distribuyen los tiempos?" → Histograma, density plot
"¿Difieren los métodos A y B?" → Box plot, violin plot
"¿Hay correlación entre X e Y?" → Scatter plot
"¿Cómo evoluciona la métrica?" → Line plot
"¿Cuál es el mejor en cada condición?" → Heatmap
```

```{admonition} 🎯 Aplicación en ML
:class: important
En tu proyecto de evaluación de kernels GPU:
- **Distribuciones**: Histograma de iteraciones para detectar outliers
- **Comparaciones**: Box plot de baseline vs restricciones para ver diferencias
- **Relaciones**: Scatter de tiempo vs throughput para entender trade-offs
- **Evolución**: Line plot de validez a través de experimentos para ver tendencias
- **Configuraciones**: Heatmap de parámetros para identificar configuración óptima
```

### Jerarquía Visual

```
1. Posición (más preciso)
   └─ Scatter plots, bar charts

2. Longitud
   └─ Bar charts, histogramas

3. Ángulo
   └─ Pie charts (evitar si posible)

4. Área
   └─ Bubble charts

5. Color/intensidad (menos preciso)
   └─ Heatmaps

Regla: Usa posición para la comparación más importante
```

### Data-Ink Ratio

Edward Tufte propuso maximizar el **data-ink ratio**:

```
Data-ink ratio = Tinta que representa datos / Tinta total

MALO (bajo ratio):          BUENO (alto ratio):
┌──────────────────┐
│  ████ Baseline   │        Baseline  ████████████ 75%
│  ████████████    │        Grammar   █████████████████ 92%
│                  │
│  ████ Grammar    │
│  ███████████████ │
│                  │
└──────────────────┘
```

Elimina: gridlines innecesarios, fondos, bordes 3D, efectos decorativos.

---

## Visualizaciones para Distribuciones

### Histograma

Para ver la **forma** de una distribución:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Datos de ejemplo: tiempos de ejecución (ms)
times_baseline = np.random.exponential(scale=2.0, size=100)
times_optimized = np.random.exponential(scale=1.5, size=100)

fig, ax = plt.subplots(figsize=(10, 6))

# Histogramas superpuestos
ax.hist(times_baseline, bins=20, alpha=0.5, label='Baseline', color='steelblue')
ax.hist(times_optimized, bins=20, alpha=0.5, label='Optimizado', color='coral')

ax.set_xlabel('Tiempo de ejecución (ms)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de Tiempos: Baseline vs Optimizado')
ax.legend()

plt.tight_layout()
plt.savefig('histogram_comparison.png', dpi=300)
```

**Cuándo usar**: Una variable continua, quieres ver forma (normal, sesgada, bimodal).

### Density Plot (KDE)

Versión suavizada del histograma:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))

sns.kdeplot(times_baseline, label='Baseline', color='steelblue', ax=ax)
sns.kdeplot(times_optimized, label='Optimizado', color='coral', ax=ax)

ax.set_xlabel('Tiempo de ejecución (ms)')
ax.set_ylabel('Densidad')
ax.set_title('Densidad de Tiempos de Ejecución')
ax.legend()

plt.tight_layout()
```

**Cuándo usar**: Comparar formas de distribuciones, datos continuos abundantes.

### Box Plot

Resume: mediana, cuartiles, outliers:

```{code-cell} ipython3
import pandas as pd

# Crear DataFrame
data = pd.DataFrame({
    'Tiempo': np.concatenate([times_baseline, times_optimized]),
    'Método': ['Baseline'] * 100 + ['Optimizado'] * 100
})

fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(data=data, x='Método', y='Tiempo', ax=ax)

ax.set_ylabel('Tiempo de ejecución (ms)')
ax.set_title('Comparación de Tiempos por Método')

plt.tight_layout()
```

**Cuándo usar**: Comparar grupos, resumen rápido, detectar outliers.

### Violin Plot

Box plot + distribución completa:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(data=data, x='Método', y='Tiempo', ax=ax, inner='box')

ax.set_ylabel('Tiempo de ejecución (ms)')
ax.set_title('Distribución Completa por Método')

plt.tight_layout()
```

**Cuándo usar**: Cuando la forma de la distribución importa (bimodal, sesgada).

---

## Visualizaciones para Comparaciones

### Bar Plot con Error Bars

Para comparar medias con incertidumbre:

```{code-cell} ipython3
# Calcular estadísticas
methods = ['Baseline', 'Optimizado', 'Autotuned']
means = [2.5, 1.8, 1.2]
stds = [0.5, 0.3, 0.2]

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(methods, means, yerr=stds, capsize=5,
              color=['steelblue', 'coral', 'forestgreen'],
              alpha=0.8)

ax.set_ylabel('Tiempo medio (ms)')
ax.set_title('Comparación de Métodos (mean ± std)')

# Añadir valores encima de barras
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
            f'{mean:.2f}', ha='center', va='bottom')

plt.tight_layout()
```

### Grouped Bar Plot

Comparar múltiples métodos en múltiples condiciones:

```{code-cell} ipython3
# Datos: tiempo por método y tamaño de input
sizes = ['256', '512', '1024', '2048']
baseline = [1.2, 2.4, 5.1, 11.2]
optimized = [1.0, 1.8, 3.2, 6.1]
autotuned = [0.9, 1.5, 2.5, 4.8]

x = np.arange(len(sizes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width, baseline, width, label='Baseline', color='steelblue')
ax.bar(x, optimized, width, label='Optimizado', color='coral')
ax.bar(x + width, autotuned, width, label='Autotuned', color='forestgreen')

ax.set_xlabel('Tamaño de input')
ax.set_ylabel('Tiempo (ms)')
ax.set_title('Tiempo de Ejecución por Método y Tamaño')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()

plt.tight_layout()
```

### Speedup Plot

Visualizar mejora relativa:

```{code-cell} ipython3
sizes = [256, 512, 1024, 2048, 4096]
speedup_opt = [1.2, 1.33, 1.59, 1.84, 2.1]
speedup_auto = [1.33, 1.6, 2.04, 2.33, 2.8]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sizes, speedup_opt, 'o-', label='Optimizado', color='coral', linewidth=2)
ax.plot(sizes, speedup_auto, 's-', label='Autotuned', color='forestgreen', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', label='Baseline')

ax.set_xlabel('Tamaño de input')
ax.set_ylabel('Speedup vs Baseline')
ax.set_xscale('log', base=2)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Speedup por Método')

plt.tight_layout()
```

---

## Visualizaciones para Relaciones

### Scatter Plot

Relación entre dos variables:

```{code-cell} ipython3
# Datos: tiempo vs throughput
times = np.random.exponential(scale=2.0, size=50)
throughput = 1000 / times + np.random.normal(0, 50, size=50)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(times, throughput, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Tiempo (ms)')
ax.set_ylabel('Throughput (GB/s)')
ax.set_title('Relación Tiempo-Throughput')

# Añadir línea de tendencia
z = np.polyfit(times, throughput, 1)
p = np.poly1d(z)
ax.plot(sorted(times), p(sorted(times)), "r--", alpha=0.8, label=f'Tendencia')
ax.legend()

plt.tight_layout()
```

### Scatter con Color por Categoría

```{code-cell} ipython3
# Añadir categoría
methods = np.random.choice(['A', 'B', 'C'], size=50)

fig, ax = plt.subplots(figsize=(10, 6))

for method, color in zip(['A', 'B', 'C'], ['steelblue', 'coral', 'forestgreen']):
    mask = methods == method
    ax.scatter(times[mask], throughput[mask], alpha=0.6,
               label=f'Método {method}', color=color, edgecolors='black')

ax.set_xlabel('Tiempo (ms)')
ax.set_ylabel('Throughput (GB/s)')
ax.set_title('Tiempo-Throughput por Método')
ax.legend()

plt.tight_layout()
```

### Heatmap de Correlación

Múltiples variables a la vez:

```{code-cell} ipython3
# Crear DataFrame con múltiples métricas
metrics = pd.DataFrame({
    'Tiempo': times,
    'Throughput': throughput,
    'Memoria': np.random.uniform(100, 500, 50),
    'Ocupación': np.random.uniform(0.3, 0.9, 50)
})

fig, ax = plt.subplots(figsize=(8, 6))

correlation = metrics.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax,
            vmin=-1, vmax=1, fmt='.2f')

ax.set_title('Matriz de Correlación')

plt.tight_layout()
```

---

## Visualizaciones para Series Temporales

### Line Plot con Banda de Confianza

```{code-cell} ipython3
# Simular múltiples runs
n_runs = 10
n_points = 50
runs = np.array([np.cumsum(np.random.normal(0.5, 0.1, n_points)) for _ in range(n_runs)])

mean = runs.mean(axis=0)
std = runs.std(axis=0)

fig, ax = plt.subplots(figsize=(10, 6))

x = range(n_points)
ax.plot(x, mean, color='steelblue', linewidth=2, label='Media')
ax.fill_between(x, mean - std, mean + std, alpha=0.3, color='steelblue', label='±1 std')
ax.fill_between(x, mean - 2*std, mean + 2*std, alpha=0.15, color='steelblue')

ax.set_xlabel('Iteración')
ax.set_ylabel('Métrica acumulada')
ax.set_title('Convergencia con Intervalo de Confianza')
ax.legend()

plt.tight_layout()
```

### Múltiples Series

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 6))

# Datos de múltiples experimentos
for i, (label, color) in enumerate([('Config A', 'steelblue'),
                                      ('Config B', 'coral'),
                                      ('Config C', 'forestgreen')]):
    y = np.cumsum(np.random.normal(0.4 + i*0.1, 0.1, n_points))
    ax.plot(range(n_points), y, label=label, color=color, linewidth=2)

ax.set_xlabel('Iteración')
ax.set_ylabel('Performance')
ax.set_title('Evolución por Configuración')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
```

---

## Heatmaps para Resultados Multidimensionales

### Heatmap de Performance

```{code-cell} ipython3
# Grid de resultados: BLOCK_M x BLOCK_N
block_m = [32, 64, 128, 256]
block_n = [32, 64, 128, 256]
performance = np.array([
    [1.2, 1.5, 1.8, 1.4],
    [1.6, 2.1, 2.5, 2.0],
    [1.9, 2.4, 2.8, 2.3],
    [1.5, 1.9, 2.2, 1.8]
])

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(performance, cmap='YlGn', aspect='auto')

# Etiquetas
ax.set_xticks(range(len(block_n)))
ax.set_yticks(range(len(block_m)))
ax.set_xticklabels(block_n)
ax.set_yticklabels(block_m)
ax.set_xlabel('BLOCK_N')
ax.set_ylabel('BLOCK_M')
ax.set_title('Speedup por Configuración de Bloque')

# Añadir valores en celdas
for i in range(len(block_m)):
    for j in range(len(block_n)):
        ax.text(j, i, f'{performance[i, j]:.1f}x',
                ha='center', va='center', color='black')

plt.colorbar(im, label='Speedup')
plt.tight_layout()
```

### Heatmap de Significancia

```{code-cell} ipython3
# p-values de comparaciones
methods = ['A', 'B', 'C', 'D']
p_values = np.array([
    [1.0, 0.02, 0.001, 0.15],
    [0.02, 1.0, 0.08, 0.03],
    [0.001, 0.08, 1.0, 0.01],
    [0.15, 0.03, 0.01, 1.0]
])

fig, ax = plt.subplots(figsize=(8, 6))

# Colores: verde = significativo, rojo = no significativo
im = ax.imshow(p_values, cmap='RdYlGn_r', vmin=0, vmax=0.1)

ax.set_xticks(range(len(methods)))
ax.set_yticks(range(len(methods)))
ax.set_xticklabels(methods)
ax.set_yticklabels(methods)
ax.set_title('p-values de Comparaciones Pairwise')

for i in range(len(methods)):
    for j in range(len(methods)):
        color = 'white' if p_values[i, j] < 0.05 else 'black'
        ax.text(j, i, f'{p_values[i, j]:.3f}',
                ha='center', va='center', color=color, fontsize=10)

plt.colorbar(im, label='p-value')
plt.tight_layout()
```

---

## Dashboard de Resultados

### Diseño de Layout

```{code-cell} ipython3
fig = plt.figure(figsize=(16, 10))

# Layout: 2x3 grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel 1: Distribución de tiempos
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(data=data, x='Método', y='Tiempo', ax=ax1)
ax1.set_title('Distribución de Tiempos')

# Panel 2: Speedup por tamaño
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(sizes, speedup_opt, 'o-', label='Opt')
ax2.plot(sizes, speedup_auto, 's-', label='Auto')
ax2.axhline(1.0, color='gray', linestyle='--')
ax2.set_xscale('log', base=2)
ax2.legend()
ax2.set_title('Speedup vs Tamaño')

# Panel 3: Heatmap de configuración
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(performance, annot=True, fmt='.1f', ax=ax3)
ax3.set_title('Config Óptima')

# Panel 4: Scatter tiempo-throughput
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(times, throughput, alpha=0.6)
ax4.set_xlabel('Tiempo')
ax4.set_ylabel('Throughput')
ax4.set_title('Tiempo vs Throughput')

# Panel 5: Convergencia
ax5 = fig.add_subplot(gs[1, 1:])
ax5.plot(range(n_points), mean, label='Media')
ax5.fill_between(range(n_points), mean-std, mean+std, alpha=0.3)
ax5.set_xlabel('Iteración')
ax5.set_title('Convergencia del Experimento')
ax5.legend()

plt.suptitle('Dashboard de Resultados de Performance', fontsize=14, fontweight='bold')
plt.savefig('dashboard.png', dpi=300, bbox_inches='tight')
```

---

## Accesibilidad en Visualizaciones

### Paletas de Colores

```python
# EVITAR: Rojo-Verde (daltonismo)
bad_colors = ['red', 'green']

# PREFERIR: Paletas colorblind-friendly
good_colors = sns.color_palette('colorblind')

# O usar paletas específicas
from matplotlib import cm
palette_viridis = cm.viridis  # Secuencial
palette_coolwarm = cm.coolwarm  # Divergente
```

### Checklist de Accesibilidad

```
☐ Colores: No dependo solo de rojo/verde
☐ Contraste: Suficiente entre fondo y elementos
☐ Tamaño de fuente: Mínimo 12pt para texto, 10pt para etiquetas
☐ Patrones: Usar patrones además de colores para distinguir
☐ Alt text: Descripción textual para reportes digitales
☐ Leyenda: Clara, sin ambigüedad
☐ Ejes: Etiquetados con unidades
```

### Añadir Patrones

```{code-cell} ipython3
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 6))

# Barras con patrones
patterns = ['/', '\\', 'x', 'o']
colors = ['steelblue', 'coral', 'forestgreen', 'purple']

for i, (method, val, pattern, color) in enumerate(
    zip(methods, means, patterns, colors)):
    bar = ax.bar(i, val, color=color, hatch=pattern, edgecolor='black')

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods)
ax.set_ylabel('Valor')
ax.set_title('Comparación con Patrones')

plt.tight_layout()
```

---

## Exportación para Publicación

### Resolución y Formato

```python
# Para paper/tesis
fig.savefig('figure.pdf', dpi=300, bbox_inches='tight')  # Vectorial
fig.savefig('figure.png', dpi=300, bbox_inches='tight')  # Raster alta res

# Para presentaciones
fig.savefig('figure_slides.png', dpi=150, bbox_inches='tight')

# Para web
fig.savefig('figure_web.png', dpi=72, bbox_inches='tight')
```

### Estilo Consistente

```{code-cell} ipython3
# Definir estilo al inicio del notebook/script
plt.style.use('seaborn-v0_8-whitegrid')

# O configuración manual
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 100,
    'savefig.dpi': 300,
})
```

---

## Integración con Weights & Biases (W&B)

W&B es una plataforma para trackear experimentos ML. Complementa matplotlib con dashboards interactivos y comparación automática de runs.

### Setup Básico

```{code-cell} ipython3
import wandb

# Inicializar proyecto
wandb.init(
    project="gpu-kernel-optimization",
    name="baseline-run-1",
    config={
        "method": "baseline",
        "temperature": 0.0,
        "seed": 42
    }
)

# Durante experimento
for i in range(100):
    accuracy = run_iteration(i)
    wandb.log({"iteration": i, "accuracy": accuracy})

wandb.finish()
```

### Comparar Métodos

```{code-cell} ipython3
# Run 1: Baseline
wandb.init(project="gpu-kernels", name="baseline")
for i in range(100):
    wandb.log({"accuracy": baseline_acc[i]})
wandb.finish()

# Run 2: Con gramática
wandb.init(project="gpu-kernels", name="grammar")
for i in range(100):
    wandb.log({"accuracy": grammar_acc[i]})
wandb.finish()

# Dashboard automático en: https://wandb.ai/<user>/gpu-kernels
```

**Ventajas:** Comparación automática de runs, historial de experimentos, colaboración en equipo, exportación de gráficos.

---

## Ejercicios y Reflexión

```{admonition} ✅ Verifica tu comprensión
:class: note
1. **Tipos de gráfico**: ¿Cuándo usarías un box plot vs un violin plot?
2. **Accesibilidad**: ¿Por qué la paleta rojo-verde es problemática?
3. **Data-ink ratio**: ¿Qué elementos eliminarías de un gráfico sobrecargado?
4. **Exportación**: ¿Qué diferencia hay entre PDF y PNG para publicación?
```

### Ejercicio 1: Elegir Visualización

Para cada pregunta, indica el tipo de gráfico más apropiado:

1. "¿Cómo se distribuyen los tiempos de ejecución del kernel A?"
2. "¿Es el kernel B significativamente más rápido que A?"
3. "¿Hay correlación entre tamaño de bloque y throughput?"
4. "¿Cuál combinación de parámetros es óptima?"
5. "¿Cómo evoluciona la pérdida durante el entrenamiento?"

### Ejercicio 2: Crear Dashboard

Diseña un dashboard de 4 paneles que responda:
- ¿Cuál método es mejor overall?
- ¿Cómo varía con el tamaño de input?
- ¿Son las diferencias significativas?
- ¿Hay outliers preocupantes?

### Ejercicio 3: Accesibilidad

Toma una de tus visualizaciones actuales y:
1. Cambia la paleta de colores a una colorblind-friendly
2. Añade patrones además de colores
3. Asegura fuentes legibles
4. Añade descripción textual

### Reflexión

1. ¿Qué visualización usarías para un resumen ejecutivo de 1 minuto?
2. ¿Cómo cambiaría tu visualización si la audiencia fuera técnica vs no-técnica?
3. ¿Qué información puede perderse al simplificar una visualización?

---

**Próxima semana (final)**: Aprenderemos cómo reportar resultados estadísticos en formato académico profesional.

## Errores Comunes

```{admonition} ⚠️ Trampas de visualización
:class: warning

1. **Ejes truncados**: Comenzar el eje Y en valor distinto de 0 exagera diferencias visualmente.
2. **Gráficos 3D innecesarios**: Añaden complejidad sin información adicional, dificultan lectura.
3. **Paletas no accesibles**: Rojo-verde excluye ~8% de la población (daltonismo).
4. **Barras apiladas para comparaciones**: Dificulta comparar categorías no alineadas con el eje.
```

## Ejercicio Práctico: Visualización Correcta vs Incorrecta

**Escenario**: Comparar performance de tres métodos de optimización de kernels.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Datos de ejemplo
np.random.seed(42)
methods = ['Baseline', 'Optimizado', 'Autotuned']
times = {
    'Baseline': np.random.normal(5.0, 0.3, 50),
    'Optimizado': np.random.normal(4.5, 0.4, 50),
    'Autotuned': np.random.normal(4.2, 0.5, 50)
}

# --- VERSIÓN INCORRECTA ---
print("=== Versión INCORRECTA (múltiples problemas) ===\n")

fig_bad, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Error 1: Eje Y truncado (exagera diferencias)
means = [np.mean(times[m]) for m in methods]
ax1.bar(methods, means, color=['red', 'green', 'blue'])  # Error 2: rojo-verde
ax1.set_ylim([4.0, 5.2])  # Eje truncado
ax1.set_ylabel('Tiempo')
ax1.set_title('Comparación de Métodos')
ax1.text(0.5, 0.95, '⚠ PROBLEMAS:\n1. Eje Y truncado\n2. Rojo-verde\n3. Sin error bars\n4. Sin unidades',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Error 3: Pie chart para comparaciones (difícil de leer)
ax2.pie(means, labels=methods, autopct='%1.1f%%',
        colors=['red', 'green', 'blue'])
ax2.set_title('Distribución de Tiempos (?)')
ax2.text(0.5, -1.3, '⚠ PROBLEMA:\nPie chart inadecuado\npara comparar tiempos',
         transform=ax2.transAxes, fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
print("Ejemplo de figura INCORRECTA mostrandos múltiples anti-patrones.")
```

Para sanear todos estos problemas a nivel de interpretación visual y analítica, debemos aplicar las mejores prácticas que garanticen precisión y accesibilidad en nuestras comunicaciones:

```{code-cell} ipython3
# --- VERSIÓN CORRECTA ---
print("\n=== Versión CORRECTA (mejores prácticas) ===\n")

fig_good, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bar plot CORRECTO: eje desde 0, error bars, colores accesibles
means = [np.mean(times[m]) for m in methods]
stds = [np.std(times[m], ddof=1) for m in methods]
colors_accessible = ['#0173B2', '#DE8F05', '#029E73']  # Colorblind-friendly

bars = ax1.bar(methods, means, yerr=stds, capsize=8,
               color=colors_accessible, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Tiempo medio (ms)', fontsize=12)
ax1.set_ylim([0, max(means) * 1.3])  # Desde 0
ax1.set_title('Comparación de Métodos (mean ± std)', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Añadir valores
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
             f'{mean:.2f}ms', ha='center', fontweight='bold')

ax1.text(0.5, 0.95, '✓ BUENO:\n• Eje desde 0\n• Error bars\n• Colores accesibles\n• Unidades claras',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 2. Box plot: muestra distribuciones completas
data_list = [times[m] for m in methods]
bp = ax2.boxplot(data_list, labels=methods, patch_artist=True)

for patch, color in zip(bp['boxes'], colors_accessible):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_ylabel('Tiempo (ms)', fontsize=12)
ax2.set_title('Distribuciones Completas', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.text(0.5, 0.95, '✓ BUENO:\n• Muestra outliers\n• Mediana vs media\n• Variabilidad visible',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 3. Violin plot: distribución + estadísticas
import pandas as pd
data_df = pd.DataFrame({
    'Tiempo': np.concatenate([times[m] for m in methods]),
    'Método': np.concatenate([[m]*len(times[m]) for m in methods])
})

parts = ax3.violinplot([times[m] for m in methods], positions=range(len(methods)),
                        showmeans=True, showmedians=True)

for pc, color in zip(parts['bodies'], colors_accessible):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)

ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods)
ax3.set_ylabel('Tiempo (ms)', fontsize=12)
ax3.set_title('Distribuciones con Densidad', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.text(0.5, 0.95, '✓ BUENO:\n• Forma de distribución\n• Media y mediana\n• Detecta bimodalidad',
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 4. Intervalos de confianza
from scipy import stats as sp_stats

ci_95 = []
for m in methods:
    se = stds[methods.index(m)] / np.sqrt(len(times[m]))
    ci = 1.96 * se
    ci_95.append(ci)

ax4.errorbar(range(len(methods)), means, yerr=ci_95,
             fmt='o', markersize=10, capsize=10, capthick=2,
             color='black', ecolor=colors_accessible, elinewidth=3)

for i, (mean, ci, color) in enumerate(zip(means, ci_95, colors_accessible)):
    ax4.fill_between([i-0.2, i+0.2], mean-ci, mean+ci,
                     alpha=0.3, color=color)

ax4.set_xticks(range(len(methods)))
ax4.set_xticklabels(methods)
ax4.set_ylabel('Tiempo medio (ms)', fontsize=12)
ax4.set_ylim([0, max(means) * 1.3])
ax4.set_title('Intervalos de Confianza 95%', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.text(0.5, 0.95, '✓ BUENO:\n• Incertidumbre visible\n• Solapamiento claro\n• Significancia visual',
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.show()
print("Opciones visuales más correctas prestando atención al detalle.")
```

Se vuelve imperativo contar con un listado estructurado mentalmente para auditar nuestras métricas previas de cara a presentarlos apropiadamente, y una buena lista base se condensa aquí:

```{code-cell} ipython3
# --- CHECKLIST DE VERIFICACIÓN ---
print("\n=== Checklist de Verificación ===\n")

checklist = {
    "Eje Y comienza en 0 (o justificado)": True,
    "Error bars o IC presentes": True,
    "Colores colorblind-friendly": True,
    "Unidades especificadas en ejes": True,
    "Título descriptivo": True,
    "Leyenda clara (si aplica)": True,
    "Fuente legible (≥10pt)": True,
    "Formato exportable (PNG/PDF)": True,
    "Data-ink ratio alto": True,
    "Sin efectos 3D innecesarios": True
}

for item, status in checklist.items():
    symbol = "✓" if status else "✗"
    print(f"{symbol} {item}")

print("\n--- LECCIONES CLAVE ---")
print("1. El eje Y truncado puede hacer que diferencias pequeñas parezcan enormes")
print("2. Los colores rojo-verde excluyen a personas con daltonismo (~8% población)")
print("3. Siempre incluye medidas de incertidumbre (error bars, IC)")
print("4. Pie charts son malos para comparar valores - usa barras")
print("5. Box plots > bar plots cuando la distribución importa")
```

---

## Referencias

- Tufte, E. (2001). The Visual Display of Quantitative Information (2nd ed.). Graphics Press.
- Weights & Biases. [Documentation](https://docs.wandb.ai/). W&B.
- MLflow. [Documentation](https://mlflow.org/docs/latest/). MLflow.
- Wickham, H. (2010). [A Layered Grammar of Graphics](https://doi.org/10.1198/jcgs.2009.07098). Journal of Computational and Graphical Statistics.
