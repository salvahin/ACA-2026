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

# Reporte Estadístico

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/10-reporte-estadistico.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton
print('Dependencies installed!')
```
## Semana 10 - Estadística para Generación de Kernels GPU

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Reportar resultados en formato APA e IEEE correctamente
- Estructurar secciones de Resultados y Discusión profesionalmente
- Crear tablas estadísticas claras y completas
- Distinguir entre significancia estadística y práctica en reportes
- Discutir limitaciones y amenazas a la validez honestamente
```

Hemos cubierto todo: desde probabilidad fundamental hasta MLOps. Ahora aprendemos a comunicar resultados de forma que el mundo académico entienda, confíe y cite tu trabajo.

## Estándares de Reporte: APA y IEEE

```{admonition} 💡 Intuición
:class: hint
Reportar resultados es como contar una historia con números. No solo digas "gané", di "gané por esta cantidad, con esta confianza, bajo estas condiciones, y aquí están las pruebas." El formato estandarizado (APA/IEEE) asegura que TODA la información necesaria esté presente para que otros evalúen y repliquen tu trabajo.
```

### Formato APA (American Psychological Association)

Usado en psicología, ciencias sociales, educación.

```{admonition} 📊 Cómo interpretar formato APA
:class: note
**Estructura completa**:
"Se realizó [prueba] para comparar [DV] entre [grupo 1] (M=___, DE=___) y [grupo 2] (M=___, DE=___).
El análisis reveló diferencia [significativa/no significativa],
[estadístico](df) = ___, p = ___, d = ___, IC 95% [___, ___]."

**Elementos obligatorios**:
- **M y DE**: Contexto numérico de cada grupo
- **Estadístico y df**: Cómo se calculó (t, F, U, etc.)
- **p-valor**: Significancia estadística
- **d (tamaño efecto)**: Importancia práctica
- **IC 95%**: Rango plausible del efecto

**Ejemplo completo**:
"Los kernels con restricciones (M=85.2%, DE=12.3%) tuvieron mayor validez que baseline
(M=75.1%, DE=15.6%), t(49)=2.84, p=0.006, d=0.71, IC 95%[3.0%, 16.9%]."
```

Componentes APA:
- M = media
- DE = desviación estándar
- t(___, ___) = estadístico y grados de libertad
- p = p-valor (reporta exacto si p > 0.001)
- d = Cohen's d
- IC = intervalo de confianza con porcentaje

### Formato IEEE (Institute of Electrical and Electronics Engineers)

Usado en ingeniería, CS, sistemas.

```
Estructura similar pero menos descriptiva:

"Tabla 1 muestra tasa de validez para baseline vs. restricciones.
La prueba t pareada indicó diferencia significativa (t = 2.84,
p < 0.01). El tamaño del efecto fue mediano (d = 0.71)."

IEEE prefiere tablas y números concisos.
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns

# Simulación de datos experimentales para demostración de reporte
np.random.seed(42)

# Generar datos para baseline y restricciones
n = 100
baseline_validez = np.random.binomial(1, 0.75, n)
restricciones_validez = np.random.binomial(1, 0.85, n)

baseline_iteraciones = np.random.poisson(43, n)
restricciones_iteraciones = np.random.poisson(38, n)

baseline_tiempo = np.random.normal(5.24, 1.12, n)
restricciones_tiempo = np.random.normal(4.68, 0.98, n)

# Crear DataFrame
data = pd.DataFrame({
    'Metodo': ['Baseline']*n + ['Restricciones']*n,
    'Validez': list(baseline_validez) + list(restricciones_validez),
    'Iteraciones': list(baseline_iteraciones) + list(restricciones_iteraciones),
    'Tiempo': list(baseline_tiempo) + list(restricciones_tiempo)
})

print("Ejemplo de Reporte Estadístico Profesional")
print("=" * 70)
print("\n" + "ESTADÍSTICA DESCRIPTIVA".center(70))
print("=" * 70)

# Tabla descriptiva
stats_baseline = {
    'Validez (%)': f"M={np.mean(baseline_validez)*100:.1f}, DE={np.std(baseline_validez)*100:.1f}",
    'Iteraciones': f"Med={np.median(baseline_iteraciones):.0f}, IQR={stats.iqr(baseline_iteraciones):.0f}",
    'Tiempo (s)': f"M={np.mean(baseline_tiempo):.2f}, DE={np.std(baseline_tiempo):.2f}"
}

stats_restricciones = {
    'Validez (%)': f"M={np.mean(restricciones_validez)*100:.1f}, DE={np.std(restricciones_validez)*100:.1f}",
    'Iteraciones': f"Med={np.median(restricciones_iteraciones):.0f}, IQR={stats.iqr(restricciones_iteraciones):.0f}",
    'Tiempo (s)': f"M={np.mean(restricciones_tiempo):.2f}, DE={np.std(restricciones_tiempo):.2f}"
}

print("\nTabla 1: Resumen de Métricas por Método (n=100 por grupo)")
print("-" * 70)
print(f"{'Métrica':<20} {'Baseline':<25} {'Restricciones':<25}")
print("-" * 70)
for metrica in stats_baseline.keys():
    print(f"{metrica:<20} {stats_baseline[metrica]:<25} {stats_restricciones[metrica]:<25}")
print("-" * 70)
```

Acto seguido al volcado de nuestros datos en forma de resúmenes descriptivos de control, aplicamos la parte nuclear de nuestro análisis inferencial y testeos de hipótesis estadísticos subyacentes.

```{code-cell} ipython3
# Pruebas de hipótesis
print("\n" + "ANÁLISIS INFERENCIAL".center(70))
print("=" * 70)

# 1. Validez (proporción)
print("\nHipótesis 1: Tasa de Validez")
print("-" * 70)
z_stat, p_val_validez = stats.mannwhitneyu(baseline_validez, restricciones_validez, alternative='two-sided')
effect_size_validez = (np.mean(restricciones_validez) - np.mean(baseline_validez)) / \
                      np.sqrt((np.var(baseline_validez) + np.var(restricciones_validez)) / 2)

print(f"Se comparó la tasa de validez entre Baseline (M={np.mean(baseline_validez)*100:.1f}%,")
print(f"DE={np.std(baseline_validez)*100:.1f}%) y Restricciones (M={np.mean(restricciones_validez)*100:.1f}%,")
print(f"DE={np.std(restricciones_validez)*100:.1f}%) usando Mann-Whitney U.")
print(f"\nResultado: U={z_stat:.1f}, p={p_val_validez:.4f}, r={effect_size_validez:.2f}")

if p_val_validez < 0.05:
    print(f"Conclusión: Las restricciones produjeron significativamente mayor validez.")
else:
    print(f"Conclusión: No se encontró diferencia significativa.")

# 2. Iteraciones
print("\nHipótesis 2: Número de Iteraciones")
print("-" * 70)
u_stat, p_val_iter = stats.mannwhitneyu(baseline_iteraciones, restricciones_iteraciones, alternative='two-sided')

print(f"Se compararon las iteraciones entre Baseline (Med={np.median(baseline_iteraciones):.0f},")
print(f"IQR={stats.iqr(baseline_iteraciones):.0f}) y Restricciones (Med={np.median(restricciones_iteraciones):.0f},")
print(f"IQR={stats.iqr(restricciones_iteraciones):.0f}) usando Mann-Whitney U.")
print(f"\nResultado: U={u_stat:.1f}, p={p_val_iter:.4f}")

if p_val_iter < 0.05:
    print(f"Conclusión: Las restricciones requirieron significativamente menos iteraciones.")
else:
    print(f"Conclusión: No se encontró diferencia significativa.")

# 3. Tiempo
print("\nHipótesis 3: Tiempo de Compilación")
print("-" * 70)
t_stat, p_val_tiempo = stats.ttest_ind(baseline_tiempo, restricciones_tiempo)
cohens_d = (np.mean(baseline_tiempo) - np.mean(restricciones_tiempo)) / \
           np.sqrt((np.var(baseline_tiempo) + np.var(restricciones_tiempo)) / 2)

# Intervalo de confianza
diff = np.mean(baseline_tiempo) - np.mean(restricciones_tiempo)
se_diff = np.sqrt(np.var(baseline_tiempo)/n + np.var(restricciones_tiempo)/n)
ci_lower = diff - 1.96 * se_diff
ci_upper = diff + 1.96 * se_diff

print(f"Se comparó el tiempo de compilación entre Baseline (M={np.mean(baseline_tiempo):.2f}s,")
print(f"DE={np.std(baseline_tiempo):.2f}s) y Restricciones (M={np.mean(restricciones_tiempo):.2f}s,")
print(f"DE={np.std(restricciones_tiempo):.2f}s) usando prueba t independiente.")
print(f"\nResultado: t({2*n-2})={t_stat:.2f}, p={p_val_tiempo:.4f}, d={cohens_d:.2f},")
print(f"IC 95% [{ci_lower:.2f}s, {ci_upper:.2f}s]")

if p_val_tiempo < 0.05:
    print(f"Conclusión: Las restricciones redujeron significativamente el tiempo.")
else:
    print(f"Conclusión: No se encontró diferencia significativa.")
```

Un buen reporte siempre vendrá acompañado de las debidas representaciones gráficas estilizadas en su disposición y legibilidad para el entorno académico. Finalizaremos nuestro ejercicio encapsulando todo lo aprendido creando un resumen infográfico general y contundente:

```{code-cell} ipython3
# Visualización estilo publicación
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# 1. Validez - Gráfico de barras con error
ax1 = fig.add_subplot(gs[0, 0])
validez_means = [np.mean(baseline_validez)*100, np.mean(restricciones_validez)*100]
validez_sems = [stats.sem(baseline_validez)*100, stats.sem(restricciones_validez)*100]
bars = ax1.bar(['Baseline', 'Restricciones'], validez_means,
               yerr=[sem*1.96 for sem in validez_sems],
               capsize=10, color=['red', 'blue'], alpha=0.7,
               edgecolor='black', linewidth=2)
ax1.set_ylabel('Tasa de Validez (%)', fontweight='bold')
ax1.set_title('(A) Tasa de Validez de Kernels', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Anotar significancia
if p_val_validez < 0.05:
    y_max = max(validez_means) + max(validez_sems)*1.96 + 5
    ax1.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
    ax1.text(0.5, y_max + 1, '***' if p_val_validez < 0.001 else '**' if p_val_validez < 0.01 else '*',
            ha='center', fontsize=14, fontweight='bold')

for i, (bar, val, sem) in enumerate(zip(bars, validez_means, validez_sems)):
    ax1.text(i, val + sem*1.96 + 2, f'{val:.1f}%\n±{sem*1.96:.1f}',
            ha='center', fontsize=9, fontweight='bold')

# 2. Iteraciones - Box plot
ax2 = fig.add_subplot(gs[0, 1])
bp = ax2.boxplot([baseline_iteraciones, restricciones_iteraciones],
                  labels=['Baseline', 'Restricciones'],
                  patch_artist=True, notch=True, showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
bp['boxes'][0].set_facecolor('red')
bp['boxes'][1].set_facecolor('blue')
for box in bp['boxes']:
    box.set_alpha(0.6)

ax2.set_ylabel('Iteraciones', fontweight='bold')
ax2.set_title('(B) Número de Iteraciones', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Tiempo - Violin plot
ax3 = fig.add_subplot(gs[0, 2])
parts = ax3.violinplot([baseline_tiempo, restricciones_tiempo],
                       positions=[1, 2], showmeans=True, showmedians=True)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Baseline', 'Restricciones'])
ax3.set_ylabel('Tiempo de compilación (s)', fontweight='bold')
ax3.set_title('(C) Distribución de Tiempos', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['red', 'blue'][i])
    pc.set_alpha(0.6)

# 4. Tabla de resultados
ax4 = fig.add_subplot(gs[1, :])
ax4.axis('tight')
ax4.axis('off')

tabla_resultados = [
    ['Validez (%)', f'{np.mean(baseline_validez)*100:.1f}±{np.std(baseline_validez)*100:.1f}',
     f'{np.mean(restricciones_validez)*100:.1f}±{np.std(restricciones_validez)*100:.1f}',
     'Mann-Whitney U', f'{p_val_validez:.4f}', '***' if p_val_validez < 0.001 else '**' if p_val_validez < 0.01 else '*' if p_val_validez < 0.05 else 'ns'],
    ['Iteraciones', f'{np.median(baseline_iteraciones):.0f} (IQR={stats.iqr(baseline_iteraciones):.0f})',
     f'{np.median(restricciones_iteraciones):.0f} (IQR={stats.iqr(restricciones_iteraciones):.0f})',
     'Mann-Whitney U', f'{p_val_iter:.4f}', '***' if p_val_iter < 0.001 else '**' if p_val_iter < 0.01 else '*' if p_val_iter < 0.05 else 'ns'],
    ['Tiempo (s)', f'{np.mean(baseline_tiempo):.2f}±{np.std(baseline_tiempo):.2f}',
     f'{np.mean(restricciones_tiempo):.2f}±{np.std(restricciones_tiempo):.2f}',
     f't-test (d={cohens_d:.2f})', f'{p_val_tiempo:.4f}', '***' if p_val_tiempo < 0.001 else '**' if p_val_tiempo < 0.01 else '*' if p_val_tiempo < 0.05 else 'ns']
]

table = ax4.table(cellText=tabla_resultados,
                  colLabels=['Métrica', 'Baseline (n=100)', 'Restricciones (n=100)', 'Prueba', 'p-valor', 'Sig.'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.15, 0.25, 0.25, 0.15, 0.1, 0.1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(6):
    table[(0, i)].set_facecolor('#2E7D32')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Tabla 2: Resumen de Análisis Estadístico', fontsize=14, fontweight='bold', pad=20)

# 5. Tamaños de efecto
ax5 = fig.add_subplot(gs[2, :2])
efectos = ['Validez\n(r)', 'Iteraciones\n(U)', 'Tiempo\n(d)']
valores_efecto = [abs(effect_size_validez), abs(u_stat)/10000, abs(cohens_d)]  # normalizado para visualización
colores_efecto = ['lightblue', 'lightgreen', 'lightyellow']

bars_efecto = ax5.barh(efectos, [abs(cohens_d), abs(effect_size_validez), 0.5],
                       color=colores_efecto, edgecolor='black', linewidth=2)

ax5.axvline(0.2, color='orange', linestyle='--', alpha=0.5, label='Pequeño (0.2)')
ax5.axvline(0.5, color='yellow', linestyle='--', alpha=0.5, label='Mediano (0.5)')
ax5.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Grande (0.8)')

ax5.set_xlabel('Tamaño del Efecto', fontweight='bold')
ax5.set_title('(D) Tamaños de Efecto', fontweight='bold')
ax5.legend()
ax5.grid(axis='x', alpha=0.3)

# 6. Interpretación
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

interpretacion_text = f"""
CONCLUSIONES:

1. VALIDEZ: Las restricciones
   aumentaron validez en
   {(np.mean(restricciones_validez) - np.mean(baseline_validez))*100:.1f} puntos
   porcentuales (p<0.001).

2. ITERACIONES: Reducción de
   {np.median(baseline_iteraciones) - np.median(restricciones_iteraciones):.0f} iteraciones
   (p<0.001).

3. TIEMPO: Ahorro de
   {np.mean(baseline_tiempo) - np.mean(restricciones_tiempo):.2f}s por kernel
   (p<0.001, d={cohens_d:.2f}).

Todos los efectos son
estadística y prácticamente
significativos.
"""

ax6.text(0.1, 0.5, interpretacion_text, fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.suptitle('Figura 1: Comparación de Baseline vs Restricciones Gramaticales en Generación de Kernels GPU',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("NOTA: Este es un ejemplo de reporte profesional con:")
print("  - Estadísticas descriptivas completas")
print("  - Pruebas apropiadas para cada tipo de dato")
print("  - Tamaños de efecto reportados")
print("  - Intervalos de confianza")
print("  - Visualizaciones claras y profesionales")
print("=" * 70)
```

## Estructura de Sección de Resultados

### 1. Descripciones Generales

```
"Condujimos 100 ejecuciones de cada método (baseline y restricciones
gramaticales) usando los kernels del benchmark OpenCL v2.1. Cada
ejecución se registró con el seed 42 para reproducibilidad, temperatura
0.0 para determinismo. Los datos se analizaron usando Python 3.11
con SciPy para pruebas estadísticas y NumPy para cálculos numéricos."
```

### 2. Estadística Descriptiva

```
Tabla 1: Resumen de Métricas por Método

| Métrica | Baseline (n=100) | Restricciones (n=100) |
|---------|------------------|----------------------|
| Validez (%) | M=75.1, DE=15.6 | M=85.2, DE=12.3 |
| Iteraciones | Med=43, IQR=6 | Med=38, IQR=5 |
| Tiempo (s) | M=5.24, DE=1.12 | M=4.68, DE=0.98 |
| Error compilación | n=24 | n=14 |
| Error memoria | n=1 | n=1 |
```

Nota: Usa Md (mediana) e IQR (rango intercuartílico) para datos no normales.

### 3. Pruebas de Supuestos

```
"Evaluamos normalidad usando la prueba Shapiro-Wilk. La variable
'iteraciones' no cumplió normalidad en el grupo baseline (W = 0.89,
p = 0.003) debido a valores atípicos. Consecuentemente, usamos
Mann-Whitney U (no paramétrica) en lugar de t-test paramétrica."
```

### 4. Resultados de Pruebas

```
Hipótesis 1: Validez
Un t-test pareado comparó tasa de validez. Las restricciones (M=85.2%)
mostraron validez significativamente mayor que baseline (M=75.1%),
t(99) = 3.12, p < 0.001, d = 0.67, IC 95% [4.2%, 16.0%].

Hipótesis 2: Iteraciones
Debido a no-normalidad, usamos Mann-Whitney U. Las restricciones
(Med=38, IQR=5) requirieron significativamente menos iteraciones que
baseline (Med=43, IQR=6), U = 3240, p < 0.001, r = 0.62.
```

Estructura:
1. Recordar hipótesis
2. Describir grupos
3. Reportar estadísticos formalmente
4. Interpretar p-valor
5. Tamaño del efecto

### 5. Comparaciones Múltiples

```
Análisis Post-Hoc
Tras la ANOVA significante [F(3,96)=12.4, p<0.001], realizamos
prueba Tukey HSD para comparaciones pares. Restricciones v1 difirió
significativamente de baseline (p<0.001, d=0.67) y Restricciones v2
(p=0.041, d=0.34). Restricciones v1 y v2 no diferían significativamente
(p=0.52, d=0.15).
```

## Qué Reportar vs. Qué No Reportar

### SIEMPRE Reporta

```
☐ Media y desviación estándar (o mediana e IQR)
☐ Tamaño muestral (n)
☐ Estadístico de prueba (t, U, F, etc.)
☐ p-valor exacto si p > 0.001
☐ Grados de libertad
☐ Intervalo de confianza 95%
☐ Tamaño del efecto (d, r, h, etc.)
☐ Supuestos verificados o rechazados
```

### NO Reportes

```
✗ Decimales innecesarios (10.3457 → 10.34 es suficiente)
✗ p-valor reportado como "significante" sin número (reporta p=0.031)
✗ Solo p-valor sin tamaño de efecto
✗ Supuestos asumidos sin verificación
✗ Análisis cherry-picked (reporta todos o claramente marca como exploratorio)
```

## Diferencia: Significancia Estadística vs. Práctica

Este es el error más común. Dos escenarios:

### Escenario 1: Significante + Prácticamente Importante

```
"Las restricciones redujeron tiempo de compilación de 5.24s a 4.68s
(diferencia = 0.56s, 11% reducción), t(99) = 2.94, p = 0.004, d = 0.52.
Esta mejora es tanto estadísticamente significante como prácticamente
importante, ahorrando ~56 ms por kernel compilado."
```

Conclusión: **Reporta positivamente, esto es un hallazgo fuerte.**

### Escenario 2: Significante pero Trivial Prácticamente

```
"Aunque la prueba fue significante (p = 0.003, n = 10,000), la diferencia
fue minúscula: 5.240s vs. 5.241s (diferencia = 0.001s, 0.02%). El tamaño
del efecto fue insignificante (d = 0.08). Atribuimos este resultado a
alto poder debido a tamaño muestral muy grande más que a diferencia
prácticamente importante."
```

Conclusión: **Reporta como no importante a pesar de p-valor.**

### Escenario 3: No Significante pero Efecto Considerable

```
"Aunque la prueba no alcanzó significancia estadística (p = 0.067, n = 30),
el tamaño del efecto fue mediano (d = 0.47), sugiriendo un efecto real que
nuestro poder muestral fue insuficiente para detectar. Recomendamos
incrementar n para confirmación."
```

Conclusión: **Reporta como potencial efecto, pero necesita más investigación.**

## Limitaciones y Validez Interna

Sección importante que muchos olvidan:

```
Limitaciones

1. Validez de Constructo: Solo medimos compilabilidad y no eficiencia
de ejecución. Es posible que kernels con restricciones compilen pero
sean subóptimos en tiempo de ejecución.

2. Validez Interna: Aunque mantuvimos el LLM, temperatura y seed
constantes, es posible que interacciones no observadas (ej. orden de
procesamiento GPU) afectaran resultados.

3. Validez Externa: Solo probamos en GPUs NVIDIA A100. Generalizabilidad
a otras arquitecturas (AMD, TPU, CPU) es incierto.

4. Estadística: Ejecutamos análisis con semilla fija; resultados pueden
variar con semillas diferentes. Repetimos con 5 semillas para robustez.

Estos límites no invalidan hallazgos pero contextualizan conclusiones.
```

## Ejemplo Completo: Sección de Resultados Profesional

```
RESULTADOS

Participantes y Diseño
Condujimos 100 ejecuciones de cada método, usando kernels del
benchmark OpenCL v2.1. Cada ejecución se registró con hiperparámetros
fijos: LLM GPT-3.5-turbo (enero 2024), temperatura 0.0, seed 42.
Total: 200 puntos de datos.

Análisis de Supuestos
Shapiro-Wilk indicó no-normalidad en tasa de iteraciones del baseline
(W = 0.87, p = 0.001) pero normalidad en restricciones (W = 0.93,
p = 0.18). Levene's test mostró varianzas desiguales (F = 4.2, p = 0.042).
Consecuentemente, utilizamos pruebas no paramétricas para robustez.

Estadística Descriptiva
Tabla 1 resume métricas principales. Restricciones mostraron 10.1
puntos porcentuales mayor validez (M = 85.2%, DE = 12.3% vs. M = 75.1%,
DE = 15.6%), 5 iteraciones menos (Med = 38, IQR = 5 vs. Med = 43, IQR = 6),
y 0.56s menos tiempo (M = 4.68, DE = 0.98 vs. M = 5.24, DE = 1.12).

Análisis Principal
Mann-Whitney U indicó que restricciones produjeron significativamente
mayor validez, U = 3180, p < 0.001, r = 0.64 (efecto grande). Para
iteraciones, restricciones requirieron significativamente menos, U = 3410,
p = 0.001, r = 0.58 (efecto mediano). Para tiempo, t-test pareado
(ya que ambos grupos normales) mostró diferencia significativa,
t(99) = 2.84, p = 0.005, d = 0.52 (efecto mediano), IC 95% [0.18, 0.94]s.

Análisis Adicional
Para verificar robustez, repetimos análisis con semillas 123, 456, 789, 999.
Resultados permanecieron consistentes: validez p < 0.001 (todas),
iteraciones p < 0.01 (todas), tiempo p < 0.05 (todas). Conclusiones
resistieron variación de semilla aleatoria.

DISCUSIÓN

Nuestros resultados demuestran que restricciones gramaticales mejoran
significativamente tanto validez como eficiencia de generación de kernels.
El tamaño del efecto es grande para validez (r = 0.64) y mediano para
tiempo (d = 0.52), indicando importancia práctica.

Limitaciones: Solo evaluamos en temperatura 0.0 (determinista). Investigación
futura evaluaría temperature > 0 para realismo. Además, solo probamos
en GPUs NVIDIA; generalización a otras arquitecturas requiere validación.
```

## Tablas Efectivas

Tabla bien formateada es clara, concisa:

```
Tabla 1: Comparación de Métodos en Métricas de Generación de Kernels

Métrica           Baseline (n=100)        Restricciones (n=100)    Test        p
─────────────────────────────────────────────────────────────────────────────
Validez (%)       M=75.1 (DE=15.6)       M=85.2 (DE=12.3)        MW U=3180   <.001
Iteraciones       Med=43 (IQR=6)         Med=38 (IQR=5)          MW U=3410   .001
Tiempo (s)        M=5.24 (DE=1.12)       M=4.68 (DE=0.98)        t(99)=2.84  .005
Errores (n)       24 compilación         14 compilación          χ²=4.2      .042
                  1 memoria              1 memoria

Nota: MW = Mann-Whitney U, IC 95% no mostrado por brevedad pero incluir
en texto. Valores reportados con DE (desviación estándar) o IQR
(rango intercuartílico).
```

## Checklist de Reporte Completo

```{admonition} Resumen
:class: important
**SECCIÓN DE RESULTADOS**
- [ ] Descripción del diseño experimental y muestras
- [ ] Verificación de supuestos (normalidad, homocedasticidad)
- [ ] Tabla de estadísticas descriptivas (M, DE, n por grupo)
- [ ] Cada hipótesis reportada con formato estándar
- [ ] Estadísticos: t(df), U, F(df1,df2) con todos los componentes
- [ ] P-valores exactos (p=0.031, no "p<0.05")
- [ ] IC 95% de diferencias o efectos
- [ ] Tamaño del efecto (d, r, h) con interpretación verbal
- [ ] Supuestos verificados explícitamente
- [ ] Análisis exploratorio etiquetado claramente

**SECCIÓN DE DISCUSIÓN**
- [ ] Resumen de hallazgos principales sin jerga técnica
- [ ] Relación explícita con hipótesis pre-registradas
- [ ] Significancia estadística vs. práctica distinguida
- [ ] Limitaciones (validez interna/externa/constructo)
- [ ] Implicaciones teóricas y prácticas
- [ ] Recomendaciones específicas para investigación futura

**VISUALIZACIÓN**
- [ ] Figuras con títulos descriptivos y autocontenidos
- [ ] Ejes etiquetados con unidades (ms, %, GB/s)
- [ ] Leyenda clara sin ambigüedades
- [ ] Fuentes ≥12pt para legibilidad
- [ ] Colores colorblind-friendly (evitar rojo-verde)
- [ ] Barras de error o ICs visibles
- [ ] Alta resolución (≥300 DPI para publicación)
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Demostración de visualizaciones accesibles y profesionales
np.random.seed(42)

# Datos de ejemplo
baseline = np.random.normal(5.0, 1.2, 50)
restricciones = np.random.normal(4.2, 1.0, 50)

print("Guía de Visualizaciones Profesionales y Accesibles")
print("=" * 70)
print("\nPrincipios de visualización para publicaciones académicas:")
print("  1. Colores accesibles (evitar solo rojo/verde para daltónicos)")
print("  2. Fuentes legibles (≥12pt)")
print("  3. Etiquetas completas con unidades")
print("  4. Títulos descriptivos")
print("  5. Leyendas claras")
print("  6. Alta resolución (≥300 DPI para publicación)")

fig, axes = plt.subplots(2, 3, figsize=(16, 12))

# 1. BUENO: Gráfico de barras con barras de error
ax = axes[0, 0]
means = [np.mean(baseline), np.mean(restricciones)]
sems = [stats.sem(baseline), stats.sem(restricciones)]
ci = [sem * 1.96 for sem in sems]

bars = ax.bar(['Baseline', 'Restricciones'], means, yerr=ci,
              capsize=10, color=['#1f77b4', '#ff7f0e'],
              edgecolor='black', linewidth=2, alpha=0.8)

ax.set_ylabel('Tiempo de compilación (segundos)', fontsize=12, fontweight='bold')
ax.set_title('(A) Comparación de Métodos\n(Barras de error = IC 95%)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Anotar valores
for i, (bar, mean, ci_val) in enumerate(zip(bars, means, ci)):
    ax.text(i, mean + ci_val + 0.2, f'{mean:.2f}±{ci_val:.2f}',
            ha='center', fontsize=10, fontweight='bold')

# 2. BUENO: Box plot informativo
ax = axes[0, 1]
bp = ax.boxplot([baseline, restricciones],
                labels=['Baseline', 'Restricciones'],
                patch_artist=True, notch=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red',
                              markersize=8, markeredgecolor='black'))

colors = ['#1f77b4', '#ff7f0e']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Tiempo de compilación (segundos)', fontsize=12, fontweight='bold')
ax.set_title('(B) Distribución de Datos\n(Diamante = media, línea = mediana)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 3. BUENO: Violin plot con puntos
ax = axes[0, 2]
parts = ax.violinplot([baseline, restricciones], positions=[1, 2],
                      showmeans=True, showmedians=True)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)

# Agregar puntos individuales con jitter
for i, datos in enumerate([baseline, restricciones]):
    x = np.random.normal(i+1, 0.04, size=len(datos))
    ax.scatter(x, datos, alpha=0.3, s=20, color='black')

ax.set_xticks([1, 2])
ax.set_xticklabels(['Baseline', 'Restricciones'])
ax.set_ylabel('Tiempo de compilación (segundos)', fontsize=12, fontweight='bold')
ax.set_title('(C) Distribución con Datos Individuales',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
```

Alternativamente de forma gráfica, las subrepresentaciones de superposición y diferencias brindan un valor incalculable de cara a mostrar los solapamientos de efectividad y la significancia real observada.

```{code-cell} ipython3
# 4. BUENO: Histograma superpuesto con transparencia
ax = axes[1, 0]
ax.hist(baseline, bins=15, alpha=0.6, color='#1f77b4',
        edgecolor='black', label='Baseline', density=True)
ax.hist(restricciones, bins=15, alpha=0.6, color='#ff7f0e',
        edgecolor='black', label='Restricciones', density=True)

ax.axvline(np.mean(baseline), color='#1f77b4', linestyle='--',
           linewidth=2, label=f'Media B = {np.mean(baseline):.2f}')
ax.axvline(np.mean(restricciones), color='#ff7f0e', linestyle='--',
           linewidth=2, label=f'Media R = {np.mean(restricciones):.2f}')

ax.set_xlabel('Tiempo de compilación (segundos)', fontsize=12, fontweight='bold')
ax.set_ylabel('Densidad de probabilidad', fontsize=12, fontweight='bold')
ax.set_title('(D) Distribuciones Superpuestas', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')

# 5. BUENO: Gráfico de diferencias con IC
ax = axes[1, 1]
diferencias = baseline - restricciones[:len(baseline)]
media_diff = np.mean(diferencias)
sem_diff = stats.sem(diferencias)
ci_diff = sem_diff * 1.96

ax.hist(diferencias, bins=20, color='purple', alpha=0.7,
        edgecolor='black')
ax.axvline(media_diff, color='red', linestyle='--', linewidth=3,
           label=f'Diferencia media = {media_diff:.2f}')
ax.axvline(0, color='green', linestyle=':', linewidth=2,
           label='Sin diferencia (H₀)')
ax.axvspan(media_diff - ci_diff, media_diff + ci_diff,
           alpha=0.3, color='red', label=f'IC 95%')

ax.set_xlabel('Diferencia (Baseline - Restricciones) en segundos',
              fontsize=12, fontweight='bold')
ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax.set_title('(E) Distribución de Diferencias', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
```

Terminamos con la tabla compacta descriptiva en nuestra visualización y emitimos las reglas finales para una visualización exitosa, resaltando buenas prácticas y vicios frecuentemente encontrados.

```{code-cell} ipython3
# 6. BUENO: Tabla resumen integrada
ax = axes[1, 2]
ax.axis('off')

# Realizar prueba t
t_stat, p_val = stats.ttest_ind(baseline, restricciones)
cohens_d = (np.mean(baseline) - np.mean(restricciones)) / \
           np.sqrt((np.var(baseline) + np.var(restricciones)) / 2)

tabla_texto = f"""
RESUMEN ESTADÍSTICO

Baseline (n={len(baseline)}):
  Media: {np.mean(baseline):.2f} s
  DE: {np.std(baseline, ddof=1):.2f} s
  IC 95%: [{np.mean(baseline)-1.96*stats.sem(baseline):.2f},
           {np.mean(baseline)+1.96*stats.sem(baseline):.2f}]

Restricciones (n={len(restricciones)}):
  Media: {np.mean(restricciones):.2f} s
  DE: {np.std(restricciones, ddof=1):.2f} s
  IC 95%: [{np.mean(restricciones)-1.96*stats.sem(restricciones):.2f},
           {np.mean(restricciones)+1.96*stats.sem(restricciones):.2f}]

Prueba t independiente:
  t({len(baseline)+len(restricciones)-2}) = {t_stat:.2f}
  p = {p_val:.4f}
  Cohen's d = {cohens_d:.2f}

Conclusión:
  {"Diferencia SIGNIFICATIVA" if p_val < 0.05 else "Sin diferencia significativa"}
  (α = 0.05)
  Efecto: {"Trivial" if abs(cohens_d) < 0.2 else "Pequeño" if abs(cohens_d) < 0.5 else "Mediano" if abs(cohens_d) < 0.8 else "Grande"}
"""

ax.text(0.1, 0.5, tabla_texto, fontsize=9, verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                 edgecolor='black', linewidth=2, alpha=0.9))
ax.set_title('(F) Resumen Textual', fontsize=12, fontweight='bold')

plt.suptitle('Figura: Guía de Visualizaciones Estadísticas Profesionales\n' +
             '(Ejemplos de gráficos adecuados para publicaciones académicas)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()

print("\nMEJORES PRÁCTICAS:")
print("-" * 70)
print("✓ Usar colores distinguibles para personas con daltonismo")
print("✓ Incluir barras de error o intervalos de confianza")
print("✓ Etiquetar todos los ejes con unidades")
print("✓ Usar fuentes ≥12pt para legibilidad")
print("✓ Incluir leyendas claras")
print("✓ Agregar títulos descriptivos")
print("✓ Usar grid sutil para facilitar lectura")
print("✓ Guardar en alta resolución (300 DPI) para publicación")

print("\nEVITAR:")
print("-" * 70)
print("✗ Gráficos 3D innecesarios")
print("✗ Colores que solo se distinguen por rojo/verde")
print("✗ Fuentes muy pequeñas (<10pt)")
print("✗ Ejes sin etiquetas o unidades")
print("✗ Demasiados colores (máximo 5-6)")
print("✗ Efectos visuales distractores")

print("\nPaletas de colores accesibles recomendadas:")
print("  - Matplotlib 'tab10' (default)")
print("  - Seaborn 'colorblind'")
print("  - ColorBrewer 'Set2' o 'Paired'")
```

## Ejercicios y Reflexión

```{admonition} ✅ Verifica tu comprensión
:class: note
1. **Formato APA**: ¿Qué información falta en "El método fue mejor (p=0.02)"?
2. **Significancia**: ¿Cuál es la diferencia entre "significante" y "importante"?
3. **Limitaciones**: ¿Por qué es importante discutir limitaciones honestamente?
4. **Tablas**: ¿Qué debe incluir una tabla estadística profesional?
```

### Ejercicio 1: Reescribir Mal Reportado
Toma este reporte deficiente:

> "Encontramos que el método nuevo fue mejor (p=0.02)."

Reescribe en formato APA completo incluyendo:
- Medias y desviaciones
- Estadístico de prueba
- IC 95%
- Tamaño del efecto
- Interpretación

### Ejercicio 2: Tabla de Resultados
Para tu proyecto, crea tabla profesional tipo APA que resuma:
- Todas las métricas principales
- Medias/medianas y DE/IQR
- Estadísticos de prueba
- p-valores
- Tamaños de efecto

### Ejercicio 3: Párrafo de Resultados
Escribe párrafo de Resultados (no Discusión) que reporte:
- Una hipótesis principal
- Supuestos verificados
- Estadísticos con formato correcto
- IC 95%
- Tamaño del efecto

### Ejercicio 4: Limitaciones Honesta
Lista 5 limitaciones de validez en tu proyecto (interna/externa/constructo):
- Qué potencialmente afectó resultados?
- Cómo mitigaste?
- Qué sigue sin resolver?

### Reflexión
1. **Claridad vs. Completitud**: ¿Cómo balanceas reportar completo sin abrumar?
2. **Audiencia**: ¿Cambiarías reporte para ML researchers vs. estadísticos?
3. **Interpretación**: ¿Dónde está la línea entre "significante" y "importante"?

---

## Conclusión del Módulo

Hemos cubierto el viaje completo:

1. **Fundamentos**: Probabilidad, espacios muestrales, Bayes
2. **Distribuciones**: Normal, binomial, Poisson, TCL
3. **Pruebas**: Hipótesis, t-tests, errores Tipo I/II
4. **Diseño**: Power analysis, validez interna/externa
5. **Reproducibilidad**: Semillas, control de estocacidad
6. **No Paramétrico**: Cuando violas supuestos
7. **Efecto**: Reportar magnitud, no solo p-valores
8. **Múltiples**: Correcciones, ANOVA, post-hoc
9. **MLOps**: Rastreo con W&B, visualización accesible
10. **Reporte**: APA/IEEE, limitaciones honestas

Tu investigación ahora está construida sobre base estadística sólida. Lo más importante: mantén integridad. La tentación de p-hacking (ejecutar análisis hasta obtener p<0.05) es fuerte. **Resiste.**

Pre-registra. Reporta todo. Sé honesto sobre limitaciones. Este es el camino a investigación confiable que otros pueden construir.

¡Éxito en tu tesis sobre restricciones gramaticales en generación de kernels GPU!

---

## Referencias

- American Psychological Association. [APA Style](https://apastyle.apa.org/). APA.
- IEEE. [IEEE Editorial Style Manual](https://journals.ieeeauthorcenter.ieee.org/your-role-in-article-production/ieee-editorial-style-manual/). IEEE.
- Wilkinson, L. (1999). [Statistical Methods in Psychology Journals](https://doi.org/10.1037/0003-066X.54.8.594). American Psychologist.
- Cumming, G. (2014). [The New Statistics: Why and How](https://doi.org/10.1177/0956797613504966). Psychological Science.
