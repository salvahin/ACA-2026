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

# Lectura 8: MLOps

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q mlflow hydra-core omegaconf seaborn
    print('Dependencias instaladas!')
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/08_mlops_visualizacion.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Explicar qué es MLOps, por qué es importante y su ciclo de vida
- Implementar tracking sistemático de experimentos usando MLflow
- Versionar configuraciones de experimentos con archivos YAML y Hydra
- Configurar MLflow con Databricks para tracking escalable
- Aplicar checklists de MLOps para garantizar reproducibilidad (git commit, seeds, config)
- Comprender la relación entre MLOps y LLMOps para modelos de lenguaje
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[MLOps in 10 Minutes (DataTalksClub)](https://www.youtube.com/watch?v=7kZrrCsJdEM)** - Diagramas de arquitectura del ciclo de vida de un modelo de IA en producción.
```


## Introducción

Has entrenado modelos, generado código, evaluado resultados. Pero, ¿cómo **rastrear** todo? ¿Cómo saber qué hiperparámetros usaste hace 3 semanas? ¿Cómo comparar 50 experimentos?

**MLOps (Machine Learning Operations)** resuelve esto: logging sistemático, versionado de experimentos, y gestión del ciclo de vida de modelos. Esta lectura te da las herramientas para investigación reproducible.

:::{figure} diagrams/rag_pipeline.png
:name: fig-rag-pipeline
:alt: Pipeline RAG mostrando recuperación de documentos e integración con el LLM
:align: center
:width: 90%

**Figura 1:** Pipeline RAG (Retrieval-Augmented Generation): un patrón LLMOps clave donde la base de conocimiento externa se indexa como vectores, se recuperan fragmentos relevantes en cada consulta, y se inyectan como contexto al LLM para respuestas fundamentadas y actualizadas.
:::

### ¿Qué es MLOps?

```{admonition} 📚 Definición
:class: important
**MLOps** es la práctica de aplicar principios de DevOps al desarrollo y despliegue de modelos de Machine Learning.

Objetivo: Hacer que el desarrollo de ML sea **reproducible**, **escalable** y **mantenible**.
```

```
MLOps = ML + DevOps

DevOps tradicional:
  Código → Build → Test → Deploy → Monitor

MLOps:
  Datos → Feature Engineering → Entrenamiento → Validación → Deploy → Monitor
                                       ↑
                          Tracking de experimentos
                          Versionado de modelos
                          Reproducibilidad
```

### Ciclo de Vida de MLOps

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

# Visualización del ciclo de vida MLOps
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Definir las etapas del ciclo
etapas = [
    ('Datos', (2, 8), 'lightblue'),
    ('Feature\nEngineering', (5, 9), 'lightgreen'),
    ('Entrenamiento', (8, 8), 'lightyellow'),
    ('Evaluación', (9, 5), 'lightsalmon'),
    ('Registro', (8, 2), 'lightpink'),
    ('Deploy', (5, 1), 'lightcyan'),
    ('Monitoreo', (2, 2), 'lavender'),
]

# Dibujar nodos
for nombre, (x, y), color in etapas:
    circle = plt.Circle((x, y), 0.8, color=color, ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, nombre, ha='center', va='center', fontsize=9, weight='bold')

# Dibujar flechas
flechas = [
    ((2, 8), (5, 9)),
    ((5, 9), (8, 8)),
    ((8, 8), (9, 5)),
    ((9, 5), (8, 2)),
    ((8, 2), (5, 1)),
    ((5, 1), (2, 2)),
    ((2, 2), (2, 8)),  # Loop back
]

for (x1, y1), (x2, y2) in flechas:
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/length * 0.8, dy/length * 0.8
    ax.annotate('', xy=(x2 - dx, y2 - dy), xytext=(x1 + dx, y1 + dy),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Texto central
ax.text(5, 5, 'MLOps\nCiclo de Vida', ha='center', va='center',
        fontsize=14, weight='bold', color='darkblue')

ax.set_title('Ciclo de Vida de MLOps', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

print("Etapas del Ciclo de Vida:")
print("-" * 50)
print("1. DATOS: Recolección, validación, versionado")
print("2. FEATURE ENGINEERING: Transformación, selección")
print("3. ENTRENAMIENTO: Experimentos, hiperparámetros")
print("4. EVALUACIÓN: Métricas, validación cruzada")
print("5. REGISTRO: Model registry, versionado")
print("6. DEPLOY: Serving, API, batch inference")
print("7. MONITOREO: Drift detection, performance")
```

### MLOps vs LLMOps

```{admonition} 🔗 LLMOps para Modelos de Lenguaje
:class: seealso
**LLMOps** extiende MLOps para las necesidades específicas de Large Language Models:

| Aspecto | MLOps Tradicional | LLMOps |
|---------|-------------------|--------|
| **Datos** | Datasets tabulares | Corpus de texto, prompts |
| **Entrenamiento** | Desde cero | Fine-tuning, prompt engineering |
| **Evaluación** | Accuracy, F1 | Perplexity, humanos, LLM-as-Judge |
| **Deploy** | API REST | Streaming, context windows |
| **Costo** | GPU hours | Tokens, $/1M tokens |
| **Monitoreo** | Data drift | Prompt injection, alucinaciones |

Las herramientas que aprenderás (MLflow) aplican a ambos paradigmas.
```

---

## Parte 1: El Problema del Tracking

:::{figure} images/AI_08_01_MLOps_Tracking_Componentes.jpeg
:name: fig-mlops-tracking
:alt: Componentes de MLOps tracking
:align: center
:width: 90%

**Figura 2:** Componentes del Tracking en MLOps - experimentos, parámetros, métricas, artefactos y versiones de modelos.
:::

### Escenario Común

```
Semana 1:
  - Entrené modelo con lr=0.001, batch=32
  - Resultados: 78% accuracy
  - Guardé en: model_v1.pt

Semana 3:
  - Entrené con lr=0.0001, batch=64
  - Resultados: 82% accuracy
  - Guardé en: model_final_v2_really_final.pt

Semana 5:
  - ¿Cuál era la configuración de model_v1?
  - ¿Por qué v2 es mejor?
  - ¿Qué datos usé?
  → No sé 😱
```

### Lo Que Necesitas Rastrear

```
1. CÓDIGO
   - Versión de git commit
   - Dependencias (requirements.txt)

2. DATOS
   - Dataset usado
   - Splits train/val/test
   - Preprocesamiento

3. CONFIGURACIÓN
   - Hiperparámetros
   - Arquitectura
   - Seeds aleatorias

4. MÉTRICAS
   - Loss por época
   - Métricas de evaluación
   - Tiempo de entrenamiento

5. ARTEFACTOS
   - Checkpoints del modelo
   - Gráficas generadas
   - Predicciones de ejemplo
```

```{admonition} 🤔 Reflexiona
:class: hint
¿Qué información perderías si no trackearas el git commit? Imagina que quieres reproducir un experimento de hace 3 meses pero el código cambió 50 veces desde entonces. ¿Cómo sabrías qué versión usar?
```

---

## Parte 2: MLflow - Tracking de Experimentos

**MLflow** es la herramienta estándar de la industria para tracking de experimentos de ML. Es open source y se integra perfectamente con Databricks.

### ¿Por Qué MLflow?

```
Ventajas de MLflow:
1. OPEN SOURCE: Sin costos de licencia
2. SELF-HOSTED: Control total sobre tus datos
3. DATABRICKS NATIVE: Integración perfecta con la plataforma
4. FLEXIBLE: Funciona con cualquier framework (PyTorch, TF, sklearn)
5. MODEL REGISTRY: Versionado y staging de modelos
```

### Setup Básico

```{code-cell} ipython3
:tags: [skip-execution]

import mlflow

# Opción 1: Tracking local (archivos)
mlflow.set_tracking_uri("./mlruns")

# Opción 2: Tracking server
# mlflow.set_tracking_uri("http://localhost:5000")

# Opción 3: Databricks (veremos después)
# mlflow.set_tracking_uri("databricks")

# Crear/seleccionar experimento
mlflow.set_experiment("kernel-generation")

# Iniciar un run
with mlflow.start_run(run_name="experiment-001"):
    # Log parámetros
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("model", "codellama-7b")
    mlflow.log_param("batch_size", 32)

    # Log métricas (con step para series temporales)
    for epoch in range(10):
        train_loss = 1.0 / (epoch + 1)  # Simulación
        mlflow.log_metric("train_loss", train_loss, step=epoch)

    # Log artefactos (archivos)
    # mlflow.log_artifact("model.pt")

    # Log modelo completo (con firma)
    # mlflow.pytorch.log_model(model, "model")

print("Experimento registrado en MLflow!")
```

### Estructura de un Experimento MLflow

```
mlruns/
├── 0/                          # Experimento "Default"
├── 1/                          # Experimento "kernel-generation"
│   ├── meta.yaml               # Metadatos del experimento
│   ├── abc123/                 # Run ID
│   │   ├── meta.yaml           # Metadatos del run
│   │   ├── params/             # Parámetros guardados
│   │   │   ├── learning_rate
│   │   │   ├── model
│   │   │   └── batch_size
│   │   ├── metrics/            # Métricas por step
│   │   │   └── train_loss
│   │   ├── artifacts/          # Archivos (modelos, plots, etc.)
│   │   │   └── model/
│   │   └── tags/               # Tags del run
│   └── def456/                 # Otro run
└── ...
```

### Logging Completo de un Experimento

```{code-cell} ipython3
:tags: [skip-execution]

import mlflow
import mlflow.pytorch
import torch

def train_with_mlflow(config):
    """Ejemplo completo de entrenamiento con MLflow tracking"""

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=config["run_name"]):
        # 1. Log todos los parámetros
        mlflow.log_params({
            "model_name": config["model"],
            "learning_rate": config["lr"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "optimizer": config["optimizer"],
            "seed": config["seed"],
        })

        # 2. Log tags para organización
        mlflow.set_tags({
            "team": "nlp-research",
            "project": "kernel-generation",
            "git_commit": get_git_hash(),
        })

        # 3. Entrenar (simulado)
        for epoch in range(config["epochs"]):
            train_loss = 1.0 / (epoch + 1)
            val_loss = 1.2 / (epoch + 1)

            # Log métricas por época
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": config["lr"] * (0.95 ** epoch),
            }, step=epoch)

        # 4. Log métricas finales
        mlflow.log_metrics({
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
        })

        # 5. Log modelo
        # mlflow.pytorch.log_model(model, "model")

        # 6. Log artefactos adicionales
        # mlflow.log_artifact("training_curve.png")

        print(f"Run completado: {mlflow.active_run().info.run_id}")

# Configuración de ejemplo
config = {
    "experiment_name": "kernel-generation",
    "run_name": "baseline-v1",
    "model": "codellama-7b",
    "lr": 1e-4,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "adamw",
    "seed": 42,
}

# train_with_mlflow(config)
```

### Comparar Experimentos

```{code-cell} ipython3
:tags: [skip-execution]

import mlflow
import pandas as pd

# Obtener todos los runs de un experimento
experiment = mlflow.get_experiment_by_name("kernel-generation")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Convertir a DataFrame para análisis
print("Comparación de Experimentos:")
print(runs[["run_id", "params.learning_rate", "params.model",
            "metrics.final_val_loss", "status"]].head(10))

# Encontrar el mejor run
best_run = runs.loc[runs["metrics.final_val_loss"].idxmin()]
print(f"\nMejor run: {best_run['run_id']}")
print(f"Val Loss: {best_run['metrics.final_val_loss']:.4f}")
```

---

## Parte 3: MLflow con Databricks

**Databricks** es una plataforma unificada de data + AI que incluye MLflow de forma nativa.

### Setup de Databricks

```{code-cell} ipython3
:tags: [skip-execution]

# En un notebook de Databricks, MLflow está pre-configurado
import mlflow

# El tracking URI ya apunta a Databricks
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Los experimentos se guardan en el Workspace
mlflow.set_experiment("/Users/tu-email@empresa.com/kernel-generation")

# El resto del código es idéntico
with mlflow.start_run():
    mlflow.log_param("model", "codellama-7b")
    mlflow.log_metric("accuracy", 0.95)
```

### Model Registry en Databricks

```{code-cell} ipython3
:tags: [skip-execution]

import mlflow

# Registrar un modelo en el Model Registry
model_uri = f"runs:/{run_id}/model"
model_name = "kernel-generator"

# Registrar nueva versión
mlflow.register_model(model_uri, model_name)

# Transicionar a staging/production
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Mover a Staging
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# Después de validación, mover a Production
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)
```

### Ejercicio Práctico: MLflow con Databricks

```{admonition} 🛠️ Ejercicio: Tracking en Databricks
:class: tip
**Objetivo:** Configurar un pipeline completo de MLflow en Databricks.

**Pasos:**
1. Crear un workspace en Databricks Community Edition (gratis)
2. Crear un notebook y configurar un experimento
3. Entrenar un modelo simple (sklearn o pytorch)
4. Log parámetros, métricas y el modelo
5. Registrar el modelo en el Model Registry
6. Comparar diferentes runs en la UI

**Código base:**
```python
# En Databricks notebook
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Crear datos de ejemplo
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Configurar experimento
mlflow.set_experiment("/Users/tu-email/mi-experimento")

# Entrenar con diferentes configuraciones
for n_estimators in [10, 50, 100]:
    with mlflow.start_run(run_name=f"rf-{n_estimators}"):
        # Log parámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("algorithm", "RandomForest")

        # Entrenar
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # Evaluar
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log modelo
        mlflow.sklearn.log_model(model, "model")

        print(f"n_estimators={n_estimators}, accuracy={accuracy:.4f}")
```
```

---

## Parte 4: Versionado de Experimentos

### Estructura de Directorios

```
experiments/
├── 2024-03-01_baseline/
│   ├── config.yaml
│   ├── metrics.json
│   ├── model.pt
│   └── logs/
├── 2024-03-05_grammar-v1/
│   ├── config.yaml
│   ├── metrics.json
│   ├── model.pt
│   └── logs/
└── 2024-03-10_grammar-v2/
    ├── config.yaml
    ├── metrics.json
    ├── model.pt
    └── logs/
```

### Config Files (YAML)

```yaml
# config.yaml
experiment:
  name: "grammar-v2"
  date: "2024-03-10"
  git_commit: "abc123"

model:
  name: "codellama-7b"
  quantization: "int8"

training:
  learning_rate: 0.0001
  batch_size: 64
  epochs: 20

generation:
  temperature: 0.1
  max_tokens: 512
  grammar:
    enabled: true
    version: "L1-L4-v2"

data:
  train_path: "data/train.json"
  val_path: "data/val.json"

seeds:
  numpy: 42
  torch: 42
  python: 42
```

### Hydra para Configuración

```{code-cell} ipython3
:tags: [skip-execution]

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig):
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Model: {cfg.model.name}")

    # Hydra guarda automáticamente la config usada
    # en outputs/YYYY-MM-DD/HH-MM-SS/.hydra/

if __name__ == "__main__":
    train()
```

---

## Parte 5: Visualización de Resultados

### Principios de Visualización

```
1. CLARIDAD sobre decoración
   - Un mensaje por gráfica
   - Eliminar elementos innecesarios

2. COMPARACIÓN efectiva
   - Misma escala para comparar
   - Agrupar relacionados

3. HONESTIDAD
   - Mostrar incertidumbre (barras de error)
   - No truncar ejes engañosamente

4. ACCESIBILIDAD
   - Colores distinguibles (colorblind-friendly)
   - Etiquetas legibles
```

### Gráficas Comunes en ML

**1. Curvas de Entrenamiento**

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
epochs = np.arange(1, 21)
train_loss = 2.0 * np.exp(-0.1 * epochs) + np.random.normal(0, 0.05, 20)
val_loss = 2.0 * np.exp(-0.08 * epochs) + np.random.normal(0, 0.08, 20)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_loss, label='Train', color='blue')
ax.plot(epochs, val_loss, label='Validation', color='orange')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Progress')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

**2. Comparación de Métodos (Boxplot)**

```{code-cell} ipython3
import seaborn as sns
import pandas as pd

# Datos de ejemplo
baseline_speedups = np.random.normal(0.95, 0.15, 50)
grammar_speedups = np.random.normal(1.25, 0.20, 50)

data = {
    'Method': ['Baseline']*50 + ['Grammar']*50,
    'Speedup': np.concatenate([baseline_speedups, grammar_speedups])
}
df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Method', y='Speedup', data=df, ax=ax)
ax.set_title('Speedup Distribution: Baseline vs Grammar')
ax.axhline(y=1.0, color='red', linestyle='--', label='PyTorch baseline')
ax.legend()
plt.show()
```

**3. Heatmap de Resultados**

```{code-cell} ipython3
# Resultados por configuración
results = np.array([
    [0.85, 0.82, 0.79],  # temp=0.1
    [0.80, 0.78, 0.75],  # temp=0.5
    [0.72, 0.70, 0.68],  # temp=1.0
])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    results,
    annot=True,
    fmt='.2f',
    xticklabels=['L1-L2', 'L1-L3', 'L1-L4'],
    yticklabels=['T=0.1', 'T=0.5', 'T=1.0'],
    cmap='YlGnBu',
    ax=ax
)
ax.set_xlabel('Grammar Level')
ax.set_ylabel('Temperature')
ax.set_title('Correctness Rate by Configuration')
plt.show()
```

**4. Scatter con Tendencia**

```{code-cell} ipython3
# Datos de ejemplo
tokens_generated = np.random.randint(50, 500, 100)
execution_time = 0.5 * tokens_generated + np.random.normal(0, 20, 100)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(tokens_generated, execution_time, alpha=0.6)

# Línea de tendencia
z = np.polyfit(tokens_generated, execution_time, 1)
p = np.poly1d(z)
ax.plot(tokens_generated, p(tokens_generated), "r--", label='Trend')

ax.set_xlabel('Tokens Generated')
ax.set_ylabel('Execution Time (ms)')
ax.set_title('Generation Cost vs Token Count')
ax.legend()
plt.show()
```

### Paletas de Colores Accesibles

```{code-cell} ipython3
# Paleta colorblind-friendly
colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE']

# O usar seaborn
sns.set_palette('colorblind')

# Verificar accesibilidad:
# - coblis.org (simulador de daltonismo)
# - Evitar rojo-verde como único diferenciador

# Ejemplo de visualización
fig, ax = plt.subplots(figsize=(8, 4))
for i, color in enumerate(colors):
    ax.barh(i, 1, color=color, label=f'Color {i+1}')
ax.set_yticks(range(len(colors)))
ax.set_yticklabels([f'Color {i+1}' for i in range(len(colors))])
ax.set_title('Colorblind-Friendly Palette')
ax.legend()
plt.show()
```

---

## Parte 6: Reportes Automatizados

### Generación de Reportes

```python
def generate_experiment_report(run_dir: str) -> str:
    """Genera reporte Markdown de un experimento."""

    config = load_config(f"{run_dir}/config.yaml")
    metrics = load_json(f"{run_dir}/metrics.json")

    report = f"""
# Experiment Report: {config['experiment']['name']}

## Configuration
- Model: {config['model']['name']}
- Learning Rate: {config['training']['learning_rate']}
- Grammar: {config['generation']['grammar']['version']}

## Results
- Final Accuracy: {metrics['accuracy']:.2%}
- Average Speedup: {metrics['speedup']:.2f}x
- Success Rate: {metrics['success_rate']:.2%}

## Visualizations
![Training Curve](training_curve.png)
![Results Comparison](comparison.png)

## Notes
{load_notes(run_dir)}
"""
    return report
```

### Dashboard con Streamlit

```python
import streamlit as st

st.title("Kernel Generation Experiments")

# Selector de experimentos
experiments = list_experiments()
selected = st.selectbox("Select experiment", experiments)

# Mostrar métricas
metrics = load_metrics(selected)
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("Speedup", f"{metrics['speedup']:.2f}x")
col3.metric("Success Rate", f"{metrics['success_rate']:.2%}")

# Gráfica interactiva
fig = create_training_plot(selected)
st.plotly_chart(fig)
```

---

## Parte 7: Checklist de MLOps

### Antes del Experimento

```
□ Versión de código commiteada (git)
□ Config file creado
□ Seeds definidas para reproducibilidad
□ Espacio en disco verificado
□ wandb/mlflow inicializado
```

### Durante el Experimento

```
□ Logging de métricas cada N steps
□ Checkpoints periódicos
□ Monitoreo de recursos (GPU memory, CPU)
□ Early stopping configurado
```

### Después del Experimento

```
□ Métricas finales guardadas
□ Mejor modelo guardado
□ Gráficas generadas
□ Notas escritas (qué funcionó, qué no)
□ Comparación con baseline documentada
```

---

## Ejercicios Prácticos

### Ejercicio 1: Setup wandb

Configura wandb para tu proyecto y loguea un experimento simple.

### Ejercicio 2: Config con Hydra

Crea una estructura de configs para experimentos de generación de kernels.

### Ejercicio 3: Dashboard

Crea un dashboard simple en Streamlit que muestre resultados de múltiples experimentos.

---

## Preguntas de Reflexión

1. ¿Qué información perderías si no usaras tracking de experimentos?

2. ¿Cómo decidirías entre wandb y MLflow para tu proyecto?

3. ¿Qué visualizaciones serían más útiles para comunicar resultados de KernelBench?

---

## Recursos

- **Weights & Biases Docs**: docs.wandb.ai
- **MLflow Docs**: mlflow.org/docs
- **Hydra**: hydra.cc
- **Streamlit**: streamlit.io
- **Matplotlib Best Practices**: matplotlib.org/stable/tutorials

---

## Errores Comunes

```{admonition} ⚠️ Errores frecuentes
:class: warning

1. **No trackear git commit**: Olvidas qué versión del código usaste. Siempre guarda `git rev-parse HEAD` en config.
2. **Seeds no fijadas**: Resultados no reproducibles. Fija seeds de Python, NumPy, PyTorch y CUDA.
3. **Ejes truncados engañosamente**: Iniciar eje Y en 0.95 en lugar de 0 exagera diferencias. Sé honesto.
4. **Colores no accesibles**: Usar solo rojo-verde excluye a personas con daltonismo. Usa paletas colorblind-friendly.
```

## Ejercicio Práctico: Detectar Overfitting Visualmente

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Simular datos de entrenamiento con overfitting
np.random.seed(42)
epochs = np.arange(1, 51)

# Escenario 1: Entrenamiento saludable
train_loss_good = 2.0 * np.exp(-0.08 * epochs) + np.random.normal(0, 0.03, 50)
val_loss_good = 2.0 * np.exp(-0.07 * epochs) + np.random.normal(0, 0.05, 50)

# Escenario 2: Overfitting claro (loss de validación sube después de epoch 20)
train_loss_overfit = 2.0 * np.exp(-0.12 * epochs) + np.random.normal(0, 0.02, 50)
val_loss_overfit = 2.0 * np.exp(-0.08 * epochs[:20]).tolist() + \
                   (0.8 + 0.02 * np.arange(30)).tolist()
val_loss_overfit = np.array(val_loss_overfit) + np.random.normal(0, 0.05, 50)
```

Un buen científico de datos sabe **leer curvas**. Generalmente, se grafican ambos indicadores `(Train / Validation)` para vigilar este punto de inflexión donde empezamos a sobre-entrenar (memorizar) en lugar de aprender.

```{code-cell} ipython3
# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica 1: Entrenamiento saludable
axes[0].plot(epochs, train_loss_good, label='Train Loss', color='#0077BB', linewidth=2)
axes[0].plot(epochs, val_loss_good, label='Validation Loss', color='#EE7733', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Entrenamiento Saludable')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 2.5)

# Gráfica 2: Overfitting
axes[1].plot(epochs, train_loss_overfit, label='Train Loss', color='#0077BB', linewidth=2)
axes[1].plot(epochs, val_loss_overfit, label='Validation Loss', color='#EE7733', linewidth=2)
axes[1].axvline(x=20, color='red', linestyle='--', label='Inicio de Overfitting', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Overfitting Claro')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 2.5)

plt.tight_layout()
plt.savefig('overfitting_detection.png', dpi=150, bbox_inches='tight')
plt.show()
```

Por supuesto, no podemos depender siempre del ojo humano para detectar estos estallidos. Técnicamente hablando, el patrón visual de la derecha puede ser modelado como un algoritmo de automatización ("Autocorrector" de Early Stopping) monitorizando qué tantas épocas consecutivas duran sin reducir el Loss.

```{code-cell} ipython3
# Análisis automático
def detectar_overfitting(train_loss, val_loss, patience=5):
    """
    Detecta overfitting comparando tendencias de loss.

    Args:
        train_loss: Array de training loss por época
        val_loss: Array de validation loss por época
        patience: Cuántas épocas consecutivas de incremento en val_loss

    Returns:
        Época donde comienza overfitting (o None)
    """
    min_val_loss = val_loss[0]
    min_epoch = 0
    contador = 0

    for epoch, loss in enumerate(val_loss):
        if loss < min_val_loss:
            min_val_loss = loss
            min_epoch = epoch
            contador = 0
        else:
            contador += 1
            if contador >= patience:
                return min_epoch + 1

    return None

# Aplicar detector
overfit_epoch_good = detectar_overfitting(train_loss_good, val_loss_good)
overfit_epoch_bad = detectar_overfitting(train_loss_overfit, val_loss_overfit)

print("Análisis de Overfitting:")
print(f"  Escenario 1 (saludable): {'No overfitting detectado' if overfit_epoch_good is None else f'Overfitting en época {overfit_epoch_good}'}")
print(f"  Escenario 2 (overfitting): {'No overfitting detectado' if overfit_epoch_bad is None else f'Overfitting en época {overfit_epoch_bad}'}")
```

## Ejercicio Práctico: Obtener Git Hash

```{code-cell} ipython3
import subprocess
import os

def get_git_hash():
    """
    Obtiene el commit hash actual de git.

    Returns:
        String con el hash corto (7 caracteres) o "unknown" si no es un repo git
    """
    try:
        # Ejecutar git rev-parse HEAD
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def get_git_info():
    """
    Obtiene información completa de git para reproducibilidad.

    Returns:
        Dict con hash, branch, y estado (dirty/clean)
    """
    info = {
        "hash": "unknown",
        "branch": "unknown",
        "dirty": True
    }

    try:
        # Hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        info["hash"] = result.stdout.strip()

        # Branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        info["branch"] = result.stdout.strip()

        # Estado (dirty = hay cambios sin commitear)
        result = subprocess.run(
            ['git', 'diff', '--quiet'],
            capture_output=True
        )
        info["dirty"] = (result.returncode != 0)

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return info
```

Una vez que tenemos los envoltorios definidos en base a subprocesos, simplemente los consultamos e interrogamos el entorno local:

```{code-cell} ipython3
# Ejemplo de uso interrogativo
print("Git Hash corto:")
print(f"  {get_git_hash()}")

print("\nInformación completa de Git:")
git_info = get_git_info()
for key, value in git_info.items():
    print(f"  {key}: {value}")
```

La forma habitual en que inyectamos esto en nuestro script diario de modelaje es incluir esta metadata en nuestro YAML de Hydra (u objeto de experimento como Weights+Biases), documentando todo de manera transparente.

```{code-cell} ipython3
# Ejemplo: Empaquetar estado para incluír en la base de datos experimental
experiment_config = {
    "model": "codellama-7b",
    "learning_rate": 1e-4,
    "git_commit": get_git_hash(),
    "git_info": get_git_info(),
    "timestamp": "2024-03-10T14:30:00",
}

print("\nConfig de experimento con tracking de git:")
import json
print(json.dumps(experiment_config, indent=2))

# Advertencia si hay cambios sin commitear
if git_info["dirty"]:
    print("\n⚠️ ADVERTENCIA: Tienes cambios sin commitear. Haz commit antes de correr experimentos.")
```

```{admonition} ✅ Verifica tu comprensión
:class: note
1. ¿Qué ventajas tiene usar wandb sobre simplemente guardar métricas en un archivo JSON?
2. Explica por qué es crítico fijar seeds (numpy, torch, python) para reproducibilidad.
3. ¿Cómo detectarías overfitting mirando una gráfica de training vs validation loss?
4. Diseña un checklist MLOps para tu proyecto. ¿Qué 5 elementos son más críticos?
```

## Resumen

```{admonition} Resumen
:class: important
**Conceptos clave:**
- MLOps tracking (wandb, MLflow) registra código, datos, config, métricas y artefactos sistemáticamente
- Versionado con YAML + Hydra permite reproducir experimentos exactos meses después
- Visualizaciones efectivas siguen principios: claridad > decoración, comparación justa, honestidad, accesibilidad
- Checklists MLOps garantizan: git commit trackeado, seeds fijadas, config guardada, checkpoints periódicos
- Detectar overfitting: training loss ↓ mientras validation loss ↑ indica memorización sin generalización

**Preparación final:** Has completado el módulo de AI/ML. Ahora dominas desde fundamentos (deep learning, transformers) hasta aplicaciones prácticas (fine-tuning, constrained decoding, MLOps). Estás listo para aplicar estos conceptos en proyectos reales de generación de código GPU.
```

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*

---

## Referencias

- Weights & Biases. [Documentation](https://docs.wandb.ai/). W&B.
- MLflow. [Documentation](https://mlflow.org/docs/latest/). MLflow.
- Zaharia, M. et al. (2018). [Accelerating the Machine Learning Lifecycle with MLflow](https://www.databricks.com/research/mlflow-paper). IEEE Data Engineering Bulletin.
- Tufte, E. (2001). The Visual Display of Quantitative Information (2nd ed.). Graphics Press.
- Wilkinson, L. (2005). The Grammar of Graphics (2nd ed.). Springer.
