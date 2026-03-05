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

# Lectura 8: MLOps Básico y Visualización

```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/08_mlops_visualizacion.ipynb)
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Implementar tracking sistemático de experimentos usando Weights & Biases o MLflow
- Versionar configuraciones de experimentos con archivos YAML y Hydra
- Crear visualizaciones efectivas y accesibles siguiendo principios de claridad y honestidad
- Aplicar checklists de MLOps para garantizar reproducibilidad (git commit, seeds, config)
- Detectar overfitting visualmente mediante análisis de curvas de entrenamiento y validación
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[MLOps in 10 Minutes (DataTalksClub)](https://www.youtube.com/watch?v=7kZrrCsJdEM)** - Diagramas de arquitectura del ciclo de vida de un modelo de IA en producción.
```


## Introducción

Has entrenado modelos, generado código, evaluado resultados. Pero, ¿cómo **rastrear** todo? ¿Cómo saber qué hiperparámetros usaste hace 3 semanas? ¿Cómo comparar 50 experimentos?

MLOps (Machine Learning Operations) resuelve esto: logging sistemático, versionado de experimentos, y visualización efectiva. Esta lectura te da las herramientas para investigación reproducible.

---

## Parte 1: El Problema del Tracking

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

## Parte 2: Weights & Biases (wandb)

### Setup Básico

```{code-cell} ipython3
:tags: [skip-execution]

import wandb

# Inicializar proyecto
wandb.init(
    project="kernel-generation",
    name="experiment-001",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "codellama-7b",
        "temperature": 0.1,
    }
)
```

### Logging de Métricas

```python
# Durante entrenamiento
for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = validate()

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": scheduler.get_lr()[0]
    })

# Métricas finales
wandb.log({
    "final_accuracy": 0.85,
    "total_time_hours": 2.5
})
```

### Logging de Configuración Completa

```python
config = {
    # Modelo
    "model_name": "codellama-7b",
    "quantization": "int8",

    # Entrenamiento
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "adamw",
    "weight_decay": 0.01,

    # Datos
    "dataset": "triton-corpus-v2",
    "train_size": 5000,
    "val_size": 500,

    # Generación
    "temperature": 0.1,
    "max_tokens": 512,
    "grammar_enabled": True,

    # Reproducibilidad
    "seed": 42,
    "git_commit": get_git_hash(),
}

wandb.init(project="kernel-gen", config=config)
```

### Tablas y Artefactos

```python
# Tabla de resultados
results_table = wandb.Table(
    columns=["kernel_id", "correctness", "speedup", "tokens"],
    data=[
        ["softmax_v1", True, 1.3, 245],
        ["matmul_v1", True, 1.1, 512],
        ["relu_v1", False, 0.0, 128],
    ]
)
wandb.log({"results": results_table})

# Guardar artefactos
artifact = wandb.Artifact("model-checkpoint", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)
```

### Comparación de Experimentos

```python
# En la UI de wandb:
# 1. Selecciona múltiples runs
# 2. Compara métricas lado a lado
# 3. Identifica qué configuración funciona mejor

# Programáticamente:
api = wandb.Api()
runs = api.runs("username/kernel-generation")

for run in runs:
    print(f"{run.name}: accuracy={run.summary['accuracy']}")
```

---

## Parte 3: MLflow (Alternativa)

### Setup Básico

```{code-cell} ipython3
:tags: [skip-execution]

import mlflow

# Configurar tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("kernel-generation")

# Iniciar run
with mlflow.start_run(run_name="experiment-001"):
    # Log parámetros
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("model", "codellama-7b")

    # Log métricas
    for epoch in range(10):
        mlflow.log_metric("loss", loss, step=epoch)

    # Log modelo
    mlflow.pytorch.log_model(model, "model")
```

### Comparación wandb vs MLflow

| Aspecto | wandb | MLflow |
|---------|-------|--------|
| Hosting | Cloud (gratis hasta cierto punto) | Self-hosted o cloud |
| UI | Muy pulida | Funcional |
| Colaboración | Excelente | Básica |
| Integración | Muchos frameworks | Flexible |
| Costo | Gratis para académicos | Open source |

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
