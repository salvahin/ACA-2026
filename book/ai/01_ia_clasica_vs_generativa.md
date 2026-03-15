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

# Lectura 1: Introducción al Aprendizaje Automático

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/01_ia_clasica_vs_generativa.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment (todas las dependencias ya vienen en Colab)
print('Entorno listo!')
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Comprender la evolución histórica de la IA desde el taller de Dartmouth (1956) hasta el deep learning moderno
- Distinguir entre los paradigmas de aprendizaje: supervisado, no supervisado y por refuerzo
- Identificar los dos tipos clásicos de problemas: clasificación y regresión
- Aplicar conceptos básicos de regresión lineal y logística
- Entender el problema XOR como motivación para redes neuronales multicapa
- Distinguir entre modelos discriminativos (P(Y|X)) y generativos (P(X,Y))
```

```{admonition} 🎬 Video Recomendado
:class: tip

**[Generative AI in a Nutshell (IBM Technology)](https://www.youtube.com/watch?v=G2fqAlgmoPo)** - Explicación visual sobre la diferencia entre paradigmas de IA clásica y generativa.
```

## Contexto
En esta lectura comprenderás la evolución de la IA desde sistemas basados en reglas hasta modelos generativos modernos. Aprenderás las diferencias entre paradigmas discriminativos y generativos, y cuándo aplicar cada enfoque.

## Introducción

Imagina que quieres construir un sistema inteligente. Hace 50 años, los investigadores tenían una idea clara: *codifiquemos las reglas del sentido común directamente en la máquina*. Si querías un sistema que diagnosticara enfermedades, le escribías reglas: "Si el paciente tiene fiebre Y tos Y radiografía anormal, entonces probablemente tiene neumonía."

Ese enfoque funcionaba para problemas simples, pero se topaba con una pared: ¿qué pasa cuando las reglas son demasiado complejas para escribir explícitamente? ¿Cómo codificas el patrón visual que distingue un gato de un perro?

En esta lectura exploraremos cómo la IA ha evolucionado desde ese primer enfoque de **reglas explicitas** hasta los **modelos generativos** modernos como ChatGPT. Veremos que cada enfoque responde a la pregunta: "¿Cómo podemos hacer que las máquinas aprendan?"

---

## Parte 1: El Viaje del Aprendizaje Automático

### Los Orígenes: El Taller de Dartmouth (1956)

La historia de la Inteligencia Artificial como disciplina formal comenzó en el verano de 1956, cuando John McCarthy, Marvin Minsky, Claude Shannon y Nathaniel Rochester organizaron el **Taller de Dartmouth**. Este evento histórico definió el campo y estableció la visión fundamental:

```
"Cada aspecto del aprendizaje o cualquier otra característica de la inteligencia
puede, en principio, ser descrito con tanta precisión que una máquina puede
simularlo."
— Propuesta del Taller de Dartmouth, 1956
```

Del taller emergieron ideas fundamentales que aún guían el campo:
- La IA tendría **dos fases**: entrenamiento (aprender de datos) e inferencia (aplicar lo aprendido)
- Las máquinas podrían aprender patrones sin ser programadas explícitamente
- El lenguaje, la visión y el razonamiento podrían ser computacionales

```{admonition} 💡 ¿Qué es la Inteligencia?
:class: note
Desde una perspectiva de ML, la **inteligencia** puede verse como la capacidad de:
- **Reconocer patrones** en datos (clasificar imágenes, detectar spam)
- **Generalizar** a partir de ejemplos (predecir en datos nuevos)
- **Aprender distribuciones** estadísticas subyacentes en los datos

Los modelos de ML no "piensan" — aprenden aproximaciones estadísticas de patrones en datos.
```

### Era 1: Sistemas Expertos (1960s-1980s)

:::{figure} images/AI_01_01_IA_Era_Timeline_Systems_to_LLMs.jpeg
:name: fig-ai-eras
:alt: Timeline de las cuatro eras de la IA
:align: center
:width: 90%

**Figura 1:** Las Cuatro Eras de la Inteligencia Artificial - desde sistemas expertos hasta modelos de lenguaje grande.
:::

Los **sistemas expertos** codificaban el conocimiento de humanos expertos como reglas lógicas:

```{code-cell} ipython3
# Ejemplo conceptual de un sistema experto
class Paciente:
    def __init__(self, fiebre, tos, radiografia, dolor_garganta):
        self.fiebre = fiebre
        self.tos = tos
        self.radiografia = radiografia
        self.dolor_garganta = dolor_garganta

def sistema_experto_diagnostico(paciente):
    if paciente.fiebre and paciente.tos and paciente.radiografia == "anormal":
        return "neumonía"
    elif paciente.fiebre and paciente.dolor_garganta:
        return "faringitis"
    else:
        return "sin diagnóstico claro"

# Ejemplo de uso
paciente1 = Paciente(fiebre=True, tos=True, radiografia="anormal", dolor_garganta=False)
paciente2 = Paciente(fiebre=True, tos=False, radiografia="normal", dolor_garganta=True)

print(f"Paciente 1: {sistema_experto_diagnostico(paciente1)}")
print(f"Paciente 2: {sistema_experto_diagnostico(paciente2)}")
```

**Ventajas:**
- Transparentes y fáciles de auditar
- Funcionan bien para dominios con reglas claras

**Problemas:**
- Requieren que un experto escriba cada regla
- No se adaptan a nuevas situaciones
- El número de reglas crece exponencialmente

```{admonition} 🤔 Reflexiona
:class: hint
¿Por qué crees que los sistemas expertos fracasaron en problemas complejos como reconocimiento de imágenes? ¿Podrías escribir reglas explícitas para distinguir un gato de un perro?
```

### Era 2: Aprendizaje Automático Clásico (1990s-2010s)

El gran salto: **en lugar de escribir reglas, enseña al sistema a descubrirlas a partir de datos.**

Imagina que tienes 1,000 fotos de gatos y 1,000 de perros. Un algoritmo de aprendizaje automático examina automáticamente estas imágenes y descubre patrones: "Los gatos tienen orejas triangulares, los perros tienen orejas más redondeadas," etc.

```{admonition} 🎯 Bases Estadísticas del ML
:class: important
El aprendizaje automático es fundamentalmente **estadístico**:
- El objetivo es **aprender una distribución de probabilidad** a partir de datos
- Un modelo aprende a estimar P(Y|X): "dado X, ¿cuál es la probabilidad de Y?"
- La calidad del modelo se mide por qué tan bien generaliza a datos no vistos
```

### Los Dos Tipos Clásicos de Problemas

Antes de ver los paradigmas de aprendizaje, es importante distinguir los dos tipos fundamentales de problemas:

#### Clasificación
**Objetivo:** Asignar una etiqueta categórica a cada entrada.

```
Entrada: Imagen de animal
Salida: "gato" o "perro" (categoría discreta)

Entrada: Email
Salida: "spam" o "no spam"
```

#### Regresión
**Objetivo:** Predecir un valor numérico continuo.

```
Entrada: Características de una casa (m², ubicación, habitaciones)
Salida: Precio en pesos (valor continuo)

Entrada: Datos históricos de ventas
Salida: Ventas del próximo mes
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Visualización: Clasificación vs Regresión
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Clasificación
np.random.seed(42)
class_0 = np.random.randn(50, 2) + np.array([2, 2])
class_1 = np.random.randn(50, 2) + np.array([5, 5])

axes[0].scatter(class_0[:, 0], class_0[:, 1], c='blue', alpha=0.6, label='Clase 0 (Gatos)', s=60)
axes[0].scatter(class_1[:, 0], class_1[:, 1], c='red', alpha=0.6, label='Clase 1 (Perros)', s=60)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Clasificación: Asignar Categorías', fontsize=12, weight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Regresión
x = np.linspace(0, 10, 50)
y = 2 * x + 3 + np.random.randn(50) * 2

axes[1].scatter(x, y, c='green', alpha=0.6, s=60, label='Datos observados')
axes[1].plot(x, 2*x + 3, 'r--', linewidth=2, label='Línea de regresión')
axes[1].set_xlabel('Tamaño (m²)')
axes[1].set_ylabel('Precio ($)')
axes[1].set_title('Regresión: Predecir Valores Continuos', fontsize=12, weight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Clasificación: Salida discreta (categorías)")
print("Regresión: Salida continua (números)")
```

### Regresión Lineal

El modelo más simple de ML: una línea recta que mejor se ajusta a los datos.

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

donde:
- y es la predicción
- x₁, x₂, ..., xₙ son las características de entrada
- w₁, w₂, ..., wₙ son los pesos (coeficientes)
- b es el sesgo (intercepto)
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de ejemplo: área de casa vs precio
np.random.seed(42)
area = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140]).reshape(-1, 1)
precio = area.flatten() * 15000 + 100000 + np.random.randn(10) * 50000

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(area, precio)

# Predicción
area_nueva = np.array([[85], [125]])
predicciones = modelo.predict(area_nueva)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(area, precio, c='blue', s=100, alpha=0.7, label='Datos reales', edgecolors='black')
plt.plot(area, modelo.predict(area), 'r-', linewidth=2, label='Regresión lineal')
plt.scatter(area_nueva, predicciones, c='green', s=150, marker='*', label='Predicciones', edgecolors='black')

plt.xlabel('Área (m²)', fontsize=12)
plt.ylabel('Precio ($)', fontsize=12)
plt.title('Regresión Lineal: Predecir Precio de Casas', fontsize=14, weight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Ecuación aprendida: Precio = {modelo.coef_[0]:.2f} × Área + {modelo.intercept_:.2f}")
print(f"\nPredicciones:")
for a, p in zip(area_nueva.flatten(), predicciones):
    print(f"  Casa de {a} m² → ${p:,.0f}")
```

### Regresión Logística

Para **clasificación binaria**, usamos regresión logística. A pesar del nombre, es un modelo de clasificación.

```
P(y=1|x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))

La función sigmoide (σ) convierte cualquier valor en una probabilidad entre 0 y 1.
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generar datos de clasificación binaria
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Entrenar modelo
modelo = LogisticRegression()
modelo.fit(X, y)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Datos y frontera de decisión
ax = axes[0]
ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.6, s=60, label='Clase 0')
ax.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.6, s=60, label='Clase 1')

# Dibujar frontera de decisión
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
Z = modelo.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Regresión Logística: Frontera de Decisión', fontsize=12, weight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Función sigmoide
ax = axes[1]
z = np.linspace(-6, 6, 100)
sigmoid = 1 / (1 + np.exp(-z))
ax.plot(z, sigmoid, 'b-', linewidth=3)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Umbral (0.5)')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(z, 0, sigmoid, where=(sigmoid >= 0.5), alpha=0.3, color='red', label='Clase 1')
ax.fill_between(z, 0, sigmoid, where=(sigmoid < 0.5), alpha=0.3, color='blue', label='Clase 0')
ax.set_xlabel('z = w·x + b')
ax.set_ylabel('P(y=1|x)')
ax.set_title('Función Sigmoide: Convierte Score a Probabilidad', fontsize=12, weight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Regresión Logística:")
print("  - Entrada: características numéricas")
print("  - Salida: probabilidad P(clase=1)")
print("  - Decisión: si P > 0.5, predice clase 1; sino, clase 0")
```

### Los Paradigmas de Aprendizaje

Los principales paradigmas son:

#### 2.1 Aprendizaje Supervisado

**La idea:** tienes datos con etiquetas. Quieres que la máquina aprenda a etiquetar nuevos datos.

```
Entrada: Fotos de animales
Salida Deseada: [gato, perro, gato, gato, perro, ...]

El sistema aprende: ¿qué características predicen cada etiqueta?
```

**Algoritmos comunes:** Regresión logística, máquinas de soporte vectorial (SVM), árboles de decisión.

#### 2.2 Aprendizaje No Supervisado

**La idea:** tienes datos sin etiquetas. El sistema debe encontrar patrones o estructura internamente.

```
Entrada: Clientes de una tienda (sin información de grupo)
Salida: Clusters de clientes similares
```

**Algoritmo común:** K-medias, clustering jerárquico.

#### 2.3 Aprendizaje por Refuerzo

**La idea:** un agente aprende mediante prueba y error, recibiendo recompensas por buenas acciones.

```
Agente intenta jugar ajedrez → pierde → obtiene recompensa negativa
Agente intenta otro movimiento → gana → obtiene recompensa positiva
```

Después de miles de intentos, el sistema aprende estrategias ganadoras.

### Era 3: Aprendizaje Profundo (2010s)

Las redes neuronales profundas revolucionaron todo. Esta era se caracteriza por:

1. **Redes Neuronales Multicapa** (solucionan el problema XOR)
2. **Backpropagation** para entrenar redes profundas
3. **Gradiente Descendente Estocástico (SGD)** y optimizadores modernos
4. **Arquitecturas especializadas**: CNNs, RNNs, LSTMs, y finalmente Transformers

```{admonition} 🔧 Gradiente Descendente y Optimizadores
:class: note
El **gradiente descendente** es el algoritmo fundamental para entrenar redes neuronales:

1. Calcula el error (loss) de la predicción
2. Calcula el gradiente (dirección de mayor aumento del error)
3. Actualiza los pesos en la dirección **opuesta** al gradiente

```
W_nuevo = W_viejo - learning_rate × gradiente
```

**SGD (Stochastic Gradient Descent):** Actualiza pesos usando mini-batches en lugar de todo el dataset.

**Optimizadores modernos (Adam, AdaW):** Adaptan el learning rate por parámetro. Se explicarán en detalle en la **Lectura 2**.
```

Antes de hablar de arquitecturas, necesitamos entender **modelos discriminativos vs generativos**.

---

## Parte 2: Discriminativos vs Generativos

:::{figure} diagrams/discriminative_vs_generative.png
:name: fig-discriminative-vs-generative
:alt: Comparación entre modelos discriminativos y generativos en machine learning
:align: center
:width: 90%

**Figura 2:** Modelos discriminativos (P(Y|X)) vs generativos (P(X,Y)): los discriminativos aprenden la frontera de decisión; los generativos aprenden la distribución completa de los datos y pueden generar nuevas muestras.
:::

Esta es una distinción fundamental que aún hoy define cómo pensamos sobre IA.

### Modelos Discriminativos

**Pregunta que responden:** "¿A qué categoría pertenece esto?"

```
Entrada: Foto de animal
Modelo discriminativo: "Esto es un gato" (probabilidad: 87%)

¿Cómo funciona? Dibuja una línea (o hipersuperficie) entre gatos y perros.
```

**Ejemplos:**
- Clasificación de imágenes: "¿es un gato o un perro?"
- Detección de spam: "¿es este email spam o no?"
- Diagnóstico médico: "¿tiene esta radiografía un tumor?"

**Ventaja clave:** Son muy buenos en lo que hacen. Si solo necesitas clasificar, son eficientes.

### Modelos Generativos

**Pregunta que responden:** "¿Cuál es la distribución subyacente de los datos? ¿Puedo generar nuevos ejemplos?"

```
Entrada: La palabra "gato"
Modelo generativo: Genera una descripción realista de un gato

O:

Entrada: Primeras palabras de un texto
Modelo generativo: Completa la oración de forma coherente
```

**Ejemplos:**
- Traducción automática
- Resumen de textos
- Generación de imágenes (DALL-E, Stable Diffusion)
- Chatbots (GPT)

**Ventaja clave:** Entienden la distribución completa de los datos, no solo cómo clasificar.

### El Factor P(X|Y) vs P(X,Y)

Matemáticamente:
- **Discriminativo** modela P(Y|X): "¿cuál es la probabilidad de la etiqueta Y dado el dato X?"
- **Generativo** modela P(X,Y): "¿cuál es la probabilidad conjunta de X e Y?" De aquí puedes derivar P(X|Y) si lo necesitas, pero también P(X): "¿qué datos son probables?"

:::{figure} images/AI_01_02_Modelos_Discriminativo_vs_Generativo.jpeg
:name: fig-disc-gen
:alt: Comparación visual entre modelos discriminativos y generativos
:align: center
:width: 90%

**Figura 3:** Modelos Discriminativos vs Generativos - el discriminativo traza límites de decisión mientras el generativo modela la distribución completa.
:::

```{admonition} 📚 Conexión
:class: seealso
Esta distinción entre modelos discriminativos y generativos será fundamental cuando estudiemos arquitecturas Transformer en la Lectura 4. BERT (encoder) es discriminativo mientras GPT (decoder) es generativo.
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Visualización: Discriminativo vs Generativo
np.random.seed(42)

# Generar datos para dos clases
class_0 = np.random.randn(100, 2) + np.array([2, 2])
class_1 = np.random.randn(100, 2) + np.array([5, 5])

# Crear figura
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Modelo Discriminativo: solo dibuja la frontera de decisión
axes[0].scatter(class_0[:, 0], class_0[:, 1], c='blue', alpha=0.6, label='Clase 0')
axes[0].scatter(class_1[:, 0], class_1[:, 1], c='red', alpha=0.6, label='Clase 1')
x_line = np.linspace(0, 8, 100)
y_line = x_line  # Frontera simple para ilustrar
axes[0].plot(x_line, y_line, 'k--', linewidth=2, label='Frontera de decisión')
axes[0].set_title('Modelo Discriminativo\nP(Y|X): Clasifica nuevos puntos', fontsize=12)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Modelo Generativo: modela la distribución de cada clase
from matplotlib.patches import Ellipse

axes[1].scatter(class_0[:, 0], class_0[:, 1], c='blue', alpha=0.6, label='Clase 0')
axes[1].scatter(class_1[:, 0], class_1[:, 1], c='red', alpha=0.6, label='Clase 1')

# Agregar elipses para mostrar que modela la distribución
ellipse_0 = Ellipse(xy=(2, 2), width=4, height=4, angle=0,
                    edgecolor='blue', fc='None', lw=2, linestyle='--')
ellipse_1 = Ellipse(xy=(5, 5), width=4, height=4, angle=0,
                    edgecolor='red', fc='None', lw=2, linestyle='--')
axes[1].add_patch(ellipse_0)
axes[1].add_patch(ellipse_1)

axes[1].set_title('Modelo Generativo\nP(X,Y): Modela distribuciones completas', fontsize=12)
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Discriminativo: Aprende la frontera de decisión directamente")
print("Generativo: Aprende cómo se distribuyen los datos de cada clase")
```

---

## Parte 3: De Redes Neuronales a LLMs

### El Perceptrón (1950s)

```
Entrada → [suma ponderada] → [función escalón] → Salida (0 o 1)

x1 →|
    |--[w1*x1 + w2*x2 + b > 0?] → Salida
x2 →|--[w2]
      [b]
```

Podía resolver problemas lineales simples. Su limitación: no podía resolver el problema XOR.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Implementación simple de un perceptrón
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0

                # Actualizar pesos
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
```

A continuación, vamos a evaluar cómo nuestro brillante (pero limitado) perceptrón intenta resolver dos problemas matemáticos elementales clásicos: **La compuerta AND** (Linealmente separable) y **La compuerta XOR** (No Lineal).

```{code-cell} ipython3
# Demostración en problemas analíticos lógicos
# 1. Compuerta AND (Debería funcionar)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron()
perceptron.fit(X_and, y_and)

# 2. Compuerta XOR (Debería fallar catastróficamente por limitación matemática)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
```

Para demostrar lo que sucede topológicamente, vamos a trazar la frontera geométrica de decisión sobre un plano cartesiano intentando segmentar ambos conjuntos.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AND (funciona)
axes[0].scatter(X_and[y_and==0][:, 0], X_and[y_and==0][:, 1],
                c='blue', s=100, label='Clase 0', edgecolors='k')
axes[0].scatter(X_and[y_and==1][:, 0], X_and[y_and==1][:, 1],
                c='red', s=100, label='Clase 1', edgecolors='k')

# Dibujar frontera de decisión
x_line = np.linspace(-0.5, 1.5, 100)
if perceptron.weights[1] != 0:
    y_line = -(perceptron.weights[0] * x_line + perceptron.bias) / perceptron.weights[1]
    axes[0].plot(x_line, y_line, 'k--', linewidth=2, label='Frontera')

axes[0].set_xlim(-0.5, 1.5)
axes[0].set_ylim(-0.5, 1.5)
axes[0].set_title('Perceptrón: AND (Funciona)\nLinealmente Separable', fontsize=12)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dibujando el problema XOR (no funciona puesto que no es linealmente separable)
axes[1].scatter(X_xor[y_xor==0][:, 0], X_xor[y_xor==0][:, 1],
                c='blue', s=100, label='Clase 0', edgecolors='k')
axes[1].scatter(X_xor[y_xor==1][:, 0], X_xor[y_xor==1][:, 1],
                c='red', s=100, label='Clase 1', edgecolors='k')
axes[1].text(0.5, -0.3, 'No hay línea recta que separe las clases',
             ha='center', fontsize=10, style='italic')
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylim(-0.5, 1.5)
axes[1].set_title('Perceptrón: XOR (Falla)\nNo Linealmente Separable', fontsize=12)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"AND: Predicciones = {perceptron.predict(X_and)}, Esperado = {y_and}")
print("\nXOR requiere redes multicapa con funciones de activación no lineales")
```

### Arquitecturas de Redes Neuronales

```{admonition} 📌 Alcance de esta Lección
:class: note
Esta sección introduce arquitecturas de redes neuronales **de forma conceptual e histórica** para completar
el panorama evolutivo de la IA. Los fundamentos matemáticos detallados de backpropagation,
funciones de activación (ReLU, sigmoid, tanh), tensores y optimizadores se cubren
en profundidad en la **Lectura 2: Fundamentos de Deep Learning**.
```

#### 1. Redes Fully Connected (Densas)

La arquitectura más básica: cada neurona está conectada con todas las neuronas de la capa siguiente.

```
Capa de Entrada → Capas Ocultas → Capa de Salida
     (n)              (m)              (k)

Cada conexión tiene un peso que se aprende durante el entrenamiento.
```

Múltiples capas de "neuronas" conectadas. Cada conexión tiene un peso. Al sumar múltiples transformaciones no-lineales, podemos aproximar cualquier función.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Implementación simple de red neuronal multicapa
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        # Inicializar pesos y sesgos
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:  # ReLU para capas ocultas
                a = np.maximum(0, z)
            else:  # Sigmoid para capa de salida
                a = self.sigmoid(z)
            activations.append(a)
        return activations

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X)

            # Calcular pérdida
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)

            # Backward pass (simplificado)
            error = activations[-1] - y
            deltas = [error]

            for i in range(len(self.weights) - 1, 0, -1):
                delta = deltas[-1] @ self.weights[i].T
                if i > 0:  # ReLU derivative
                    delta = delta * (activations[i] > 0)
                deltas.append(delta)

            deltas.reverse()

            # Actualizar pesos
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * (activations[i].T @ deltas[i])
                self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True)

        return losses
```

Teniendo la red estructural base definida ahora creamos un "Dataset No Lineal" con topología de Lunas cruzadas, el modelo será entrenado durante mil épocas iterativas con una tasa de aprendizaje alta.

```{code-cell} ipython3
from sklearn.datasets import make_moons

# Generar datos no lineales bidimensionales (topología XOR-like)
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
y = y.reshape(-1, 1)

# Instanciar y Entrenar red neuronal multicapa profunda [Input, Hidden_1, Hidden_2, Output]
nn = NeuralNetwork([2, 8, 4, 1], learning_rate=0.3)
losses = nn.train(X, y, epochs=1000)
```

Por último verifiquemos computacionalmente dibujando la "curva de aprendizaje" a lo largo de las mil épocas y también trazando topológicamente la nueva y sinuosa barrera que fue capaz de inferir nuestro modelo sobre el espacio bi-dimensional.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Interfaz Visual Múltiple
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Datos y frontera de decisión
ax = axes[0]
h = 0.01
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])[-1]
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap='RdYlBu')
ax.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1],
           c='blue', s=50, edgecolors='k', label='Clase 0')
ax.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1],
           c='red', s=50, edgecolors='k', label='Clase 1')
ax.set_title('Red Neuronal Multicapa\nResuelve XOR (No Lineal)', fontsize=12, weight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Arquitectura
ax = axes[1]
ax.axis('off')

layer_positions = [0.1, 0.35, 0.65, 0.9]
layer_labels = ['Input\n(2)', 'Hidden 1\n(8)', 'Hidden 2\n(4)', 'Output\n(1)']
neuron_counts = [2, 8, 4, 1]

for i, (x_pos, label, n_neurons) in enumerate(zip(layer_positions, layer_labels, neuron_counts)):
    y_positions = np.linspace(0.2, 0.8, n_neurons)
    for y_pos in y_positions:
        color = 'lightblue' if i == 0 else 'lightgreen' if i < 3 else 'lightcoral'
        circle = plt.Circle((x_pos, y_pos), 0.03, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)

    ax.text(x_pos, 0.05, label, ha='center', fontsize=10, weight='bold')

    # Dibujar conexiones
    if i < len(layer_positions) - 1:
        for y1 in y_positions:
            next_y_positions = np.linspace(0.2, 0.8, neuron_counts[i+1])
            for y2 in next_y_positions:
                ax.plot([x_pos + 0.03, layer_positions[i+1] - 0.03],
                       [y1, y2], 'gray', alpha=0.2, linewidth=0.5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Arquitectura de la Red', fontsize=12, weight='bold')

# 3. Curva de aprendizaje
axes[2].plot(losses, linewidth=2, color='blue')
axes[2].set_xlabel('Época', fontsize=11)
axes[2].set_ylabel('Pérdida (MSE)', fontsize=11)
axes[2].set_title('Curva de Aprendizaje', fontsize=12, weight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_yscale('log')

plt.tight_layout()
plt.show()

# Predicciones
predictions = nn.forward(X)[-1]
accuracy = np.mean((predictions > 0.5) == y)
print(f"Precisión en datos de entrenamiento: {accuracy*100:.2f}%")
print(f"\nLa red multicapa resuelve problemas NO LINEALES que un perceptrón simple no puede.")
print(f"Arquitectura: {nn.layer_sizes}")
print(f"Total de parámetros: {sum(w.size for w in nn.weights) + sum(b.size for b in nn.biases)}")
```

**Clave:** La función de activación (ReLU, tanh, sigmoid) introduce no-linealidad.

### El Algoritmo de Retropropagación (Backpropagation)

La retropropagación es cómo entrenamos redes neuronales profundas. Conceptualmente:

1. **Forward pass:** alimenta datos a través de la red
2. **Calcula error:** ¿cuánto se equivocó la predicción?
3. **Backward pass:** propaga ese error hacia atrás, calculando cuánto debe cambiar cada peso
4. **Actualización:** ajusta los pesos en la dirección que reduce el error

Es como subir una montaña ciega: sientes hacia dónde está más empinado (gradiente) y das un paso en esa dirección.

#### 2. Redes Convolucionales (CNNs)

Especializadas para procesar imágenes. Usan **filtros** (kernels) que se deslizan sobre la imagen para detectar patrones locales.

```
Imagen → [Filtros Convolucionales] → [Pooling] → [Fully Connected] → Predicción

Ejemplo (clasificación de dígitos):
  28×28 imagen → Conv1 (32 filtros) → Pool → Conv2 (64 filtros) → Pool → Dense → "5"
```

**Ventajas:**
- Invarianza a traslación (detecta gatos en cualquier posición)
- Eficientes en parámetros (comparten pesos)
- Revolucionaron la visión por computadora (AlexNet, 2012)

#### 3. Series de Tiempo y Modelos Secuenciales

Para datos donde el **orden importa** (texto, audio, series financieras), necesitamos modelos que procesen secuencias.

**Métodos clásicos de autoregresión:**
- AR (Autoregressive): predice basándose en valores pasados
- ARIMA: agrega diferenciación y promedios móviles
- Limitación: asumen relaciones lineales

### RNNs y la Secuencia

Las redes neuronales recurrentes (RNNs) fueron diseñadas para procesar secuencias:

```
Entrada: "Quiero aprender IA"
RNN: Procesa palabra por palabra, manteniendo un "estado" de memoria

Palabra 1: "Quiero" → Estado1
Palabra 2: "aprender" + Estado1 → Estado2
Palabra 3: "IA" + Estado2 → Estado3 (contiene contexto de todas las palabras anteriores)
```

**Problema:** Las RNNs sufren de desvanecimiento de gradientes (el error se diluye en secuencias largas).

### LSTMs y GRUs

Arquitecturas mejoradas que mantienen "vías" especiales para que la información importante no desaparezca a través de secuencias largas.

### Autoencoders: Aprendiendo Representaciones

Antes de los Transformers, los **autoencoders** introdujeron una idea crucial: aprender representaciones comprimidas de los datos.

```
Entrada → [Encoder] → Representación Latente (cuello de botella) → [Decoder] → Reconstrucción
                              ↑
                    Vector de dimensión reducida
                    que captura la "esencia" de los datos
```

**Estructura de "reloj de arena":** La red se comprime en el medio, forzando al modelo a aprender qué información es esencial.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Visualización conceptual de un autoencoder
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Capas del autoencoder
layer_sizes = [8, 6, 4, 2, 4, 6, 8]  # Forma de reloj de arena
layer_names = ['Input', 'Enc 1', 'Enc 2', 'Latent', 'Dec 1', 'Dec 2', 'Output']
colors = ['lightblue', 'lightgreen', 'lightgreen', 'gold', 'lightcoral', 'lightcoral', 'lightblue']

x_positions = np.linspace(0.1, 0.9, len(layer_sizes))

for i, (x, size, name, color) in enumerate(zip(x_positions, layer_sizes, layer_names, colors)):
    y_positions = np.linspace(0.5 - size*0.05, 0.5 + size*0.05, size)
    for y in y_positions:
        circle = plt.Circle((x, y), 0.02, color=color, ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
    ax.text(x, 0.1, name, ha='center', fontsize=10, weight='bold')

    # Conexiones
    if i < len(layer_sizes) - 1:
        next_y = np.linspace(0.5 - layer_sizes[i+1]*0.05, 0.5 + layer_sizes[i+1]*0.05, layer_sizes[i+1])
        for y1 in y_positions[::2]:
            for y2 in next_y[::2]:
                ax.plot([x + 0.02, x_positions[i+1] - 0.02], [y1, y2], 'gray', alpha=0.2, linewidth=0.5)

# Anotaciones
ax.annotate('Encoder\n(compresión)', xy=(0.25, 0.85), fontsize=11, ha='center', color='green', weight='bold')
ax.annotate('Decoder\n(reconstrucción)', xy=(0.75, 0.85), fontsize=11, ha='center', color='red', weight='bold')
ax.annotate('Cuello de\nbotella', xy=(0.5, 0.15), fontsize=10, ha='center', color='goldenrod', weight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Arquitectura de Autoencoder: Forma de Reloj de Arena', fontsize=14, weight='bold')
plt.show()

print("Autoencoder:")
print("  - Input: datos originales (ej: imagen 784 dims)")
print("  - Latent: representación comprimida (ej: 32 dims)")
print("  - Output: reconstrucción del input original")
print("  - Objetivo: minimizar diferencia entre input y output")
```

### Autoencoders Variacionales (VAEs)

Los **VAEs** extienden los autoencoders para permitir **generación** de nuevos datos:

```
Diferencia clave:
- Autoencoder: latente es un vector fijo
- VAE: latente es una distribución (media + varianza)

Esto permite muestrear nuevos puntos del espacio latente y generar datos nuevos.
```

```{admonition} 📚 Conexión con Transformers
:class: seealso
Los conceptos de encoder-decoder de los autoencoders se trasladan directamente a la arquitectura Transformer:
- **Encoder**: comprime la entrada en representaciones (como BERT)
- **Decoder**: genera salidas a partir de representaciones (como GPT)

Los Transformers no son autoencoders tradicionales, pero comparten esta filosofía de codificación-decodificación.
```

### Transformers: El Cambio de Juego

Los Transformers (2017) reemplazan la recurrencia con **mecanismo de atención**. Veremos detalles en la Lectura 4, pero la idea clave:

```
En lugar de procesar palabra por palabra en orden,
el Transformer puede ver toda la secuencia simultáneamente
y aprender qué palabras son relevantes unas para otras.
```

Esto escala mucho mejor y permite entrenar con más datos.

```{admonition} 📚 Conexión
:class: seealso
El mecanismo de atención que mencionamos aquí será explicado en profundidad en la **Lectura 4: Arquitectura Transformer**. Por ahora, basta entender que resuelve el problema de memoria a largo plazo de las RNNs. Antes de llegar allí, la **Lectura 3** cubrirá conceptos básicos de NLP y los modelos secuenciales tradicionales.
```

---

## Parte 4: ¿Cuándo Usar Cada Enfoque?

### Usa Aprendizaje Clásico Si:

- Tienes pocos datos (< 10,000 ejemplos)
- El problema es relativamente simple
- Necesitas interpretabilidad máxima
- Recursos computacionales limitados

**Ejemplo:** Predecir si un cliente cambiará de proveedor basado en 5 características.

### Usa Aprendizaje Profundo Si:

- Tienes muchos datos (> 100,000 ejemplos)
- El problema es complejo (imágenes, texto, audio)
- Puedes acceder a GPUs
- La interpretabilidad es menos crítica

**Ejemplo:** Clasificar radiografías médicas de 1 millón de pacientes.

### Usa Modelos Generativos Si:

- Necesitas generar nuevos contenidos (imágenes, texto, código)
- Quieres un sistema versátil (un LLM puede hacer clasificación, resumir, traducir, etc.)
- Tienes datos abundantes

**Ejemplo:** Chatbot que responde preguntas, resume documentos y genera código.

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **Piensa en un problema de tu ámbito laboral.** ¿Sería mejor resolverlo con reglas explícitas, un modelo discriminativo o un modelo generativo? ¿Por qué?

2. **¿Cuál crees que es el principal tradeoff entre modelos discriminativos y generativos?** ¿Y entre IA clásica y aprendizaje profundo?

3. **Los LLMs son modelos generativos.** ¿Cómo podrían usarse para tareas de clasificación (típicamente discriminativas)?

### Ejercicios Prácticos:

1. **Identifica tres problemas en la industria** (diagnóstico, recomendaciones, detección de fraude, etc.). Para cada uno, decide qué paradigma sería más apropiado y justifica tu respuesta.

2. **Busca un artículo de investigación en arXiv** sobre un modelo generativo reciente. Nota cómo los autores lo comparan con modelos discriminativos. ¿Qué ventaja destacan?

3. **Reflexión escrita (250 palabras):** "La evolución de la IA ha sido una búsqueda de métodos cada vez más generales. ¿Crees que los LLMs son el pico de esa generalidad, o crees que veremos paradigmas aún más generales?"

---

```{admonition} ✅ Verifica tu comprensión
:class: note
1. ¿Cuál es la diferencia fundamental entre un modelo discriminativo y uno generativo en términos matemáticos?
2. Explica con tus propias palabras por qué un perceptrón simple no puede resolver el problema XOR.
3. ¿Por qué es importante la distinción entre P(Y|X) y P(X,Y) para elegir un modelo?
4. Da un ejemplo de un problema donde preferirías usar aprendizaje clásico en lugar de deep learning. Justifica tu respuesta.
```

## Errores Comunes

```{admonition} ⚠️ Errores frecuentes
:class: warning

1. **Confundir paradigmas de aprendizaje**: Supervisado requiere etiquetas, no supervisado busca estructura sin etiquetas. No son intercambiables.
2. **Creer que más datos siempre es mejor**: Sin datos de calidad, más datos solo amplifica el sesgo. Garbage in, garbage out.
3. **Ignorar el costo computacional**: Deep learning no siempre es la respuesta. Para datasets pequeños (<10K), modelos clásicos (Random Forest, SVM) suelen funcionar mejor.
4. **Confundir correlación con causalidad**: Un modelo puede aprender correlaciones espurias. Validación en datos independientes es crítica.
5. **Olvidar la interpretabilidad**: En dominios regulados (medicina, finanzas), modelos interpretables pueden ser obligatorios aunque sean menos precisos.
```

## Resumen

```{admonition} Resumen
:class: important
**Conceptos clave:**
- La IA evolucionó de sistemas basados en reglas explícitas a modelos que aprenden patrones de datos
- Paradigmas de aprendizaje: supervisado (datos etiquetados), no supervisado (estructura oculta), refuerzo (recompensas)
- Discriminativo (P(Y|X)): clasifica datos en categorías; Generativo (P(X,Y)): modela distribución completa y puede generar nuevos datos
- Transformers revolucionaron NLP con atención global, superando limitaciones de RNNs
- La elección del enfoque depende de: cantidad de datos, complejidad, recursos computacionales e interpretabilidad

**Para la siguiente lectura:** Prepárate para profundizar en los fundamentos matemáticos del deep learning: tensores, retropropagación y funciones de activación. Estos conceptos son la base de todos los modelos modernos.
```

## Puntos Clave

- **IA clásica** usaba reglas explícitas; **aprendizaje automático** descubre reglas a partir de datos
- **Supervisado:** aprender de datos etiquetados; **No supervisado:** encontrar estructura; **Refuerzo:** aprender de recompensas
- **Discriminativo:** "¿a qué categoría pertenece?"; **Generativo:** "¿cuál es la distribución de los datos?"
- **Transformers** revolucionaron el NLP reemplazando recurrencia con atención global
- **Elige la herramienta basada en:** cantidad de datos, complejidad del problema, recursos disponibles, necesidad de interpretabilidad

---

## Referencias

- Russell, S. & Norvig, P. (2020). [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/) (4th ed.). Pearson.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/). MIT Press.
- Rosenblatt, F. (1958). [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://psycnet.apa.org/doi/10.1037/h0042519). Psychological Review.
- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). arXiv:1706.03762.
- Ng, A. & Jordan, M. (2001). [On Discriminative vs. Generative Classifiers](https://papers.nips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html). NeurIPS.

