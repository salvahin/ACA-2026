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

# Lectura 1: IA Clásica vs Generativa

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Distinguir entre paradigmas de IA clásica (sistemas expertos), discriminativos y generativos
- Identificar las ventajas y limitaciones de cada enfoque según el contexto del problema
- Comprender la evolución histórica desde sistemas basados en reglas hasta modelos de deep learning
- Seleccionar el paradigma apropiado (clásico, discriminativo o generativo) para un problema dado
```

## Contexto
En esta lectura comprenderás la evolución de la IA desde sistemas basados en reglas hasta modelos generativos modernos. Aprenderás las diferencias entre paradigmas discriminativos y generativos, y cuándo aplicar cada enfoque.

## Introducción

Imagina que quieres construir un sistema inteligente. Hace 50 años, los investigadores tenían una idea clara: *codifiquemos las reglas del sentido común directamente en la máquina*. Si querías un sistema que diagnosticara enfermedades, le escribías reglas: "Si el paciente tiene fiebre Y tos Y radiografía anormal, entonces probablemente tiene neumonía."

Ese enfoque funcionaba para problemas simples, pero se topaba con una pared: ¿qué pasa cuando las reglas son demasiado complejas para escribir explícitamente? ¿Cómo codificas el patrón visual que distingue un gato de un perro?

En esta lectura exploraremos cómo la IA ha evolucionado desde ese primer enfoque de **reglas explicitas** hasta los **modelos generativos** modernos como ChatGPT. Veremos que cada enfoque responde a la pregunta: "¿Cómo podemos hacer que las máquinas aprendan?"

---

## Parte 1: El Viaje del Aprendizaje Automático

### Era 1: Sistemas Expertos (1960s-1980s)

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

Imagina que tienes 1,000 fotos de gatos y 1,000 de perros. Un algoritmo de aprendizaje automático examine automáticamente estas imágenes y descubra patrones: "Los gatos tienen orejas triangulares, los perros tienen orejas más redondeadas," etc.

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

Las redes neuronales profundas revolucionaron todo. Pero antes de hablar de eso, necesitamos entender **modelos discriminativos vs generativos**.

---

## Parte 2: Discriminativos vs Generativos

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

:::{figure} diagrams/discriminative_vs_generative.png
:name: fig-disc-gen
:alt: Comparación visual entre modelos discriminativos y generativos
:align: center
:width: 90%

**Figura 2:** Modelos Discriminativos vs Generativos - comparación de enfoques de aprendizaje.
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

# Demostración: Problema AND (linealmente separable)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron()
perceptron.fit(X_and, y_and)

# Visualización
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

# XOR (no funciona)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

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

### Redes Neuronales Multicapa

```
Capa de Entrada → Capas Ocultas → Capa de Salida
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

# Generar datos no lineales (problema XOR-like)
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
y = y.reshape(-1, 1)

# Entrenar red neuronal multicapa
nn = NeuralNetwork([2, 8, 4, 1], learning_rate=0.3)
losses = nn.train(X, y, epochs=1000)

# Visualización
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

### Transformers: El Cambio de Juego

Los Transformers (2017) reemplazan la recurrencia con **mecanismo de atención**. Veremos detalles en la próxima lectura, pero la idea clave:

```
En lugar de procesar palabra por palabra en orden,
el Transformer puede ver toda la secuencia simultáneamente
y aprender qué palabras son relevantes unas para otras.
```

Esto escala mucho mejor y permite entrenar con más datos.

```{admonition} 📚 Conexión
:class: seealso
El mecanismo de atención que mencionamos aquí será explicado en profundidad en la Lectura 4: Arquitectura Transformer. Por ahora, basta entender que resuelve el problema de memoria a largo plazo de las RNNs.
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

