import os

BOOK_DIR = "/Users/salvahin/TC3002B-2026/book"

def refactor_nn_md():
    filepath = os.path.join(BOOK_DIR, "ai", "01_ia_clasica_vs_generativa.md")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the big code cell
    if "class NeuralNetwork:" in content and "def train(self, X, y" in content and "refactored" not in content.lower():
        start_idx = content.find("```{code-cell} ipython3\n# Implementación manual de una red neuronal multicapa")
        end_idx = content.find("```", start_idx + 10) + 3
        
        replacement = """
### 1. Construyendo la Red: Forward Pass

En lugar de ver todo el código de entrenamiento de golpe, vamos a construirlo paso a paso.
Primero, necesitamos definir la estructura (pesos y sesgos) y cómo los datos fluyen hacia adelante (`forward pass`).

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class NeuralNetworkForward:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Inicialización de pesos aleatorios
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            # Multiplicación de matrices: Z = X*W + b
            z = activations[-1] @ self.weights[i] + self.biases[i]
            
            # Activación
            if i < len(self.weights) - 1:
                a = np.maximum(0, z) # ReLU
            else:
                a = self.sigmoid(z)  # Salida probabilística
                
            activations.append(a)
        return activations

# Probando el Forward Pass en datos de prueba
dummy_X = np.array([[0.5, -0.2]])
nn_test = NeuralNetworkForward([2, 4, 1])
dummy_out = nn_test.forward(dummy_X)[-1]

print(f"Entrada (Shape {dummy_X.shape}): {dummy_X}")
print(f"Predicción aleatoria (Shape {dummy_out.shape}): {dummy_out}")
```

### 2. El Algoritmo de Retropropagación (Backpropagation)

La red produce predicciones aleatorias porque sus pesos no están entrenados.
Para que aprenda, debemos:
1. Calcular el **Error** (qué tan lejos está la predicción de la realidad).
2. Usar **Backpropagation** para mandar el error hacia atrás y calcular el gradiente (dirección hacia la cual ajustar los pesos).

```{code-cell} ipython3
class NeuralNetwork(NeuralNetworkForward):
    def __init__(self, layer_sizes, learning_rate=0.1):
        super().__init__(layer_sizes)
        self.lr = learning_rate
        
    def backward_and_update(self, y_true, activations):
        # 1. Calcular Error Final
        error = activations[-1] - y_true
        deltas = [error]
        
        # 2. Retropropagar el error a capas ocultas
        for i in range(len(self.weights) - 1, 0, -1):
            delta = deltas[-1] @ self.weights[i].T
            
            # Derivada de ReLU: 1 si activacion > 0, sino 0
            delta = delta * (activations[i] > 0)
            deltas.append(delta)
            
        deltas.reverse()
        
        # 3. Actualizar Pesos (Gradient Descent)
        for i in range(len(self.weights)):
            grad_w = activations[i].T @ deltas[i]
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)
            
            # Aserciones de forma (Crucial para no equivocarse!)
            assert grad_w.shape == self.weights[i].shape, "Error de dimension en pesos"
            
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grad_b

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            # Paso 1: Forward
            activations = self.forward(X)
            
            # Guardar pérdida MSE
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)
            
            # Paso 2: Backward
            self.backward_and_update(y, activations)
            
        return losses
```

### 3. Entrenamiento y Visulización en la práctica

Corramos nuestra red construida paso a paso en un problema no-lineal (las dos lunas).

```{code-cell} ipython3
# Generar datos no lineales (problema XOR-like)
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
y = y.reshape(-1, 1)

# Entrenar
nn = NeuralNetwork([2, 8, 4, 1], learning_rate=0.3)
losses = nn.train(X, y, epochs=1000)

# Visualización simplificada
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Frontera de decisión
ax = axes[0]
h = 0.05
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])[-1]
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap='RdYlBu')
ax.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1], c='blue', s=40, edgecolors='k')
ax.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], c='red', s=40, edgecolors='k')
ax.set_title('Decisión (No Lineal)')

# 2. Curva de aprendizaje
axes[1].plot(losses, color='blue')
axes[1].set_yscale('log')
axes[1].set_title('Reducción del Error (Loss)')

plt.show()

predictions = nn.forward(X)[-1]
print(f"Precisión final: {np.mean((predictions > 0.5) == y)*100:.2f}%")
print(f"La red multicapa resuelve problemas NO LINEALES que un perceptrón simple no puede.")
```
"""
        new_content = content[:start_idx] + replacement.strip() + content[end_idx:]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Patched 01_ia_clasica_vs_generativa.md")

def refactor_dl_md():
    filepath = os.path.join(BOOK_DIR, "ai", "02_fundamentos_deep_learning.md")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "def forward(self, X):" in content and "ACA Framework" not in content:
        # We replace the forward method
        old_forward = """
    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
"""
        new_forward = """
    def forward(self, X):
        self.X = X
        
        # ACA Framework: Inline shape explanations
        # X shape: (batch_size, input_dim) -> e.g. (300, 2)
        # W1 shape: (input_dim, hidden_dim) -> e.g. (2, 4)
        # Result Z1: (300, 2) @ (2, 4) = (300, 4)
        self.Z1 = np.dot(X, self.W1) + self.b1
        assert self.Z1.shape == (X.shape[0], self.W1.shape[1]), f"Z1 shape incorrect: {self.Z1.shape}"
        
        self.A1 = self.relu(self.Z1)
        
        # Z2 shape: (300, 4) @ (4, 1) = (300, 1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        assert self.Z2.shape == (X.shape[0], self.W2.shape[1]), f"Z2 shape incorrect: {self.Z2.shape}"
        
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
"""
        content = content.replace(old_forward, new_forward)
        
        old_backward = """
    def backward(self, y, learning_rate=0.01):
        m = y.shape[0]
        
        # Gradientes capa de salida
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Gradientes capa oculta
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(self.X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Actualización de pesos
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
"""
        new_backward = """
    def backward(self, y, learning_rate=0.01):
        m = y.shape[0]
        
        # Gradientes capa de salida
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # ACA Framework: Checking backward shapes before update
        assert dW2.shape == self.W2.shape, f"dW2 dim mismatch: {dW2.shape} != {self.W2.shape}"
        assert db2.shape == self.b2.shape, f"db2 dim mismatch: {db2.shape} != {self.b2.shape}"
        
        # Gradientes capa oculta
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(self.X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        assert dW1.shape == self.W1.shape, f"dW1 dim mismatch: {dW1.shape} != {self.W1.shape}"
        assert db1.shape == self.b1.shape, f"db1 dim mismatch: {db1.shape} != {self.b1.shape}"
        
        # Actualización de pesos
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
"""
        content = content.replace(old_backward, new_backward)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Patched 02_fundamentos_deep_learning.md")
        
def refactor_fsm_md():
    filepath = os.path.join(BOOK_DIR, "compilers", "06-pipeline-fsm-xgrammar.md")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if "def calculate_first_follow(grammar):" in content and "Probando con una Gramática de Expresiones Aritméticas" not in content:
        # We need to find where calculate_first_follow is called
        start_idx = content.find("# Ejemplo 1: Expresiones aritméticas")
        
        if start_idx != -1:
            end_idx = content.find("# Ejercicio práctico: Usar FIRST/FOLLOW", start_idx)
            
            replacement = """
```

### Probando con una Gramática de Expresiones Aritméticas

Vamos a aplicar nuestra función a una gramática real para ver cómo los conjuntos `FIRST` y `FOLLOW` revelan la estructura del lenguaje antes de que el código siquiera se ejecute.

```{code-cell} ipython3
# Ejemplo 1: Expresiones aritméticas
grammar1 = {
    'rules': [
        "expr   → term (('+' | '-') term)*",
        "term   → factor (('*' | '/') factor)*",
        "factor → '(' expr ')' | number | identifier"
    ],
    'FIRST': {
        'expr': {'(', 'number', 'identifier'},
        'term': {'(', 'number', 'identifier'},
        'factor': {'(', 'number', 'identifier'}
    },
    'FOLLOW': {
        'expr': {')', '$', '+', '-'},  # $ = fin de entrada
        'term': {')', '$', '+', '-', '*', '/'},
        'factor': {')', '$', '+', '-', '*', '/'}
    }
}

# ACA Framework: Imprimamos y analicemos visualmente las reglas paso a paso
calculate_first_follow(grammar1)
```

### Expandiendo a estructuras de control (IF/ELSE)

El mismo motor puede usarse para estructuras más complejas. Observa cómo cambian los conjuntos válidos.

```{code-cell} ipython3
# Ejemplo 2: Statements con if-else
grammar2 = {
    'rules': [
        "statement → if_stmt | assign_stmt | block",
        "if_stmt   → 'if' expr ':' statement ['else' ':' statement]",
        "assign_stmt → identifier '=' expr",
        "block     → '{' statement+ '}'",
        "expr      → identifier | number"
    ],
    'FIRST': {
        'statement': {'if', 'identifier', '{'},
        'if_stmt': {'if'},
        'assign_stmt': {'identifier'},
        'block': {'{'},
        'expr': {'identifier', 'number'}
    },
    'FOLLOW': {
        'statement': {'else', '}', '$'},
        'if_stmt': {'else', '}', '$'},
        'assign_stmt': {'else', '}', '$'},
        'block': {'else', '}', '$'},
        'expr': {':', '=', 'else', '}', '$'}
    }
}

calculate_first_follow(grammar2)
```

### Simulando la Inferencia Constreñida (Constrained Decoding)

Todo este esfuerzo matemático es para que, al momento en el que el LLM intenta predecir la siguiente palabra, nosotros podamos "taparle" el vocabulario no válido.

```{code-cell} ipython3
"""
            new_content = content[:start_idx] + replacement.lstrip() + content[end_idx:]
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("Patched 06-pipeline-fsm-xgrammar.md")

def refactor_gpu_md():
    filepath = os.path.join(BOOK_DIR, "project_2", "01_gpu_fundamentals.md")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "## Jerarquía de Memoria GPU" in content and "ACA Framework" not in content:
        start_idx = content.find("## Jerarquía de Memoria GPU")
        
        replacement = """
### ACA Framework: Simulando el Grid en Python Puro

Antes de ver código compilado para GPU (CUDA o Triton), construyamos el modelo mental de cómo los identificadores mágicos (`blockIdx`, `threadIdx`) mapean tu posición 3D a un arreglo lineal (1D) en memoria. En Python, esto es simple aritmética modular que todo thread de GPU ejecuta en hardware.

```{code-cell} ipython3
# Simulación en Python puro de la indexación GPU
def simulate_gpu_grid_mapping(grid_dim_x, block_dim_x):
    print(f"\\nLanzando Kernel simulado: Grid de {grid_dim_x} bloques, {block_dim_x} threads por bloque")
    print(f"Total de threads que se ejecutarán en paralelo: {grid_dim_x * block_dim_x}")
    print("="*70)
    
    # Simulamos lo que cada thread calcula internamente
    for blockIdx_x in range(grid_dim_x):
        for threadIdx_x in range(block_dim_x):
            
            # FÓRMULA MÁGICA DE CUDA/TRITON (De 2D a 1D)
            global_id = (blockIdx_x * block_dim_x) + threadIdx_x
            
            print(f"Bloque [{blockIdx_x}], Thread local [{threadIdx_x}] -> Accederá al índice Global de Memoria: [{global_id}]")

# Lanzar 3 bloques, cada uno con 4 threads
simulate_gpu_grid_mapping(grid_dim_x=3, block_dim_x=4)
```

## Jerarquía de Memoria GPU
"""
        new_content = content.replace("## Jerarquía de Memoria GPU", replacement.strip())
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Patched 01_gpu_fundamentals.md")

if __name__ == "__main__":
    refactor_nn_md()
    refactor_dl_md()
    refactor_fsm_md()
    refactor_gpu_md()
