import json

filepath = "/Users/salvahin/TC3002B-2026/book/notebooks/01_ia_clasica_vs_generativa.ipynb"

with open(filepath, 'r') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the index of the big NN code cell
target_idx = -1
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "class NeuralNetwork:" in source and "def train(self, X, y" in source:
            target_idx = i
            break

if target_idx != -1:
    print(f"Found target cell at index {target_idx}")
    
    # We will replace this cell with a sequence of markdown and code cells
    
    # 1. Forward Pass Only
    cell_fw_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Construyendo la Red: Forward Pass\n",
            "\n",
            "En lugar de ver todo el código de entrenamiento de golpe, vamos a construirlo paso a paso.\n",
            "Primero, necesitamos definir la estructura (pesos y sesgos) y cómo los datos fluyen hacia adelante (`forward pass`).\n"
        ]
    }
    
    cell_fw_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.datasets import make_moons\n",
            "\n",
            "class NeuralNetworkForward:\n",
            "    def __init__(self, layer_sizes):\n",
            "        self.layer_sizes = layer_sizes\n",
            "        self.weights = []\n",
            "        self.biases = []\n",
            "        \n",
            "        # Inicialización de pesos aleatorios\n",
            "        for i in range(len(layer_sizes) - 1):\n",
            "            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1\n",
            "            b = np.zeros((1, layer_sizes[i+1]))\n",
            "            self.weights.append(w)\n",
            "            self.biases.append(b)\n",
            "            \n",
            "    def sigmoid(self, z):\n",
            "        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))\n",
            "        \n",
            "    def forward(self, X):\n",
            "        activations = [X]\n",
            "        for i in range(len(self.weights)):\n",
            "            # Multiplicación de matrices: Z = X*W + b\n",
            "            z = activations[-1] @ self.weights[i] + self.biases[i]\n",
            "            \n",
            "            # Activación\n",
            "            if i < len(self.weights) - 1:\n",
            "                a = np.maximum(0, z) # ReLU\n",
            "            else:\n",
            "                a = self.sigmoid(z)  # Salida probabilística\n",
            "                \n",
            "            activations.append(a)\n",
            "        return activations\n",
            "\n",
            "# Probando el Forward Pass en datos de prueba\n",
            "dummy_X = np.array([[0.5, -0.2]])\n",
            "nn_test = NeuralNetworkForward([2, 4, 1])\n",
            "dummy_out = nn_test.forward(dummy_X)[-1]\n",
            "\n",
            "print(f\"Entrada (Shape {dummy_X.shape}): {dummy_X}\")\n",
            "print(f\"Predicción aleatoria (Shape {dummy_out.shape}): {dummy_out}\")\n"
        ]
    }
    
    # 2. Backpropagation
    cell_bw_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. El Algoritmo de Retropropagación (Backpropagation)\n",
            "\n",
            "La red produce predicciones aleatorias porque sus pesos no están entrenados.\n",
            "Para que aprenda, debemos:\n",
            "1. Calcular el **Error** (qué tan lejos está la predicción de la realidad).\n",
            "2. Usar **Backpropagation** para mandar el error hacia atrás y calcular el gradiente (dirección hacia la cual ajustar los pesos).\n"
        ]
    }
    
    cell_bw_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class NeuralNetwork(NeuralNetworkForward):\n",
            "    def __init__(self, layer_sizes, learning_rate=0.1):\n",
            "        super().__init__(layer_sizes)\n",
            "        self.lr = learning_rate\n",
            "        \n",
            "    def backward_and_update(self, y_true, activations):\n",
            "        # 1. Calcular Error Final\n",
            "        error = activations[-1] - y_true\n",
            "        deltas = [error]\n",
            "        \n",
            "        # 2. Retropropagar el error a capas ocultas\n",
            "        for i in range(len(self.weights) - 1, 0, -1):\n",
            "            delta = deltas[-1] @ self.weights[i].T\n",
            "            \n",
            "            # Derivada de ReLU: 1 si activacion > 0, sino 0\n",
            "            delta = delta * (activations[i] > 0)\n",
            "            deltas.append(delta)\n",
            "            \n",
            "        deltas.reverse()\n",
            "        \n",
            "        # 3. Actualizar Pesos (Gradient Descent)\n",
            "        for i in range(len(self.weights)):\n",
            "            grad_w = activations[i].T @ deltas[i]\n",
            "            grad_b = np.sum(deltas[i], axis=0, keepdims=True)\n",
            "            \n",
            "            # Aserciones de forma (Crucial para no equivocarse!)\n",
            "            assert grad_w.shape == self.weights[i].shape, \"Error de dimension en pesos\"\n",
            "            \n",
            "            self.weights[i] -= self.lr * grad_w\n",
            "            self.biases[i] -= self.lr * grad_b\n",
            "\n",
            "    def train(self, X, y, epochs=1000):\n",
            "        losses = []\n",
            "        for epoch in range(epochs):\n",
            "            # Paso 1: Forward\n",
            "            activations = self.forward(X)\n",
            "            \n",
            "            # Guardar pérdida MSE\n",
            "            loss = np.mean((activations[-1] - y) ** 2)\n",
            "            losses.append(loss)\n",
            "            \n",
            "            # Paso 2: Backward\n",
            "            self.backward_and_update(y, activations)\n",
            "            \n",
            "        return losses\n"
        ]
    }
    
    # 3. Viz
    cell_viz_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Entrenamiento y Visulización en la práctica\n",
            "\n",
            "Corramos nuestra red construida paso a paso en un problema no-lineal (las dos lunas).\n"
        ]
    }
    
    cell_viz_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generar datos no lineales (problema XOR-like)\n",
            "X, y = make_moons(n_samples=300, noise=0.15, random_state=42)\n",
            "y = y.reshape(-1, 1)\n",
            "\n",
            "# Entrenar\n",
            "nn = NeuralNetwork([2, 8, 4, 1], learning_rate=0.3)\n",
            "losses = nn.train(X, y, epochs=1000)\n",
            "\n",
            "# Visualización simplificada\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "\n",
            "# 1. Frontera de decisión\n",
            "ax = axes[0]\n",
            "h = 0.05\n",
            "x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5\n",
            "y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5\n",
            "xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\n",
            "\n",
            "Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])[-1]\n",
            "Z = Z.reshape(xx.shape)\n",
            "\n",
            "ax.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap='RdYlBu')\n",
            "ax.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1], c='blue', s=40, edgecolors='k')\n",
            "ax.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], c='red', s=40, edgecolors='k')\n",
            "ax.set_title('Decisión (No Lineal)')\n",
            "\n",
            "# 2. Curva de aprendizaje\n",
            "axes[1].plot(losses, color='blue')\n",
            "axes[1].set_yscale('log')\n",
            "axes[1].set_title('Reducción del Error (Loss)')\n",
            "\n",
            "plt.show()\n",
            "\n",
            "predictions = nn.forward(X)[-1]\n",
            "print(f\"Precisión final: {np.mean((predictions > 0.5) == y)*100:.2f}%\")\n",
            "print(f\"La red multicapa resuelve problemas NO LINEALES que un perceptrón simple no puede.\")\n"
        ]
    }
    
    # Assemble
    new_cells = cells[:target_idx] + [cell_fw_md, cell_fw_code, cell_bw_md, cell_bw_code, cell_viz_md, cell_viz_code] + cells[target_idx+1:]
    nb['cells'] = new_cells
    
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook successfully refactored.")
else:
    print("Target cell not found.")

