import json

filepath = "/Users/salvahin/TC3002B-2026/book/notebooks/02_fundamentos_deep_learning.ipynb"

with open(filepath, 'r') as f:
    nb = json.load(f)

# Find the simple network forward pass code cell (around line 347)
target_idx_1 = -1
target_idx_2 = -1

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "z1 = W1 @ x + b1" in source and "a1 = relu(z1)" in source:
            target_idx_1 = i
        if "class SimpleNetwork:" in source and "def backward_and_update" not in source: # To avoid modifying the already refactored one if it was same notebook
            target_idx_2 = i

if target_idx_1 != -1:
    print(f"Modifying forward pass cell at {target_idx_1}")
    source = nb['cells'][target_idx_1]['source']
    
    # We will inject shape assertions and explicit comments
    new_source = []
    for line in source:
        new_source.append(line)
        if "x = np.array([0.5, -0.3])" in line:
            new_source.append("    # ACA Framework: Siempre verifica la forma de tus tensores de entrada\n")
            new_source.append("    # x tiene forma (2,) que en numpy se alinea automáticamente para el producto punto\n")
            
        if "W1 = np.array([[0.1, 0.2], [0.3, 0.4]])" in line:
            new_source.append("    # W1 tiene forma (2 entradas x 2 neuronas) = (2, 2)\n")
            
        if "z1 = W1 @ x + b1" in line:
            new_source.insert(-1, "    # Multiplicación: (2,2) @ (2,) -> (2,) + (2,) b1 -> (2,)\n")
            new_source.append("    assert z1.shape == (2,), \"Error en la dimensión de Z1\"\n")
            
        if "z2 = W2 @ a1 + b2" in line:
            new_source.insert(-1, "    # W2 (1, 2) @ a1 (2,) -> (1,) + (1,) b2 -> (1,)\n")
            new_source.append("    assert z2.shape == (1,), \"Error en la dimensión de Z2\"\n")

    nb['cells'][target_idx_1]['source'] = new_source

if target_idx_2 != -1:
    print(f"Modifying backprop cell at {target_idx_2}")
    source = nb['cells'][target_idx_2]['source']
    
    new_source = []
    for line in source:
        new_source.append(line)
        if "dW2 = self.A1.T @ dZ2 / m" in line:
            new_source.append("        # ACA Framework: Aserción de formas (Gradientes deben tener la misma forma que los pesos)\n")
            new_source.append("        assert dW2.shape == self.W2.shape, f\"Dimensión dW2 {dW2.shape} != W2 {self.W2.shape}\"\n")
            
        if "dW1 = self.X.T @ dZ1 / m" in line:
            new_source.append("        assert dW1.shape == self.W1.shape, f\"Dimensión dW1 {dW1.shape} != W1 {self.W1.shape}\"\n")

    nb['cells'][target_idx_2]['source'] = new_source

with open(filepath, 'w') as f:
    json.dump(nb, f, indent=1)
    
print("02_fundamentos_deep_learning.ipynb refactored successfully.")

