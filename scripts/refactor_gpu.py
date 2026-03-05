import json

filepath = "/Users/salvahin/TC3002B-2026/book/notebooks/01_gpu_fundamentals.ipynb"

with open(filepath, 'r') as f:
    nb = json.load(f)

# Find where to inject the python simulation of GPU thread hierarchy
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell.get('source', []))
        if "## Jerarquía de Memoria GPU" in source:
            target_idx = i
            break

if target_idx != -1:
    
    sim_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### ACA Framework: Simulando el Grid en Python Puro\n",
            "\n",
            "Antes de ver código compilado para GPU (CUDA o Triton), construyamos el modelo mental de cómo los identificadores mágicos (`blockIdx`, `threadIdx`) mapean tu posición 3D a un arreglo lineal (1D) en memoria. En Python, esto es simple aritmética modular que todo thread de GPU ejecuta en hardware.\n"
        ]
    }
    
    sim_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Simulación en Python puro de la indexación GPU\n",
            "def simulate_gpu_grid_mapping(grid_dim_x, block_dim_x):\n",
            "    print(f\"\\nLanzando Kernel simulado: Grid de {grid_dim_x} bloques, {block_dim_x} threads por bloque\")\n",
            "    print(f\"Total de threads que se ejecutarán en paralelo: {grid_dim_x * block_dim_x}\")\n",
            "    print(\"=\"*70)\n",
            "    \n",
            "    # Simulamos lo que cada thread calcula internamente\n",
            "    for blockIdx_x in range(grid_dim_x):\n",
            "        for threadIdx_x in range(block_dim_x):\n",
            "            \n",
            "            # FÓRMULA MÁGICA DE CUDA/TRITON (De 2D a 1D)\n",
            "            global_id = (blockIdx_x * block_dim_x) + threadIdx_x\n",
            "            \n",
            "            print(f\"Bloque [{blockIdx_x}], Thread local [{threadIdx_x}] -> Accederá al índice Global de Memoria: [{global_id}]\")\n",
            "\n",
            "# Lanzar 3 bloques, cada uno con 4 threads\n",
            "simulate_gpu_grid_mapping(grid_dim_x=3, block_dim_x=4)\n"
        ]
    }
    
    nb['cells'] = nb['cells'][:target_idx] + [sim_md, sim_code] + nb['cells'][target_idx:]
    
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook 01_gpu_fundamentals successfully refactored.")

