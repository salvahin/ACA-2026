import json

filepath = "/Users/salvahin/TC3002B-2026/book/notebooks/06-pipeline-fsm-xgrammar.ipynb"

with open(filepath, 'r') as f:
    nb = json.load(f)

# Find the cell computing FIRST and FOLLOW sets
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "def calculate_first_follow(grammar):" in source:
            target_idx = i
            break

if target_idx != -1:
    source = nb['cells'][target_idx]['source']
    
    # We will split this cell
    new_source = []
    
    # Let's just insert a clearer explanation before the calculate_first_follow function gets called on grammar1
    for line in source:
        if "# Ejemplo 1: Expresiones aritméticas" in line:
            # We split here
            break
        new_source.append(line)
        
    part_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_source
    }
    
    part_2_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Probando con una Gramática de Expresiones Aritméticas\n",
            "\n",
            "Vamos a aplicar nuestra función a una gramática real para ver cómo los conjuntos `FIRST` y `FOLLOW` revelan la estructura del lenguaje antes de que el código siquiera se ejecute.\n"
        ]
    }
    
    part_2_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ejemplo 1: Expresiones aritméticas\n",
            "grammar1 = {\n",
            "    'rules': [\n",
            "        \"expr   → term (('+' | '-') term)*\",\n",
            "        \"term   → factor (('*' | '/') factor)*\",\n",
            "        \"factor → '(' expr ')' | number | identifier\"\n",
            "    ],\n",
            "    'FIRST': {\n",
            "        'expr': {'(', 'number', 'identifier'},\n",
            "        'term': {'(', 'number', 'identifier'},\n",
            "        'factor': {'(', 'number', 'identifier'}\n",
            "    },\n",
            "    'FOLLOW': {\n",
            "        'expr': {')', '$', '+', '-'},  # $ = fin de entrada\n",
            "        'term': {')', '$', '+', '-', '*', '/'},\n",
            "        'factor': {')', '$', '+', '-', '*', '/'}\n",
            "    }\n",
            "}\n",
            "\n",
            "# ACA Framework: Imprimamos y analicemos visualmente las reglas paso a paso\n",
            "calculate_first_follow(grammar1)\n"
        ]
    }
    
    part_3_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Expandiendo a estructuras de control (IF/ELSE)\n",
            "\n",
            "El mismo motor puede usarse para estructuras más complejas. Observa cómo cambian los conjuntos válidos.\n"
        ]
    }
    
    part_3_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ejemplo 2: Statements con if-else\n",
            "grammar2 = {\n",
            "    'rules': [\n",
            "        \"statement → if_stmt | assign_stmt | block\",\n",
            "        \"if_stmt   → 'if' expr ':' statement ['else' ':' statement]\",\n",
            "        \"assign_stmt → identifier '=' expr\",\n",
            "        \"block     → '{' statement+ '}'\",\n",
            "        \"expr      → identifier | number\"\n",
            "    ],\n",
            "    'FIRST': {\n",
            "        'statement': {'if', 'identifier', '{'},\n",
            "        'if_stmt': {'if'},\n",
            "        'assign_stmt': {'identifier'},\n",
            "        'block': {'{'},\n",
            "        'expr': {'identifier', 'number'}\n",
            "    },\n",
            "    'FOLLOW': {\n",
            "        'statement': {'else', '}', '$'},\n",
            "        'if_stmt': {'else', '}', '$'},\n",
            "        'assign_stmt': {'else', '}', '$'},\n",
            "        'block': {'else', '}', '$'},\n",
            "        'expr': {':', '=', 'else', '}', '$'}\n",
            "    }\n",
            "}\n",
            "\n",
            "calculate_first_follow(grammar2)\n"
        ]
    }
    
    # We reconstruct the cells
    # We need to find where "# Ejercicio práctico: Usar FIRST/FOLLOW para constrained decoding" starts in the original source to keep it
    
    part_4_source = []
    capture = False
    for line in source:
        if "# Ejercicio práctico: Usar FIRST/FOLLOW" in line:
            capture = True
        if capture:
            part_4_source.append(line)
            
    part_4_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Simulando la Inferencia Constreñida (Constrained Decoding)\n",
            "\n",
            "Todo este esfuerzo matemático es para que, al momento en el que el LLM intenta predecir la siguiente palabra, nosotros podamos \"taparle\" el vocabulario no válido.\n"
        ]
    }
    
    part_4_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": part_4_source
    }
    
    nb['cells'] = nb['cells'][:target_idx] + [part_1, part_2_md, part_2_code, part_3_md, part_3_code, part_4_md, part_4_code] + nb['cells'][target_idx+1:]
    
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook 06 successfully refactored.")

