# Simulaciones Interactivas - Project_1

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/SIMULACIONES_README.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton xgrammar
print('Dependencies installed!')
```

Este documento describe las simulaciones interactivas implementadas en el módulo Project_1.

## Archivos Modificados

### 1. `03_prompt_engineering.md`
**Simulación**: Anatomía de un Prompt Efectivo

**Ubicación**: Sección "Anatomía de un Prompt para Código"

**Descripción**: Visualización interactiva que muestra los 5 componentes principales de un prompt efectivo:
- System Prompt (Define el rol y comportamiento)
- Context (Información relevante)
- Few-shot Examples (Ejemplos de entrada/salida)
- User Query (La pregunta del usuario)
- Output Format (Cómo debe responder)

**Tecnología**: Plotly (gráfico de rectángulos con anotaciones)

**Funcionamiento**:
- Muestra visualmente la jerarquía de componentes
- Cada componente tiene un color distintivo
- Las descripciones están integradas en el gráfico

---

### 2. `04_xgrammar_constrained.md`
**Simulación**: Token Masking por Estado de la Gramática

**Ubicación**: Sección "Paso 6: LogitsProcessor"

**Descripción**: Visualización interactiva con dropdown que muestra cómo XGrammar enmascara tokens según el estado de la gramática:
- Estado "Inicio": Solo permite 'def'
- Estado "Después def": Solo permite identificadores
- Estado "Después (": Permite parámetros o ')'
- Estado "Body": Permite return, if, variables, números

**Tecnología**: Plotly (gráfico de barras con dropdown interactivo)

**Funcionamiento**:
- Verde = token permitido
- Rojo = token bloqueado
- Dropdown permite cambiar entre estados
- Demuestra el concepto de constrained decoding

---

### 3. `05_json_grammar.md`
**Simulación**: Estructura del JSON Schema para Kernels Triton

**Ubicación**: Sección "Diseño del Schema"

**Descripción**: Visualización de árbol que muestra la estructura jerárquica del JSON Schema:
- Nodo raíz: TritonKernel
- Ramas principales: name, args, body
- Subramas: tipos de argumentos, declaraciones

**Tecnología**: Plotly (gráfico de dispersión con líneas y nodos)

**Funcionamiento**:
- Nodos representan elementos del schema
- Líneas representan relaciones padre-hijo
- Colores distinguen diferentes tipos de componentes
- Muestra la estructura completa de un kernel Triton

---

### 4. `09_token_economics_integracion.md`
**Simulación**: Calculadora de Costos de API

**Ubicación**: Sección "Parte 1: Precios de API"

**Descripción**: Gráfico de barras comparativo que calcula costos mensuales para diferentes modelos:
- GPT-4o
- GPT-4o-mini
- Claude 3.5 Sonnet
- Claude 3 Haiku
- Llama 3.1 405B

**Escenario**: 10,000 requests/día con 1,000 tokens de entrada y 500 de salida

**Tecnología**: Plotly (gráfico de barras con colores condicionales)

**Funcionamiento**:
- Verde: costo bajo (<$100)
- Azul: costo medio ($100-$1000)
- Rojo: costo alto (>$1000)
- Muestra el costo total en cada barra

---

## Compatibilidad

### GitHub Pages
Las simulaciones funcionan en GitHub Pages porque:
- MyST Markdown soporta `{code-cell}` que se ejecuta durante la compilación
- Plotly genera HTML interactivo que se integra directamente
- No requiere servidor backend

### Google Colab
Las simulaciones funcionan en Google Colab porque:
- Todos los bloques están marcados como `{code-cell} ipython3`
- Plotly está preinstalado en Colab
- El código es Python estándar sin dependencias especiales

## Instalación de Dependencias

Para ejecutar las simulaciones localmente:

```bash
pip install plotly numpy matplotlib
```

Para Google Colab (ya incluido):
```python
# No se requiere instalación adicional
import plotly.graph_objects as go
```

## Características Técnicas

### Plotly
- **Ventaja**: Gráficos HTML interactivos que funcionan sin servidor
- **Renderizado**: Se genera HTML/JS que se puede embeber
- **Interactividad**: Dropdown menus, tooltips, zoom

### Diseño de Colores
Paleta consistente en todas las simulaciones:
- `#FF6B6B` - Rojo (elementos principales/críticos)
- `#4ECDC4` - Turquesa (contexto/información)
- `#45B7D1` - Azul (ejemplos/datos)
- `#FFE66D` - Amarillo (interacción/usuario)
- `#95E1A3` - Verde (output/resultados)
- `#DDA0DD` - Púrpura (código/statements)

## Prueba de Funcionamiento

Para verificar que las simulaciones funcionan correctamente:

1. **Localmente con Jupyter**:
   ```bash
   jupyter notebook book/project_1/03_prompt_engineering.md
   ```

2. **En Google Colab**:
   - Abrir el archivo en Colab
   - Ejecutar cada celda de código
   - Verificar que los gráficos se renderizan

3. **En GitHub Pages**:
   - Compilar el libro con Jupyter Book
   - Verificar que los gráficos interactivos aparezcan en HTML

## Solución de Problemas

### Si las simulaciones no se muestran:

1. **Verificar importaciones**:
   ```python
   import plotly.graph_objects as go
   import numpy as np
   ```

2. **Verificar versión de Plotly**:
   ```python
   import plotly
   print(plotly.__version__)  # Debe ser >= 5.0
   ```

3. **En Jupyter Book**, verificar que `_config.yml` incluye:
   ```yaml
   execute:
     execute_notebooks: auto
   ```

## Extensión Futura

Ideas para expandir las simulaciones:

1. **03_prompt_engineering.md**: Agregar comparación de resultados con diferentes prompts
2. **04_xgrammar_constrained.md**: Agregar visualización de FSM completo
3. **05_json_grammar.md**: Agregar validador en tiempo real
4. **09_token_economics_integracion.md**: Agregar calculadora interactiva con inputs del usuario

## Contacto

Para preguntas sobre las simulaciones, consultar la documentación de:
- Plotly: https://plotly.com/python/
- Jupyter Book: https://jupyterbook.org/
- MyST Markdown: https://myst-parser.readthedocs.io/
