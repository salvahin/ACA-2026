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

# 01. Setup del Entorno

```{code-cell} ipython3
import torch
import warnings

# Selección dinámica de dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    warnings.warn("No se detectó un acelerador (GPU/MPS). La ejecución será lenta.")

print(f"Usando dispositivo: {device}")
```


```{code-cell} ipython3
# Setup condicional para Google Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q transformers bitsandbytes triton vllm auto-gptq datasets evaluate
    # Nota: la lista anterior puede contener librerías extra, las cuales Colab ignorará o instalará rápido.
```



```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/01_setup_intro_llms.ipynb)
```


```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Configurar entornos virtuales reproducibles usando venv o conda
- Clonar repositorios con tags específicos para garantizar versiones exactas
- Compilar librerías C++ con bindings Python (XGrammar) usando CMake
- Crear archivos requirements.txt para reproducibilidad total
- Implementar scripts de automatización para configuración de proyectos
```

```{admonition} 📋 Prerrequisitos
:class: attention
Antes de esta lectura, asegúrate de tener:
- Python 3.8+ instalado en tu sistema
- Conocimientos básicos de línea de comandos (bash/terminal)
- Git instalado y configurado
- Acceso a terminal/consola en tu sistema operativo
```

## Introducción

Cuando comenzamos un proyecto de investigación o desarrollo, lo primero que debemos hacer es preparar nuestro entorno de manera **reproducible y consistente**. Imagina que das el código a un colega y no funciona en su máquina, pero sí en la tuya. ¿Frustante, verdad? Todo se debe a diferencias en el entorno.

En esta lectura aprenderemos cómo configurar correctamente un proyecto Python para garantizar que cualquiera pueda replicar exactamente tu trabajo. Cubriremos tres componentes esenciales: clonar repositorios con versiones específicas, manejar entornos virtuales, y construir librerías complejas como XGrammar.

## ¿Por qué importa el setup?

El entorno es como la receta de un pastel: si alguien usa harina diferente o deja el horno a otra temperatura, el resultado será distinto. En ciencia computacional, necesitamos exactitud. Pequeñas diferencias de versiones pueden causar comportamientos inesperados.

**Principio fundamental**: Tu código debe funcionar de forma idéntica en cualquier máquina que tenga el entorno correcto.

## Clonación de Repositorios con Tags Específicos

### El problema de "HEAD fluctuante"

Cuando clonas un repositorio sin especificar una versión, obtienes la rama principal (usualmente `main` o `master`) en su estado más reciente. Pero la rama evoluciona constantemente. Dos clonaciones en diferentes momentos pueden dar código distinto.

```bash
# ❌ No recomendado - obtienes lo que hay ahora
git clone https://github.com/milab-ai/xgrammar.git
cd xgrammar
# ¿Qué versión tengo? Nadie lo sabe exactamente...
```

### Usando tags para reproducibilidad

Los repositorios usan **tags** para marcar versiones específicas. Puedes pensar en un tag como una fotografía del código en un momento exacto.

```bash
# ✓ Recomendado - reproducible y explícito
git clone https://github.com/milab-ai/xgrammar.git
cd xgrammar
git checkout v0.1.0  # O el tag que necesites
```

¿Cómo saber qué tags están disponibles?

```bash
git tag -l  # Lista todos los tags disponibles
git show v0.1.0  # Ver información sobre un tag específico
```

### Crear tu propio pin de versiones

En un proyecto real, documentarías las versiones exactas:

```
# requirements-dev.txt
xgrammar==0.1.0
torch==2.1.0
triton==2.2.0
numpy==1.24.3
```

La razón: si alguien instala dentro de un año, sin estos pins instalará versiones completamente diferentes, y probablemente el código no funcione.

## Entornos Virtuales: El Aislamiento

### ¿Qué es un entorno virtual?

Cada proyecto Python puede necesitar versiones diferentes de las mismas librerías. Si instalaras todo directamente en Python del sistema, tendríamos conflictos masivos.

Un entorno virtual es una carpeta especial que contiene su propia instalación de Python y paquetes. Es como tener una "máquina virtual ligera" solo para nuestro proyecto.

### Usando `venv` (built-in)

```bash
# Crear un entorno virtual
python3 -m venv venv

# Activarlo (Linux/Mac)
source venv/bin/activate

# Activarlo (Windows)
venv\Scripts\activate

# Confirmar que está activo (verás "venv" en tu prompt)
which python  # Debe apuntar a venv/bin/python
```

Una vez activado, cualquier `pip install` va solo a este entorno.

```bash
pip install torch==2.1.0
pip list  # Verás solo lo que instalaste en THIS venv
```

Para salir:

```bash
deactivate
```

### Usando `conda` (alternativa poderosa)

`conda` es un gestor de entornos más sofisticado, especialmente útil para dependencias complejas:

```bash
# Crear un entorno
conda create --name grammar-gpu python=3.11

# Activarlo
conda activate grammar-gpu

# Desactivarlo
conda deactivate
```

Ventajas de conda:
- Maneja dependencias binarias (como CUDA) automáticamente
- Puede instalar paquetes que no están en PyPI
- Mejor control de versiones

```bash
# Instalar desde conda-forge
conda install -c conda-forge xgboost

# Instalar mezcla de conda y pip
conda install pytorch::pytorch
pip install triton
```

## El archivo `requirements.txt`

Este archivo es tu "lista de compras" para reproducibilidad.

### Formato básico

```txt
# requirements.txt
torch==2.1.0
triton==2.2.0
numpy==1.24.3
pydantic==2.3.0
```

El operador `==` significa "exactamente esta versión". Otros operadores existen:

```txt
torch>=2.0.0      # Versión 2.0.0 o posterior
torch<3.0.0       # Anterior a 3.0.0
torch>=2.0,<3.0   # Entre 2.0 (inclusive) y 3.0 (exclusivo)
```

### Generando requirements.txt

```bash
# Genera un archivo con todo lo instalado (útil, pero impreciso)
pip freeze > requirements.txt
```

El problema: `pip freeze` incluye *todo*, incluso dependencias de dependencias. Es mejor escribir manualmente solo lo que tu código importa directamente.

```txt
# requirements.txt (hecho a mano - mejor)
torch==2.1.0
triton==2.2.0
numpy==1.24.3  # Necesario si lo usas directamente
pydantic==2.3.0
```

### Reproducibilidad exacta con pip-tools

Para proyectos serios, existe una herramienta que resuelve dependencias de forma determinista:

```bash
pip install pip-tools

# Creas un archivo simple
# requirements.in
torch
triton>=2.2
```

```bash
# Esto genera requirements.txt con TODAS las versiones pinned
pip-compile requirements.in
```

El resultado es un archivo que garantiza exactitud total.

## Compilando XGrammar: CMake y nanobind

XGrammar es una librería C++ con bindings Python. Compilarla requiere pasos especiales.

### Prerequisitos

XGrammar usa:
- **CMake**: Sistema de construcción
- **nanobind**: Para crear bindings Python desde C++
- Un compilador C++ (GCC, Clang, MSVC)

```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential

# macOS (asumiendo Homebrew)
brew install cmake llvm

# Windows: Descargar CMake desde cmake.org
```

### Compilación paso a paso

```bash
# 1. Clonar con tag específico
git clone https://github.com/milab-ai/xgrammar.git
cd xgrammar
git checkout v0.1.0

# 2. Crear directorio de construcción
mkdir build
cd build

# 3. Ejecutar CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. Compilar
cmake --build . --config Release

# 5. Instalar (opcional)
cmake --install .
```

### Instalación desde fuente (en desarrollo)

Si quieres modificar XGrammar durante el desarrollo:

```bash
# Desde la raíz del repo (no en build/)
pip install -e .
```

El flag `-e` significa "editable". Cambios en el código fuente se reflejan sin recompilar.

### Verificar que XGrammar funciona

```{code-cell} ipython3
import xgrammar as xgr

# Si esto funciona sin errores, ¡está instalado!
schema = {"type": "object"}
compiled = xgr.compile(schema)
print("XGrammar funcionando correctamente")
```

## Configuración de Proyecto Completa

Aquí está el workflow completo para un nuevo miembro del equipo:

```bash
# Paso 1: Clonar el proyecto
git clone https://github.com/tu-org/grammar-kernel-project.git
cd grammar-kernel-project

# Paso 2: Clonar dependencias específicas
git submodule update --init --recursive

# Paso 3: Crear entorno
conda create -n grammar-gpu python=3.11
conda activate grammar-gpu

# Paso 4: Instalar requisitos
pip install -r requirements.txt

# Paso 5: Compilar XGrammar
cd xgrammar
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
pip install -e ..  # Desde raíz de xgrammar
cd ../../

# Paso 6: Verificar
python -c "import xgrammar; print('✓ Todo listo')"
```

### Script de automatización

Para hacerlo aún más fácil, crea un script `setup.sh`:

```bash
#!/bin/bash
set -e  # Salir si algo falla

echo "Configurando entorno..."

# Crear conda env
conda create -n grammar-gpu python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate grammar-gpu

# Instalar requisitos
pip install -r requirements.txt

# Compilar XGrammar
cd xgrammar
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
pip install -e ..
cd ../../

echo "✓ Setup completado. Ejecuta: conda activate grammar-gpu"
```

```bash
chmod +x setup.sh
./setup.sh
```

```{admonition} 🔗 Del concepto al código
:class: hint
**Reproducibilidad en acción:**
- **Teoría:** Tags de Git congelan una versión específica del código
- **Práctica:** `git checkout v0.1.0` garantiza que todos trabajen con la misma versión
- **Resultado:** Tu colega obtiene exactamente el mismo código que tú, eliminando "funciona en mi máquina"

**Entornos virtuales aislados:**
- **Teoría:** Cada proyecto necesita sus propias dependencias sin conflictos
- **Práctica:** `python -m venv venv` crea un entorno aislado
- **Resultado:** Proyecto A usa NumPy 1.24, Proyecto B usa NumPy 1.26, sin conflictos
```

```{admonition} 🏁 Milestone del Proyecto
:class: important
Después de esta lectura, podrás completar:
**Milestone 0: Configuración del Entorno**
- Entorno virtual configurado con todas las dependencias
- XGrammar compilado e instalado correctamente
- Script de validación ejecutándose sin errores
- Proyecto listo para implementar gramáticas (Lectura 4)

Este es el fundamento para todo el proyecto KernelAgent. Sin un entorno correcto, las gramáticas no compilarán y la generación fallará.
```

```{admonition} 🐛 Tips de Debugging
:class: warning
**Si tu instalación falla, verifica:**

1. **Error "command not found: cmake"**
   - Solución: Instala CMake (`brew install cmake` en macOS, `apt-get install cmake` en Ubuntu)

2. **Error "Python.h: No such file or directory"**
   - Solución: Instala python-dev (`apt-get install python3-dev`)

3. **ImportError: No module named 'xgrammar'**
   - Solución: Verifica que estás en el entorno virtual correcto (`which python`)

4. **Versiones incompatibles de dependencias**
   - Solución: Usa `pip install -r requirements.txt` con versiones exactas (con `==`)

5. **Compilación lenta o falla**
   - Solución: Asegúrate de tener suficiente RAM (>4GB) y espacio en disco (>2GB)
```

## Ejercicios

1. **Reproduce una clonación**:
   - Clona XGrammar con un tag específico
   - Verifica que `git describe --tags` muestra el tag correcto
   - Crea un entorno virtual e instálalo
   - Ejecuta el script de validación

2. **Problema de reproducibilidad**:
   - Crea un `requirements.txt` sin pins de versión
   - En una máquina virtual o contenedor distinto, instálalo dos veces
   - Usa `pip freeze` para ver si las versiones son idénticas
   - ¿Lo fueron? ¿Por qué sí o por qué no?

3. **CMake exploration**:
   - En un proyecto que uses, ejecuta `cmake --help-variable CMAKE_BUILD_TYPE`
   - Explica qué diferencia hay entre `Release` y `Debug`
   - ¿Por qué usamos `Release` para código en producción?

```{admonition} ✅ Verifica tu comprensión
:class: note
1. ¿Por qué es crítico usar tags de Git en lugar de clonar desde `main`?
   - Pista: Piensa en qué pasa si la rama main cambia mañana

2. Explica la diferencia entre `pip install numpy` y `pip install numpy==1.24.3`
   - ¿Cuál es más reproducible y por qué?

3. ¿Qué ventaja tiene CMake sobre ejecutar `gcc` manualmente?
   - Considera proyectos con múltiples archivos y plataformas

4. Diseña un flujo de setup para un nuevo miembro del equipo que nunca ha usado Python
   - Incluye: instalación de Python, creación de venv, instalación de dependencias
```

## Preguntas de Reflexión

- ¿Cuál es la diferencia conceptual entre un tag de Git y una rama?
- Si alguien clona tu proyecto sin leer el README, ¿qué pasaría? ¿Cómo lo prevenirías?
- ¿Por qué crees que CMake es útil para proyectos con múltiples lenguajes?
- ¿Cuándo elegirías `venv` vs `conda`? ¿Hay situaciones donde uno es claramente mejor?

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- Configuración de entornos reproducibles con venv/conda
- Clonación de repositorios con tags específicos para versiones exactas
- Compilación de librerías C++ (XGrammar) usando CMake
- Creación de requirements.txt para dependencias exactas
- Scripts de automatización para setup completo

**Siguiente paso:**
En la Lectura 2, explorarás cómo los LLMs especializados en código (CodeLlama, StarCoder) funcionan y por qué son mejores que modelos generales para generación de código. Con tu entorno configurado, podrás experimentar con estos modelos directamente.
```

## Recursos Útiles

- [Git tagging official docs](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [Conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [nanobind documentation](https://nanobind.readthedocs.io/)

## Conexión con el Proyecto

```{admonition} Objetivo del curso
:class: tip

Este setup te permite usar XGrammar para **constrained decoding**: controlar qué tokens puede generar un LLM para producir código Triton sintácticamente válido.

Pipeline completo:
1. Defines gramática/schema JSON → 2. XGrammar compila a FSM → 3. FSM restringe tokens del LLM → 4. Output siempre válido
```

En el contexto del proyecto KernelAgent, XGrammar es la herramienta crítica que nos permite:

- **Garantizar sintaxis válida**: El código Triton generado por el LLM siempre será sintácticamente correcto
- **Reducir alucinaciones**: Al restringir los tokens válidos, evitamos que el modelo genere construcciones inválidas
- **Acelerar iteraciones**: No necesitamos validar/corregir el código generado, ahorrando ciclos de desarrollo

El flujo completo del proyecto es:
1. Instalas XGrammar (esta lectura)
2. Defines gramáticas JSON para kernels Triton (lecturas 4-7)
3. XGrammar compila las gramáticas a máquinas de estados finitos
4. Durante generación, la FSM restringe qué tokens puede producir el LLM
5. El resultado es código Triton válido garantizado

Sin un setup correcto, nada de esto funciona. La reproducibilidad es esencial porque:
- Diferentes versiones de XGrammar pueden compilar gramáticas de forma distinta
- Los bindings Python dependen de la versión exacta de C++
- El proyecto debe funcionar igual en tu máquina, en Colab, y en producción

## Ejercicio Práctico: Verificar Instalación

```{code-cell} ipython3
# Verificar que las dependencias básicas funcionan
import json

# Simular validación de schema (sin XGrammar real por ahora)
schema = {
    "type": "object",
    "properties": {
        "kernel_name": {"type": "string", "pattern": "^kernel_[a-z]+$"},
        "block_size": {"type": "integer", "minimum": 32, "maximum": 1024}
    },
    "required": ["kernel_name", "block_size"]
}

# Ejemplos válidos e inválidos
valid_kernel = {"kernel_name": "kernel_add", "block_size": 256}
invalid_kernel = {"kernel_name": "AddKernel", "block_size": 16}  # nombre mal, size muy pequeño

def validate_simple(data, schema):
    """Validación simplificada de schema."""
    errors = []
    for prop, rules in schema.get("properties", {}).items():
        if prop in schema.get("required", []) and prop not in data:
            errors.append(f"Falta campo requerido: {prop}")
        elif prop in data:
            if rules.get("type") == "integer":
                if "minimum" in rules and data[prop] < rules["minimum"]:
                    errors.append(f"{prop} < mínimo ({rules['minimum']})")
                if "maximum" in rules and data[prop] > rules["maximum"]:
                    errors.append(f"{prop} > máximo ({rules['maximum']})")
            if rules.get("type") == "string" and "pattern" in rules:
                import re
                if not re.match(rules["pattern"], data[prop]):
                    errors.append(f"{prop} no cumple patrón {rules['pattern']}")
    return errors

print("Kernel válido:", validate_simple(valid_kernel, schema) or "OK")
print("Kernel inválido:", validate_simple(invalid_kernel, schema))

# Output esperado:
# Kernel válido: OK
# Kernel inválido: ['kernel_name no cumple patrón ^kernel_[a-z]+$', 'block_size < mínimo (32)']
```

Este ejercicio demuestra el concepto básico de validación que XGrammar hace a nivel de tokens durante la generación. La diferencia es que XGrammar:
1. Trabaja a nivel de tokens (no texto completo)
2. Valida en tiempo real (durante generación, no después)
3. Usa máquinas de estados finitos (mucho más rápido)

**Tarea adicional**: Ejecuta este código en tu entorno para verificar que tienes Python funcionando correctamente. Si falla, revisa tu instalación de Python y el entorno virtual.

---

## 🔌 CPU Fallback: Sin GPU No Hay Problema

Muchos notebooks de este módulo están optimizados para GPU, pero **todos los conceptos son ejecutables en CPU**. La siguiente tabla muestra qué cambia:

```{admonition} ¿No tienes GPU?
:class: warning
Si `torch.cuda.is_available()` regresa `False`, usa la columna **CPU Fallback** de la tabla. El código funcionará más lento pero producirá los **mismos resultados**.

Opciones de GPU gratuita: [Google Colab](https://colab.research.google.com) (T4), [Kaggle](https://www.kaggle.com) (T4 × 30 h/semana), [Lightning.ai](https://lightning.ai) (A10G).
```

| Operación | GPU (recomendado) | CPU Fallback |
|-----------|-------------------|--------------|
| Cargar LLM (7B params) | ~15 seg | ~3 min (cuantización 4-bit recomendada) |
| Generación de 100 tokens | <1 seg | ~30 seg |
| Compilar gramática XGrammar | ~2 seg | ~2 seg (igual, es CPU-bound) |
| Ejecutar kernel Triton | Sí | ❌ Usar `torch.matmul` como referencia |

```{code-cell} ipython3
import torch

# Detección automática de hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo disponible: {device.upper()}")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name}")
    print(f"  VRAM: {gpu_mem_gb:.1f} GB")
    print(f"  Apto para: modelos de hasta ~{int(gpu_mem_gb * 0.7)//7}B params (FP16)")
else:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"  RAM disponible: {ram_gb:.1f} GB")
    print(f"  💡 Opciones para CPU:")
    print(f"     • Modelos pequeños: Phi-3-mini (3.8B) o Qwen2.5-1.5B")
    print(f"     • Activa cuantización: load_in_4bit=True")
    print(f"     • Prueba primero XGrammar standalone sin LLM")

# Convención: pasa siempre `device` a tus tensores
tensor_test = torch.zeros(4).to(device)
print(f"\nTensor de prueba en {tensor_test.device}: OK")
```

---

## 🎮 Simulaciones Interactivas del Módulo

Este módulo incluye 4 simulaciones interactivas basadas en Plotly. Todas funcionan en Colab y GitHub Pages sin GPU.

```{admonition} 📊 Ver todas las simulaciones
:class: seealso
**[SIMULACIONES_README](SIMULACIONES_README.md)** — Descripción detallada de cada visualización:
1. **Prompt Engineering** (L03) — Anatomía visual de un prompt efectivo con 5 componentes coloreados
2. **XGrammar Token Masking** (L04) — Dropdown que muestra qué tokens permite cada estado de la FSM
3. **JSON Schema Tree** (L05) — Árbol jerárquico de la estructura de un kernel Triton
4. **Token Economics** (L09) — Calculadora comparativa de costos de 5 modelos LLM en producción

Requisitos: solo `plotly` y `numpy` — sin GPU necesaria.
```

---

- Python. [Virtual Environments](https://docs.python.org/3/library/venv.html). Python Docs.
- Git. [Git Basics - Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging). Git Documentation.
- Conda. [User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/). Conda Docs.
