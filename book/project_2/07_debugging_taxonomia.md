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

# Debugging de Kernels GPU y Taxonomía de Errores


```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/07_debugging_taxonomia.ipynb)
```


> **Módulo:** Project 2 - GPU Computing & Kernel Optimization
> **Semana:** 7
> **Tiempo de lectura:** ~45 minutos

---

## Introducción

Los kernels GPU son notoriamente difíciles de debuggear. A diferencia de código CPU donde puedes inspeccionar paso a paso, en GPU los threads ejecutan en paralelo, los errores pueden ser silenciosos (resultados incorrectos sin crash), no determinísticos (race conditions), o difíciles de reproducir.

Esta lectura te enseña a **pensar como un detective de GPU** con una metodología sistemática para encontrar y clasificar errores.

---

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Clasificar errores GPU en taxonomía (compilación, runtime, correctitud, performance)
- Detectar top-5 errores: bounds, race conditions, indexación 2D, dtype, divergencia
- Aplicar metodología sistemática: reproducir → clasificar → aislar → verificar
- Diseñar tests exhaustivos (boundary, determinism, reference comparison)
- Documentar errores con template para mejorar tu generador
```

---

## Taxonomía de Errores en Kernels GPU

### Categoría 1: Errores de Compilación

```
1.1 Errores de sintaxis
    - Código Python/Triton malformado
    - Ejemplo: paréntesis desbalanceados

1.2 Errores de tipo
    - Tipos incompatibles en operaciones
    - Ejemplo: tl.dot(float32, int32)

1.3 Errores de constexpr
    - Valores no constantes donde se requiere constexpr
    - Ejemplo: BLOCK = variable_runtime

1.4 Errores de API
    - Uso incorrecto de funciones Triton
    - Ejemplo: tl.load sin pointer válido
```

### Categoría 2: Errores de Runtime

```
2.1 Out of bounds
    - Acceso a memoria fuera de límites
    - Puede crashear o dar resultados basura

2.2 Race conditions
    - Múltiples threads escriben al mismo lugar
    - Resultados no determinísticos

2.3 Deadlock (raro en Triton)
    - Sincronización incorrecta
    - Kernel nunca termina

2.4 Division by zero
    - Operación inválida
    - Puede dar inf/nan silenciosamente
```

### Categoría 3: Errores de Correctitud

```
3.1 Lógica incorrecta
    - El algoritmo no implementa la especificación
    - Ejemplo: softmax sin restar max (overflow)

3.2 Off-by-one
    - Índices incorrectos por 1
    - Ejemplo: offsets < n vs offsets <= n

3.3 Broadcasting incorrecto
    - Shapes no alineados correctamente

3.4 Reducción incompleta
    - No procesar todos los elementos
```

### Categoría 4: Errores de Performance

```
4.1 Memory coalescing violado
    - Accesos no coalescidos a memoria

4.2 Bank conflicts
    - Accesos conflictivos a shared memory

4.3 Occupancy bajo
    - Muchos registros o shared memory

4.4 Trabajo redundante
    - Recalcular valores innecesariamente
```

```{code-cell} ipython3
# Quiz de Debugging - Identificar el bug
import plotly.graph_objects as go

bugs = [
    {"code": "output[idx] = input[idx * stride]", "bug": "Uncoalesced Access", "color": "#FF6B6B"},
    {"code": "if tid % 2 == 0: ... else: ...", "bug": "Warp Divergence", "color": "#4ECDC4"},
    {"code": "shared[tid % 32]", "bug": "Bank Conflict", "color": "#45B7D1"},
    {"code": "output[tid] = input[tid]  # sin mask", "bug": "Out of Bounds", "color": "#FFE66D"},
]

fig = go.Figure()

for i, b in enumerate(bugs):
    fig.add_trace(go.Table(
        header=dict(values=['Código', 'Tipo de Bug'],
                    fill_color=b['color'], font=dict(color='white', size=12)),
        cells=dict(values=[[b['code']], [b['bug']]],
                   fill_color='lavender', font=dict(size=11)),
        domain=dict(x=[0, 1], y=[1-0.25*(i+1), 1-0.25*i])
    ))

fig.update_layout(title="Taxonomía de Bugs en Kernels GPU", height=400)
fig.show()
```

:::{figure} diagrams/error_classification_tree.png
:name: fig-error-taxonomy
:alt: Árbol de clasificación de errores GPU: Compilación, Runtime, Correctitud, y Performance
:align: center
:width: 90%

**Figura 1:** Taxonomía de Errores GPU - Clasificación jerárquica que ayuda a identificar el tipo de error y su ubicación en el pipeline de ejecución.
:::

---

## Errores Comunes y Cómo Detectarlos

### Error 1: Acceso Fuera de Límites

**Síntomas:**
- Resultados incorrectos en últimos elementos
- Segmentation fault
- NaN inesperado

**Causa:**
```{code-cell} ipython3
:tags: [skip-execution]

# Malo: no verificar límites
@triton.jit
def kernel_malo(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # ¿Qué si offsets >= n?
```

**Diagnóstico:**
```{code-cell} ipython3
:tags: [skip-execution]

def test_bounds():
    # Probar con tamaños no divisibles por BLOCK_SIZE
    n = 1000
    BLOCK_SIZE = 256
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    kernel[grid](x, y, n)
    # ¿Son los últimos elementos correctos?
```

**Solución:**
```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_bien(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

### Error 2: Race Conditions

**Síntomas:**
- Resultados corruptos cuando múltiples threads escriben
- Resultados no determinísticos

**Diagnóstico:**
```{code-cell} ipython3
:tags: [skip-execution]

def test_determinismo():
    resultados = []
    for _ in range(10):
        y = kernel(x)
        resultados.append(y.clone())

    iguales = all(torch.allclose(r, resultados[0]) for r in resultados[1:])
    if not iguales:
        print("✗ Race condition probable")
```

**Solución:**
```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_bien(x_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    suma_bloque = tl.sum(x)
    # Usar atomic para acumular
    tl.atomic_add(result_ptr, suma_bloque)
```

### Error 3: Indexación 2D Incorrecta

**Síntomas:**
- Resultados transpuestos o permutados
- Patrón incorrecto visible

**Causa:**
```{code-cell} ipython3
:tags: [skip-execution]

# Común: confundir row-major vs column-major
# PyTorch: row-major (C-contiguous)

# Malo
idx = pid_h + pid_w * height  # column-major
# Correcto
idx = pid_h * width + pid_w   # row-major
```

**Diagnóstico:**
```{code-cell} ipython3
:tags: [skip-execution]

def test_indexacion_2d():
    matrix = torch.arange(50).reshape(10, 5).float()
    # matrix[3, 2] = 3*5 + 2 = 17
    out = kernel(matrix)
    assert out.flatten()[17] == expected_transform(17)
```

**Solución:**
```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_bien(matrix_ptr, out_ptr, height, width,
                BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    h_idx = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    w_idx = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    h_2d = h_idx[:, None]
    w_2d = w_idx[None, :]
    linear_idx = h_2d * width + w_2d  # row-major
    mask = (h_2d < height) & (w_2d < width)
    data = tl.load(matrix_ptr + linear_idx, mask=mask, other=0.0)
    tl.store(out_ptr + linear_idx, data, mask=mask)
```

### Error 4: Tipo de Datos Incorrecto

**Síntomas:**
- Pérdida de precisión
- Overflow/underflow
- Resultados totalmente incorrectos

**Diagnóstico:**
```{code-cell} ipython3
:tags: [skip-execution]

def test_dtype():
    x_float = torch.randn(1000, device='cuda', dtype=torch.float32)
    y = kernel(x_float)
    print(f"Input: min={x_float.min()}, max={x_float.max()}")
    print(f"Output: min={y.min()}, max={y.max()}")
    # Si output está fuera del rango razonable → dtype incorrecto
```

### Error 5: Divergencia de Warp

**Síntomas:**
- Kernel mucho más lento de lo esperado

**Solución:**
```{code-cell} ipython3
:tags: [skip-execution]

# Usar máscaras en lugar de if
@triton.jit
def kernel_bien(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    condicion = offsets % 2 == 0
    y = tl.where(condicion, x * 2, x * 3)  # Sin divergencia
    tl.store(y_ptr + offsets, y, mask=mask)
```

---

## Metodología de Debugging

### Paso 1: Reproducir el Error

```{code-cell} ipython3
:tags: [skip-execution]

def create_minimal_reproduction(kernel, failing_input):
    """Reduce el caso de fallo al mínimo necesario."""
    sizes = [8, 16, 32, 64, 128, 256]

    for size in sizes:
        small_input = failing_input[:size]
        try:
            output = run_kernel(kernel, small_input)
            expected = reference(small_input)
            if not torch.allclose(output, expected):
                return small_input  # Encontramos caso mínimo
        except Exception:
            return small_input

    return failing_input
```

### Paso 2: Clasificar el Error

```{code-cell} ipython3
:tags: [skip-execution]

def classify_error(kernel, failing_input):
    # Intenta compilar
    try:
        compile_kernel(kernel)
    except SyntaxError:
        return "COMPILATION_SYNTAX"
    except TypeError:
        return "COMPILATION_TYPE"

    # Intenta ejecutar
    try:
        output = run_kernel(kernel, failing_input)
    except RuntimeError as e:
        if "out of bounds" in str(e):
            return "RUNTIME_OOB"
        return "RUNTIME_OTHER"

    # Verifica correctitud
    expected = reference(failing_input)
    if not torch.allclose(output, expected):
        if torch.isnan(output).any():
            return "CORRECTNESS_NAN"
        return "CORRECTNESS_LOGIC"

    return "NONE"
```

### Paso 3: Aislar la Causa

```{code-cell} ipython3
:tags: [skip-execution]

def isolate_cause(kernel_code, error_type):
    lines = kernel_code.split('\n')

    if error_type == "RUNTIME_OOB":
        for i, line in enumerate(lines):
            if 'tl.load' in line and 'mask=' not in line:
                return f"Línea {i}: load sin máscara"
            if 'tl.store' in line and 'mask=' not in line:
                return f"Línea {i}: store sin máscara"

    return "Causa no identificada automáticamente"
```

### Paso 4: Verificar Corrección Sistemática

```{code-cell} ipython3
:tags: [skip-execution]

def test_kernel_correctness(kernel_fn, torch_fn):
    test_cases = [(10,), (256,), (257,), (1000,), (1000001,)]

    for size in test_cases:
        print(f"Probando tamaño {size}...", end=" ")
        x = torch.randn(*size, device='cuda')
        y_kernel = kernel_fn(x)
        y_torch = torch_fn(x)

        rel_error = (y_kernel - y_torch).abs().max() / (y_torch.abs().max() + 1e-8)

        if rel_error < 1e-5:
            print("✓")
        else:
            print(f"✗ Error: {rel_error}")
            mask = ~torch.isclose(y_kernel, y_torch, rtol=1e-5)
            print(f"  Errores en: {torch.nonzero(mask)[:5].tolist()}")
```

---

## Técnicas de Debugging para GPU

### Technique 1: Reference Testing

```{code-cell} ipython3
:tags: [skip-execution]

def reference_test(kernel, inputs, reference_fn):
    output = run_kernel(kernel, inputs)
    expected = reference_fn(inputs)

    if not torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
        diff = (output - expected).abs()
        print(f"Max diff: {diff.max().item()}")
        print(f"Mean diff: {diff.mean().item()}")

        wrong_indices = torch.where(diff > 1e-3)
        for idx in zip(*[i[:5] for i in wrong_indices]):
            print(f"  [{idx}]: got {output[idx]}, expected {expected[idx]}")
```

### Technique 2: Boundary Testing

```{code-cell} ipython3
:tags: [skip-execution]

def boundary_test(kernel):
    test_cases = [
        {"name": "size_1", "shape": (1,)},
        {"name": "size_31", "shape": (31,)},   # < warp
        {"name": "size_32", "shape": (32,)},   # = warp
        {"name": "size_33", "shape": (33,)},   # > warp
        {"name": "all_zeros", "data": torch.zeros(256)},
        {"name": "very_large", "data": torch.full((256,), 1e30)},
        {"name": "very_small", "data": torch.full((256,), 1e-30)},
    ]

    for tc in test_cases:
        try:
            inputs = tc.get("data", torch.randn(tc["shape"])).cuda()
            output = run_kernel(kernel, inputs)
            expected = reference(inputs)
            print(f"{tc['name']}: {'✓' if torch.allclose(output, expected) else '✗'}")
        except Exception as e:
            print(f"{tc['name']}: ✗ ({e})")
```

### Technique 3: Determinism Testing

```{code-cell} ipython3
:tags: [skip-execution]

def determinism_test(kernel, inputs, num_runs=10):
    outputs = [run_kernel(kernel, inputs).clone() for _ in range(num_runs)]
    reference = outputs[0]

    for i, out in enumerate(outputs[1:], 1):
        if not torch.equal(out, reference):
            diff = (out - reference).abs()
            print(f"Run {i} differs: max diff = {diff.max().item()}")
            print("WARNING: Non-deterministic behavior - possible race condition!")
            return False

    return True
```

---

## Test Suite para Kernels Generados

```{code-cell} ipython3
:tags: [skip-execution]

class KernelTestSuite:
    def __init__(self, kernel, reference_fn):
        self.kernel = kernel
        self.reference = reference_fn
        self.results = []

    def run_all(self):
        self.test_compilation()
        self.test_basic_correctness()
        self.test_boundaries()
        self.test_determinism()
        return self.summarize()

    def test_compilation(self):
        try:
            x = torch.randn(128, device='cuda')
            _ = self.kernel(x)
            self.results.append(("compilation", True, None))
        except Exception as e:
            self.results.append(("compilation", False, str(e)))

    def test_basic_correctness(self):
        for shape in [(128,), (256, 256), (32, 64, 128)]:
            x = torch.randn(shape, device='cuda')
            try:
                output = self.kernel(x)
                expected = self.reference(x)
                correct = torch.allclose(output, expected, rtol=1e-4)
                self.results.append((f"correctness_{shape}", correct, None))
            except Exception as e:
                self.results.append((f"correctness_{shape}", False, str(e)))

    def summarize(self):
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        failures = [(n, e) for n, p, e in self.results if not p]
        return {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0,
            "failures": failures
        }
```

---

## Documentación de Errores

### Error Report Template

```markdown
## Error Report: [ID]

### Descripción
[Breve descripción del error]

### Clasificación
- **Categoría:** [Compilación/Runtime/Correctitud/Performance]
- **Subcategoría:** [e.g., Out of bounds]
- **Severidad:** [Critical/High/Medium/Low]

### Reproducción
```{code-cell} ipython3
:tags: [skip-execution]

x = torch.randn(256, device='cuda')
output = kernel(x)  # Falla aquí
```

### Análisis
- **Causa raíz:** [Explicación]
- **Línea problemática:** [Código]

### Corrección
```{code-cell} ipython3
:tags: [skip-execution]

# Antes
x = tl.load(x_ptr + offsets)

# Después
x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
```
```

---

## Checklist de Debugging

```
[ ] Compila sin errores de sintaxis
[ ] Ejecuta sin crash CUDA
[ ] Verificar pequeños casos (n=10, n=256, n=257)
[ ] Verificar casos grandes (n=1000000)
[ ] Verificar tipos de datos
[ ] Verificar límites y máscaras
[ ] Comparar contra PyTorch
[ ] Verificar determinismo
[ ] Medir rendimiento
[ ] Identificar bottleneck (memory/compute)
```

---

## Resumen

```{admonition} Resumen
:class: important
**Taxonomía de errores GPU (4 categorías):**

1. **Compilación**: Sintaxis, tipos, API usage incorrecta
2. **Runtime**: Out-of-bounds, race conditions, división por zero
3. **Correctitud**: Lógica incorrecta, off-by-one, broadcasting malo
4. **Performance**: No-coalescing, bank conflicts, ocupancia baja

**Top-5 errores más comunes:**
1. **Falta máscara en load/store** → Out-of-bounds cuando n no divide BLOCK_SIZE
2. **Indexación 2D incorrecta** → Usar column-major en vez de row-major
3. **Race conditions** → Múltiples threads escriben sin atomic
4. **No sincronizar antes de medir** → Tiempos incorrectos
5. **Divergencia de warp** → Branches con `threadIdx % algo`

**Checklist de debugging:**
- [ ] Compila sin errores
- [ ] Ejecuta sin crash (probar n=10, 256, 257, 1000000)
- [ ] Resultados correctos vs PyTorch (rtol < 1e-5)
- [ ] Determinista (10 runs dan mismo resultado)
- [ ] Performance razonable (dentro de 10x de baseline esperado)
```

```{admonition} ⚠️ Antipatrón: Debugging Sin Metodología
:class: warning
**Malo**: "Cambio random hasta que funcione"
- Pruebas sin sistemática
- No documentas qué intentaste
- Mismo error reaparece después

**Bueno**: Metodología sistemática
1. **Reproducir**: Encontrar caso mínimo que falla
2. **Clasificar**: ¿Compilación? ¿Runtime? ¿Correctitud?
3. **Aislar**: Identificar línea/causa exacta
4. **Verificar**: Test que confirma el fix
5. **Documentar**: Registrar para evitar repetir
```

```{admonition} 📊 Cómo verificar corrección
:class: tip
**Suite de tests mínima:**

```python
# Test 1: Boundary cases
sizes = [1, 31, 32, 33, 255, 256, 257, 1000000]
for n in sizes:
    test_kernel(n)

# Test 2: Determinismo (race conditions)
for _ in range(10):
    result = kernel(x)
    assert torch.equal(result, first_result)

# Test 3: Reference comparison
expected = torch_reference(x)
actual = kernel(x)
assert torch.allclose(actual, expected, rtol=1e-5)
```

Si pasan los 3 tests → 95% probabilidad de que el kernel es correcto.
```

```{admonition} 🎯 En tu proyecto
:class: note
Cuando tu generador produzca kernels incorrectos:
1. Clasifica el error según taxonomía
2. Agrega constraint a la gramática para prevenirlo
3. Documenta en error log con ejemplo
4. Mejora prompts/templates con el patrón correcto

La taxonomía te ayuda a entender QUÉ falla y DÓNDE mejorar la generación.
```

---

## Ejercicios

### Ejercicio 1: Detectar el Error

```{code-cell} ipython3
:tags: [skip-execution]

@triton.jit
def kernel_buggy(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # ¿Error?
    y = tl.where(offsets >= n, x, 0.0)  # ¿Error?
    tl.store(y_ptr + offsets, y)  # ¿Error?
```

Identifica todos los problemas.

### Ejercicio 2: Clasificación

Para cada caso, identifica la categoría de error:
1. Load sin máscara
2. Softmax sin restar max
3. Índices column-major en tensor row-major

### Para Pensar

> *Si un error solo aparece con N=1000 pero no N=1024, ¿qué indica sobre la causa?*

---

## Errores Comunes

```{admonition} Errores frecuentes durante debugging
:class: warning

1. **No hacer warmup antes de medir**: Primera ejecución incluye JIT compilation y es mucho más lenta.
2. **No sincronizar GPU**: `torch.cuda.synchronize()` es crítico para timing correcto.
3. **Asumir determinismo**: Race conditions pueden dar resultados diferentes en cada ejecución.
4. **No probar casos edge**: Tamaños como n=1, n=31, n=32, n=33 exponen bugs de límites.
```

## Ejercicio Práctico: Bug Hunting con Múltiples Errores

```{code-cell} ipython3
import torch
import triton
import triton.language as tl

# Kernel con múltiples tipos de bugs
@triton.jit
def multi_bug_kernel(x_ptr, y_ptr, threshold, n, BLOCK_SIZE: tl.constexpr):
    """Suma elementos > threshold. Tiene 5 bugs diferentes."""
    pid = tl.program_id(axis=0)

    # BUG 1: Indexación incorrecta
    offsets = pid + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # BUG 2: No verificar límites
    x = tl.load(x_ptr + offsets)

    # BUG 3: Lógica invertida
    filtered = tl.where(x < threshold, x, 0.0)

    # BUG 4: Reducción sin máscara
    suma = tl.sum(filtered)

    # BUG 5: Race condition - múltiples bloques escriben al mismo lugar
    tl.store(y_ptr, suma)

def debug_exercise():
    """Ejercicio guiado para encontrar todos los bugs."""
    n = 100
    BLOCK_SIZE = 32
    threshold = 0.5

    x = torch.rand(n, device='cuda')  # Valores entre 0 y 1
    y = torch.zeros(1, device='cuda')

    # Referencia correcta
    expected = x[x > threshold].sum()

    print("=== Bug Hunting Exercise ===\n")
    print(f"Input: {n} elementos")
    print(f"Threshold: {threshold}")
    print(f"Elementos > threshold: {(x > threshold).sum().item()}")
    print(f"Suma esperada: {expected.item():.4f}\n")

    # Ejecutar kernel buggy
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    print(f"Lanzando {num_blocks} bloques...")

    try:
        multi_bug_kernel[(num_blocks,)](x, y, threshold, n, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        result = y[0].item()
        print(f"Resultado del kernel: {result:.4f}")
        print(f"Error absoluto: {abs(result - expected.item()):.4f}")
    except Exception as e:
        print(f"✗ Kernel crashed: {e}")

    # Análisis de bugs
    print("\n=== Análisis de Bugs ===\n")

    print("BUG 1: INDEXACIÓN INCORRECTA")
    print("  Problema: offsets = pid + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
    print("  Para pid=0: offsets = [32, 33, ..., 63] ¡Ignora primeros 32 elementos!")
    print("  Fix: offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
    print()

    print("BUG 2: NO VERIFICAR LÍMITES")
    print("  Problema: x = tl.load(x_ptr + offsets) sin máscara")
    print("  Para n=100, BLOCK_SIZE=32, último bloque accede [96-127] pero solo [96-99] existen")
    print("  Fix: mask = offsets < n; x = tl.load(..., mask=mask, other=0.0)")
    print()

    print("BUG 3: LÓGICA INVERTIDA")
    print("  Problema: filtered = tl.where(x < threshold, x, 0.0)")
    print("  Queremos x > threshold, no x < threshold")
    print("  Fix: filtered = tl.where(x > threshold, x, 0.0)")
    print()

    print("BUG 4: REDUCCIÓN SIN MÁSCARA")
    print("  Problema: suma = tl.sum(filtered) incluye valores fuera de límites")
    print("  Si no usamos máscara en load, valores basura afectan suma")
    print("  Fix: Usar máscara en load o aplicar máscara antes de reducir")
    print()

    print("BUG 5: RACE CONDITION")
    print("  Problema: tl.store(y_ptr, suma) - todos los bloques escriben a y_ptr[0]")
    print("  Último bloque gana, perdemos sumas de otros bloques")
    print("  Fix: tl.atomic_add(y_ptr, suma)")
    print()

    # Mostrar kernel corregido
    print("=== Kernel Corregido ===\n")
    print("""
@triton.jit
def fixed_kernel(x_ptr, y_ptr, threshold, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # FIX 1: Indexación correcta
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # FIX 2: Verificar límites
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # FIX 3: Lógica correcta
    filtered = tl.where(x > threshold, x, 0.0)

    # FIX 4: Reducción con datos válidos (máscara ya aplicada en load)
    suma = tl.sum(filtered)

    # FIX 5: Atomic para evitar race condition
    tl.atomic_add(y_ptr, suma)
    """)

# Ejecutar ejercicio
debug_exercise()

print("\n💡 Lección: Los bugs de GPU raramente vienen solos. Usa checklist sistemático.")
```

---

*Esta lectura es parte del curso "Grammar-Constrained GPU Kernel Generation" - ACA*
