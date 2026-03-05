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

# Lectura 7: Serving y Optimización

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/08_serving_llms.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton xgrammar
print('Dependencies installed!')
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Explicar cómo KV-Cache acelera generación 10-100x y calcular su costo de memoria
- Comparar frameworks de serving (vLLM, TensorRT-LLM, SGLang) y sus características
- Implementar continuous batching para maximizar throughput con múltiples usuarios
- Aplicar speculative decoding para aceleración 2-4x con modelos draft
- Evaluar trade-offs de cuantización (INT8/INT4) entre tamaño, velocidad y calidad
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Desplegar tu modelo KernelAgent en producción eligiendo el framework de serving adecuado, configurando KV-Cache óptimo, aplicando cuantización si es necesario, y usando continuous batching para manejar múltiples solicitudes concurrentes eficientemente.
```

## Introducción

Generaste un modelo excelente. ¿Ahora qué? Necesitas servir (deploy) millones de tokens por segundo a usuarios reales. Un modelo de 7B parámetros tarda 2 segundos por token en GPU estándar. Con millones de usuarios, esto es insostenible.

Esta lectura trata sobre **optimización de inference**: cómo servir LLMs rápidamente y a bajo costo.

:::{figure} diagrams/serving_architecture.png
:name: fig-serving-architecture
:alt: Arquitectura completa de serving de LLMs con varias optimizaciones
:align: center
:width: 90%

**Figura 6:** Arquitectura de serving con KV-Cache, continuous batching, cuantización y speculative decoding.
:::

---

## Parte 1: El Cuello de Botella - KV-Cache

### El Problema

```
Generación autoregresiva:
Paso 1: Entrada "¿Hola?" → Genera "Hola"
  Computa: atención sobre 1 token (entrada) + 1 token (salida)

Paso 2: Entrada "¿Hola? Hola" → Genera ","
  Computa: atención sobre 2 tokens (entrada) + 2 tokens (salida)
  ← Repite cómputo del token 1 INNECESARIAMENTE

Paso 3: Entrada "¿Hola? Hola," → Genera "¿"
  Computa: atención sobre 3 tokens (entrada) + 3 tokens (salida)
  ← Repite cómputación de tokens 1, 2 INNECESARIAMENTE
```

**Problema:** Conforme generas más tokens, recomputa la atención sobre tokens previos (cada paso tarda más).

### La Solución: KV-Cache

En lugar de recomputar, **almacena** Keys y Values de pasos anteriores:

```
Paso 1: Genera "Hola"
  Computa atención, almacena KV del token 1
  KV-Cache = {token_1: (K_1, V_1)}

Paso 2: Genera ","
  NO recomputa token 1
  Usa KV-Cache para token 1
  Computa atención solo para token 2
  KV-Cache = {token_1: (K_1, V_1), token_2: (K_2, V_2)}

Paso 3: Genera "¿"
  Usa KV-Cache para tokens 1, 2
  Computa atención solo para token 3
  KV-Cache = {..., token_3: (K_3, V_3)}
```

**Efecto:** Cada paso genera 1 token (en lugar de N tokens), velocidad casi 10-100x

### Costo de KV-Cache

```{code-cell} ipython3
# Calculadora de KV-Cache para modelos LLM
import matplotlib.pyplot as plt
import numpy as np

class KVCacheCalculator:
    """Calcula el tamaño de KV-Cache para diferentes configuraciones"""

    def __init__(self, num_layers, hidden_dim, num_heads, bytes_per_param=2):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.bytes_per_param = bytes_per_param  # fp16 = 2 bytes

    def cache_per_token(self):
        """Calcula bytes de KV-Cache por token"""
        # Keys y Values, ambos del tamaño hidden_dim
        kv_size = 2 * self.hidden_dim * self.bytes_per_param
        # Multiplicado por número de capas
        total_size = self.num_layers * kv_size
        return total_size

    def cache_for_sequence(self, seq_length):
        """Calcula KV-Cache para una secuencia completa"""
        return self.cache_per_token() * seq_length

    def cache_for_batch(self, seq_length, batch_size):
        """Calcula KV-Cache para un batch de secuencias"""
        return self.cache_for_sequence(seq_length) * batch_size

    def format_size(self, bytes_size):
        """Formatea tamaño en bytes a unidades legibles"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"

# Ejemplo: Llama 2 7B
llama_7b = KVCacheCalculator(
    num_layers=32,
    hidden_dim=4096,
    num_heads=32
)

print("Análisis de KV-Cache para Llama 2 7B")
print("="*60)
print(f"Configuración:")
print(f"  - Capas: {llama_7b.num_layers}")
print(f"  - Hidden dim: {llama_7b.hidden_dim}")
print(f"  - Precisión: fp16 ({llama_7b.bytes_per_param} bytes/param)")
print()

# Por token
per_token = llama_7b.cache_per_token()
print(f"KV-Cache por token: {llama_7b.format_size(per_token)}")

# Por secuencia
seq_length = 4096
per_seq = llama_7b.cache_for_sequence(seq_length)
print(f"KV-Cache para {seq_length} tokens: {llama_7b.format_size(per_seq)}")

# Para múltiples usuarios
batch_sizes = [1, 16, 64, 256]
print(f"\nKV-Cache para diferentes batch sizes (seq_len={seq_length}):")
for bs in batch_sizes:
    total = llama_7b.cache_for_batch(seq_length, bs)
    print(f"  {bs:3d} secuencias: {llama_7b.format_size(total)}")

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Cache vs Longitud de secuencia
seq_lengths = np.array([512, 1024, 2048, 4096, 8192])
cache_sizes = [llama_7b.cache_for_sequence(sl) / (1024**3) for sl in seq_lengths]  # GB

ax1.plot(seq_lengths, cache_sizes, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Longitud de Secuencia (tokens)', fontsize=11)
ax1.set_ylabel('KV-Cache (GB)', fontsize=11)
ax1.set_title('KV-Cache vs Longitud de Secuencia\n(1 secuencia)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Gráfico 2: Cache vs Batch size (para seq_len=4096)
batch_range = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
cache_batch = [llama_7b.cache_for_batch(4096, bs) / (1024**3) for bs in batch_range]

ax2.plot(batch_range, cache_batch, marker='s', linewidth=2, markersize=8, color='orange')
ax2.axhline(y=24, color='red', linestyle='--', label='24GB GPU (RTX 4090)')
ax2.axhline(y=80, color='green', linestyle='--', label='80GB GPU (A100)')
ax2.set_xlabel('Batch Size (secuencias concurrentes)', fontsize=11)
ax2.set_ylabel('KV-Cache Total (GB)', fontsize=11)
ax2.set_title('KV-Cache vs Batch Size\n(seq_len=4096)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

plt.tight_layout()
plt.show()

print("\nTrade-off: KV-Cache es ESENCIAL para velocidad pero consume MUCHA memoria")
```

**Trade-off:** KV-Cache es **esencial** para velocidad pero consume **mucha memoria**.

---

## Parte 2: Frameworks de Serving

### vLLM

**La mayoría popular para LLMs:**

```{code-cell} ipython3
:tags: [skip-execution]

# Ejemplo de uso de vLLM (requiere instalación: pip install vllm)
# Este código muestra la API pero requiere GPU para ejecutar

from vllm import LLM, SamplingParams

# Inicializar el modelo
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # Número de GPUs
    max_model_len=4096,      # Longitud máxima de secuencia
)

# Configurar parámetros de muestreo
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generar respuestas para múltiples prompts
prompts = [
    "Hola, ¿cómo estás?",
    "Explica qué es un transformer",
    "¿Cuál es la capital de Francia?"
]

outputs = llm.generate(prompts, sampling_params)

# Mostrar resultados
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print("-" * 60)
```

```{code-cell} ipython3
# Simulación del comportamiento de vLLM (sin GPU)
import time
from dataclasses import dataclass
from typing import List

@dataclass
class GenerationOutput:
    """Simula la salida de vLLM"""
    prompt: str
    generated_text: str
    num_tokens: int
    latency_ms: float

class MockvLLM:
    """Mock de vLLM para demostración sin GPU"""

    def __init__(self, model_name, tokens_per_second=150):
        self.model_name = model_name
        self.tokens_per_second = tokens_per_second
        print(f"[Mock] Loaded model: {model_name}")
        print(f"[Mock] Throughput: {tokens_per_second} tokens/second")

    def generate(self, prompts: List[str], max_tokens=50):
        """Simula generación con continuous batching"""
        outputs = []

        print(f"\n[Mock] Processing {len(prompts)} prompts in batch...")

        for i, prompt in enumerate(prompts):
            # Simular generación
            num_tokens = min(max_tokens, 30 + (i * 5))
            latency_ms = (num_tokens / self.tokens_per_second) * 1000

            # Texto simulado
            generated = f"[Generated response to prompt {i+1} with {num_tokens} tokens]"

            output = GenerationOutput(
                prompt=prompt,
                generated_text=generated,
                num_tokens=num_tokens,
                latency_ms=latency_ms
            )
            outputs.append(output)

        return outputs

# Demostración
print("Características de vLLM:")
print("✓ KV-Cache automático y optimizado")
print("✓ Paged Attention: asigna memoria dinámicamente")
print("✓ Continuous Batching: acepta nuevas solicitudes mientras genera")
print("✓ Multitarea: múltiples usuarios simultáneamente")
print()

# Simular vLLM
llm = MockvLLM("meta-llama/Llama-2-7b", tokens_per_second=150)

prompts = [
    "Hola, ¿cómo estás?",
    "¿Qué es un transformer?",
    "Explica Python en una frase"
]

outputs = llm.generate(prompts, max_tokens=50)

print("\nResultados:")
print("="*60)
total_tokens = 0
total_time_ms = 0

for output in outputs:
    print(f"Prompt: {output.prompt[:40]}...")
    print(f"Tokens: {output.num_tokens}")
    print(f"Latencia: {output.latency_ms:.2f} ms")
    print(f"Tokens/s: {output.num_tokens / (output.latency_ms/1000):.1f}")
    print()
    total_tokens += output.num_tokens
    total_time_ms = max(total_time_ms, output.latency_ms)  # Batching paralelo

print(f"Throughput total: {total_tokens / (total_time_ms/1000):.1f} tokens/segundo")
print(f"Latencia promedio: {total_time_ms/len(outputs):.2f} ms por request")
```

### TensorRT-LLM (NVIDIA)

```
Características:
✓ Kernel CUDA optimizados (muy rápido)
✓ Soporte para arquitecturas especiales
✓ Requiere compilación (menos flexible que vLLM)

Ejemplo:
from tensorrt_llm.llm import LLM

llm = LLM("models/llama2-7b-tensorrt")
outputs = llm.generate(
    prompts=["Hola"],
    max_new_tokens=100
)

Rendimiento típico:
- Throughput: 200-1000 tokens/segundo (depende de GPU)
- Latencia: 30-100 ms por token
```

### SGLang

```
Características:
✓ Constrained decoding integrado
✓ Batching inteligente
✓ Optimizado para cadenas de solicitudes complejas

Ejemplo:
import sglang as sgl

@sgl.function
def generate_json(s):
    s += "Extrae información. Responde en JSON:\n"
    s += sgl.gen(
        "json_output",
        max_tokens=200,
        grammar="json"  # ← Constrained decoding
    )

state = generate_json.run("Información: Juan, 30 años")
```

---

---
## Siguiente Paso
En la **[Lectura 8b](08b_serving_optimization.md)**, exploraremos técnicas avanzadas: Continuous Batching, Speculative Decoding y Cuantización para maximizar el throughput y minimizar costos.
