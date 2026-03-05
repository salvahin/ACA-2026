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

# Lectura 8b: Optimización Avanzada de Serving

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/08b_serving_optimization.ipynb)
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
- Implementar continuous batching para maximizar throughput con múltiples usuarios
- Aplicar speculative decoding para aceleración 2-4x con modelos draft
- Evaluar trade-offs de cuantización (INT8/INT4) entre tamaño, velocidad y calidad
- Elegir hardware adecuado según la carga de trabajo
```


## Parte 3: Continuous Batching

### Batching Tradicional

```
Lote 1:
  Solicitud A: "¿Hola?" (120 tokens)
  Solicitud B: "¿Cómo?" (100 tokens)

Espera a ambas:
  Solicitud A tarda 120 pasos
  Solicitud B tarda 100 pasos
  Tiempo total: max(120, 100) = 120 pasos (limitado por A)

Problema: B termina, pero espera a A.
GPU ociosa mientras B espera (ineficiencia)
```

### Continuous Batching (Dinámico)

```
Paso 1:
  Solicitud A: generar 1 token
  Solicitud B: generar 1 token
  GPU: procesa ambos parallelamente

Paso 2:
  Solicitud A: generar 1 token
  Solicitud B: generar 1 token
  NUEVA solicitud C: generar 1 token ← Se agrega dinámicamente
  GPU: procesa A, B, C

Paso 100:
  Solicitud A: TERMINÓ (120 tokens generados)
  Solicitud B: generar 1 token
  Solicitud C: generar 1 token
  GPU: procesa solo B, C (memoria liberada de A)

Resultado:
  Tiempo total: 120 pasos (igual)
  Pero MUCHO más throughput porque siempre hay trabajo
```

### Impacto en Práctica

```{code-cell} ipython3
# Simulación de Continuous Batching vs Batching tradicional
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class BatchingSimulator:
    """Simula diferentes estrategias de batching"""

    def __init__(self, ms_per_token=100):
        self.ms_per_token = ms_per_token

    def traditional_batching(self, requests):
        """Batching tradicional: espera a que todas terminen"""
        max_tokens = max(req['tokens'] for req in requests)
        total_time_ms = max_tokens * self.ms_per_token
        return {
            'total_time_ms': total_time_ms,
            'total_tokens': sum(req['tokens'] for req in requests),
            'throughput': sum(req['tokens'] for req in requests) / (total_time_ms / 1000)
        }

    def continuous_batching(self, requests):
        """Continuous batching: procesa dinámicamente"""
        active_requests = deque(requests)
        total_tokens = sum(req['tokens'] for req in requests)
        max_tokens = max(req['tokens'] for req in requests)

        # Overhead pequeño por gestión dinámica
        overhead_steps = len(requests) * 0.1
        total_time_ms = (max_tokens + overhead_steps) * self.ms_per_token

        return {
            'total_time_ms': total_time_ms,
            'total_tokens': total_tokens,
            'throughput': total_tokens / (total_time_ms / 1000)
        }

# Crear simulador
simulator = BatchingSimulator(ms_per_token=100)

# Escenario: 10 usuarios con diferentes longitudes
requests = [
    {'id': i, 'tokens': np.random.randint(50, 150)}
    for i in range(10)
]

print("Escenario: 10 usuarios concurrentes")
print("="*60)
print("Requests:")
for req in requests:
    print(f"  Usuario {req['id']}: {req['tokens']} tokens")

total_tokens = sum(req['tokens'] for req in requests)
max_tokens = max(req['tokens'] for req in requests)
print(f"\nTotal tokens: {total_tokens}")
print(f"Max tokens (request más largo): {max_tokens}")

# Batching tradicional
print("\n" + "="*60)
print("BATCHING TRADICIONAL:")
trad_result = simulator.traditional_batching(requests)
print(f"  Tiempo total: {trad_result['total_time_ms']/1000:.2f} segundos")
print(f"  Throughput: {trad_result['throughput']:.1f} tokens/s")
print(f"  Problema: GPU ociosa mientras espera requests cortos")

# Continuous batching
print("\n" + "="*60)
print("CONTINUOUS BATCHING:")
cont_result = simulator.continuous_batching(requests)
print(f"  Tiempo total: {cont_result['total_time_ms']/1000:.2f} segundos")
print(f"  Throughput: {cont_result['throughput']:.1f} tokens/s")
print(f"  Beneficio: GPU siempre ocupada, acepta nuevos requests")

# Mejora
improvement = (cont_result['throughput'] / trad_result['throughput'] - 1) * 100
print(f"\nMejora de throughput: {improvement:.1f}%")

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Comparación de throughput
strategies = ['Tradicional', 'Continuous']
throughputs = [trad_result['throughput'], cont_result['throughput']]
colors = ['#ff7f0e', '#2ca02c']

ax1.bar(strategies, throughputs, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Throughput (tokens/segundo)', fontsize=11)
ax1.set_title('Comparación de Throughput', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for i, (strategy, throughput) in enumerate(zip(strategies, throughputs)):
    ax1.text(i, throughput + 2, f'{throughput:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Comparación con diferentes números de usuarios
num_users_range = [1, 5, 10, 20, 50, 100]
trad_throughputs = []
cont_throughputs = []

for num_users in num_users_range:
    reqs = [{'id': i, 'tokens': 100} for i in range(num_users)]
    trad = simulator.traditional_batching(reqs)
    cont = simulator.continuous_batching(reqs)
    trad_throughputs.append(trad['throughput'])
    cont_throughputs.append(cont['throughput'])

ax2.plot(num_users_range, trad_throughputs, marker='o', label='Tradicional', linewidth=2)
ax2.plot(num_users_range, cont_throughputs, marker='s', label='Continuous', linewidth=2)
ax2.set_xlabel('Número de Usuarios Concurrentes', fontsize=11)
ax2.set_ylabel('Throughput (tokens/segundo)', fontsize=11)
ax2.set_title('Escalabilidad: Throughput vs Usuarios', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.show()
```

---

## Parte 4: Speculative Decoding

### La Idea

En cada paso, el modelo genera 1 token. ¿Y si un modelo **más pequeño y rápido** genera k tokens especulativos?

```
Paso 1: Modelo pequeño (rápido) especula 4 tokens
  Entrada: "¿Hola?"
  Salida especulativa: "Hola , ¿ cómo"

Paso 2: Modelo grande (lento) valida
  "¿Hola? Hola , ¿ cómo"
  Computa logits para esos 4 tokens
  Verifica si coinciden con especulación

  Si todos coinciden: ✓ Aceptamos los 4
  Si difieren en posición 3: Aceptamos 2, rechazamos 2

Resultado:
  Generamos múltiples tokens en el tiempo que tardaba 1
```

### Matemáticamente

```{code-cell} ipython3
# Simulación de Speculative Decoding
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class SpeculativeDecodingSimulator:
    """Simula speculative decoding para análisis de rendimiento"""

    def __init__(self, draft_model_speed=10, target_model_speed=2,
                 acceptance_rate=0.7, speculation_length=4):
        """
        Args:
            draft_model_speed: tokens/segundo del modelo pequeño
            target_model_speed: tokens/segundo del modelo grande
            acceptance_rate: tasa de aceptación de tokens especulativos
            speculation_length: número de tokens a especular
        """
        self.draft_speed = draft_model_speed
        self.target_speed = target_model_speed
        self.acceptance_rate = acceptance_rate
        self.k = speculation_length

    def simulate_generation(self, total_tokens: int, use_speculation: bool = True):
        """Simula generación de tokens"""
        if not use_speculation:
            # Generación normal: 1 token por paso
            time_per_token = 1.0 / self.target_speed
            total_time = total_tokens * time_per_token
            return {
                'total_time': total_time,
                'tokens_generated': total_tokens,
                'throughput': total_tokens / total_time,
                'steps': total_tokens
            }
        else:
            # Speculative decoding
            total_time = 0
            tokens_generated = 0
            steps = 0

            while tokens_generated < total_tokens:
                # Tiempo para generar k tokens especulativos (modelo pequeño)
                draft_time = self.k / self.draft_speed

                # Tiempo para validar (modelo grande verifica todos a la vez)
                validation_time = 1.0 / self.target_speed

                # Tokens aceptados (en promedio)
                accepted = int(self.k * self.acceptance_rate)
                accepted = min(accepted, total_tokens - tokens_generated)

                total_time += draft_time + validation_time
                tokens_generated += max(1, accepted)  # Al menos 1 token
                steps += 1

            return {
                'total_time': total_time,
                'tokens_generated': tokens_generated,
                'throughput': tokens_generated / total_time,
                'steps': steps
            }

    def calculate_speedup(self, total_tokens: int) -> Dict:
        """Calcula la aceleración de speculative decoding"""
        baseline = self.simulate_generation(total_tokens, use_speculation=False)
        speculative = self.simulate_generation(total_tokens, use_speculation=True)

        speedup = baseline['total_time'] / speculative['total_time']
        throughput_improvement = speculative['throughput'] / baseline['throughput']

        return {
            'baseline': baseline,
            'speculative': speculative,
            'speedup': speedup,
            'throughput_improvement': throughput_improvement
        }

# Crear simulador
simulator = SpeculativeDecodingSimulator(
    draft_model_speed=10,     # modelo pequeño: 10 tokens/s
    target_model_speed=2,     # modelo grande: 2 tokens/s
    acceptance_rate=0.7,      # 70% de aceptación
    speculation_length=4      # especula 4 tokens
)

print("SPECULATIVE DECODING - ANÁLISIS DE RENDIMIENTO")
print("="*70)
print(f"Configuración:")
print(f"  Modelo draft (pequeño): {simulator.draft_speed} tokens/s")
print(f"  Modelo target (grande): {simulator.target_speed} tokens/s")
print(f"  Tokens especulativos: {simulator.k}")
print(f"  Tasa de aceptación: {simulator.acceptance_rate * 100}%")

# Analizar para 100 tokens
tokens = 100
results = simulator.calculate_speedup(tokens)

print(f"\nGeneración de {tokens} tokens:")
print("-"*70)
print(f"{'Método':<20} {'Tiempo (s)':<15} {'Throughput':<15} {'Pasos':<10}")
print("-"*70)
print(f"{'Normal':<20} {results['baseline']['total_time']:<15.2f} "
      f"{results['baseline']['throughput']:<15.2f} "
      f"{results['baseline']['steps']:<10}")
print(f"{'Speculative':<20} {results['speculative']['total_time']:<15.2f} "
      f"{results['speculative']['throughput']:<15.2f} "
      f"{results['speculative']['steps']:<10}")
print("-"*70)
print(f"Aceleración: {results['speedup']:.2f}x")
print(f"Mejora throughput: {results['throughput_improvement']:.2f}x")

# Análisis de sensibilidad
print("\n" + "="*70)
print("ANÁLISIS DE SENSIBILIDAD")
print("="*70)

# Variar tasa de aceptación
acceptance_rates = np.linspace(0.1, 1.0, 10)
speedups_acceptance = []

for rate in acceptance_rates:
    sim = SpeculativeDecodingSimulator(
        draft_model_speed=10,
        target_model_speed=2,
        acceptance_rate=rate,
        speculation_length=4
    )
    result = sim.calculate_speedup(100)
    speedups_acceptance.append(result['speedup'])

# Variar longitud de especulación
speculation_lengths = [1, 2, 4, 6, 8, 10]
speedups_length = []

for length in speculation_lengths:
    sim = SpeculativeDecodingSimulator(
        draft_model_speed=10,
        target_model_speed=2,
        acceptance_rate=0.7,
        speculation_length=length
    )
    result = sim.calculate_speedup(100)
    speedups_length.append(result['speedup'])

# Visualización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Speedup vs Acceptance Rate
ax1.plot(acceptance_rates * 100, speedups_acceptance,
         marker='o', linewidth=2.5, markersize=8, color='blue')
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sin mejora')
ax1.fill_between(acceptance_rates * 100, 1, speedups_acceptance,
                  alpha=0.3, color='blue')
ax1.set_xlabel('Tasa de Aceptación (%)', fontsize=11)
ax1.set_ylabel('Aceleración (speedup)', fontsize=11)
ax1.set_title('Speedup vs Tasa de Aceptación\n(k=4 tokens)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Speedup vs Speculation Length
ax2.plot(speculation_lengths, speedups_length,
         marker='s', linewidth=2.5, markersize=8, color='green')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sin mejora')
ax2.fill_between(speculation_lengths, 1, speedups_length,
                  alpha=0.3, color='green')
ax2.set_xlabel('Longitud de Especulación (k tokens)', fontsize=11)
ax2.set_ylabel('Aceleración (speedup)', fontsize=11)
ax2.set_title('Speedup vs Longitud de Especulación\n(70% acceptance)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nCONCLUSIONES:")
print("-"*70)
print(f"• Óptimo tasa de aceptación: >60% para speedup significativo")
print(f"• Óptimo longitud especulación: 4-6 tokens (balance overhead/beneficio)")
print(f"• Speedup típico: 2-4x en condiciones reales")
print(f"• Requiere: modelo draft rápido y compatible con target model")
```

### Compatibilidad con Constrained Decoding

```{code-cell} ipython3
# Análisis de Speculative Decoding + Constrained Decoding
class ConstrainedSpeculativeDecoding:
    """Simula speculative decoding con restricciones"""

    def __init__(self):
        self.grammar_violations = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0

    def speculate_with_constraints(self, num_tokens: int,
                                   grammar_compliance: float = 0.9):
        """
        Simula especulación con restricciones gramaticales

        Args:
            num_tokens: número de tokens a especular
            grammar_compliance: probabilidad de que draft respete gramática
        """
        results = []

        for i in range(num_tokens):
            # Draft model especula
            respects_grammar = np.random.random() < grammar_compliance

            if respects_grammar:
                # Target model valida (alta probabilidad de aceptación)
                accepted = np.random.random() < 0.85
            else:
                # Violación de gramática = rechazo automático
                accepted = False
                self.grammar_violations += 1

            if accepted:
                self.accepted_tokens += 1
                results.append({'token': i, 'status': 'accepted'})
            else:
                self.rejected_tokens += 1
                results.append({'token': i, 'status': 'rejected'})

        return results

    def get_statistics(self):
        """Retorna estadísticas de aceptación"""
        total = self.accepted_tokens + self.rejected_tokens
        if total == 0:
            return None

        return {
            'total_tokens': total,
            'accepted': self.accepted_tokens,
            'rejected': self.rejected_tokens,
            'acceptance_rate': self.accepted_tokens / total,
            'grammar_violations': self.grammar_violations,
            'violation_rate': self.grammar_violations / total
        }

print("SPECULATIVE DECODING + CONSTRAINED DECODING")
print("="*70)
print("""
PROBLEMA:
  - Speculative decoding especula libremente
  - Constrained decoding restringe según gramática
  - ¿Cómo combinarlos eficientemente?

SOLUCIÓN:
  1. Draft model respeta restricciones (genera especulativos válidos)
  2. Target model valida (considera restricciones)
  3. Ventaja: mayor tasa de aceptación

IMPLEMENTACIÓN:
""")

# Escenario 1: Draft model SIN conocimiento de gramática
print("\nEscenario 1: Draft model SIN restricciones")
print("-"*70)
decoder_no_grammar = ConstrainedSpeculativeDecoding()
results = decoder_no_grammar.speculate_with_constraints(
    num_tokens=100,
    grammar_compliance=0.5  # Solo 50% respeta gramática
)
stats = decoder_no_grammar.get_statistics()

print(f"Tasa de aceptación: {stats['acceptance_rate']*100:.1f}%")
print(f"Violaciones de gramática: {stats['grammar_violations']} "
      f"({stats['violation_rate']*100:.1f}%)")
print(f"Tokens aceptados: {stats['accepted']}")
print(f"Tokens rechazados: {stats['rejected']}")

# Escenario 2: Draft model CON conocimiento de gramática
print("\nEscenario 2: Draft model CON restricciones")
print("-"*70)
decoder_with_grammar = ConstrainedSpeculativeDecoding()
results = decoder_with_grammar.speculate_with_constraints(
    num_tokens=100,
    grammar_compliance=0.95  # 95% respeta gramática
)
stats = decoder_with_grammar.get_statistics()

print(f"Tasa de aceptación: {stats['acceptance_rate']*100:.1f}%")
print(f"Violaciones de gramática: {stats['grammar_violations']} "
      f"({stats['violation_rate']*100:.1f}%)")
print(f"Tokens aceptados: {stats['accepted']}")
print(f"Tokens rechazados: {stats['rejected']}")

# Comparación
print("\n" + "="*70)
print("CONCLUSIÓN:")
print("-"*70)
print("""
✓ Draft model CON restricciones gramaticales:
  - Mejor tasa de aceptación (~80% vs ~40%)
  - Menos desperdicio computacional
  - Speedup efectivo más alto

✗ Draft model SIN restricciones:
  - Baja tasa de aceptación
  - Muchas especulaciones inválidas
  - Beneficio marginal

RECOMENDACIÓN:
  Entrenar draft model pequeño que respete las mismas restricciones
  que el target model. Esto maximiza la tasa de aceptación y el speedup.
""")
```

---

## Parte 5: Cuantización - Comprimiendo Pesos

### El Problema

Modelo Llama 2 7B con float32:
```
7 * 10^9 parámetros * 4 bytes = 28 GB
Imposible en GPU consumer (típicamente 24 GB)
```

### Soluciones de Cuantización

#### INT8

```
Original (float32): valor = 0.0234567
Cuantizado (INT8): valor ≈ 6 (entero entre -128 y 127)

Proceso:
  1. Escala: max_val = max(|valores|)
  2. Quantize: int8_val = round(float_val / max_val * 127)
  3. Dequantize: float_val = int8_val / 127 * max_val

Tamaño: 7B * 1 byte = 7 GB ✓ (cabe en GPU)
Pérdida de precisión: ~2-3% en calidad
```

#### INT4

```
Similar a INT8 pero con 4 bits (16 valores posibles)
Tamaño: 7B * 0.5 bytes = 3.5 GB
Pérdida de precisión: ~5-7% en calidad
Más rápido que INT8 (menos memoria bandwidth)
```

#### Métodos Avanzados: GPTQ y AWQ

```
GPTQ (Gradient-based Post-Training Quantization):
  - Cuantiza por grupo de pesos
  - Usa información de Hessian para ubicación óptima
  - Pérdida de precisión: <1% en muchos casos
  - Compilación lenta, inference rápida

AWQ (Activation-aware Quantization):
  - Observa qué weights son más "activos"
  - Protege weights importantes
  - Pérdida de precisión: ~1-2%
  - Mejor que GPTQ en algunos casos
```

### Trade-off Cuantización

```{code-cell} ipython3
# Comparación de diferentes niveles de cuantización
import matplotlib.pyplot as plt
import numpy as np

class QuantizationComparison:
    """Compara diferentes niveles de cuantización"""

    def __init__(self, model_size_billions=7):
        self.model_size = model_size_billions
        self.configs = {
            'float32': {
                'bytes_per_param': 4,
                'speed_multiplier': 1.0,
                'quality_degradation': 0.0,
                'color': '#3498db'
            },
            'float16': {
                'bytes_per_param': 2,
                'speed_multiplier': 0.95,
                'quality_degradation': 0.001,
                'color': '#2ecc71'
            },
            'INT8': {
                'bytes_per_param': 1,
                'speed_multiplier': 0.90,
                'quality_degradation': 0.01,
                'color': '#f39c12'
            },
            'INT4 (AWQ)': {
                'bytes_per_param': 0.5,
                'speed_multiplier': 0.80,
                'quality_degradation': 0.02,
                'color': '#e74c3c'
            }
        }

    def calculate_metrics(self, precision):
        """Calcula métricas para una precisión dada"""
        config = self.configs[precision]
        size_gb = self.model_size * config['bytes_per_param']
        base_speed = 100  # tokens/s base
        speed = base_speed * config['speed_multiplier']
        quality = 100 * (1 - config['quality_degradation'])

        return {
            'size_gb': size_gb,
            'speed_tokens_s': speed,
            'quality_percent': quality,
            'color': config['color']
        }

    def print_comparison(self):
        """Imprime comparación de todas las configuraciones"""
        print(f"Comparación de Cuantización - Modelo {self.model_size}B")
        print("="*70)
        print(f"{'Precisión':<15} {'Tamaño':<12} {'Velocidad':<15} {'Calidad':<10}")
        print("-"*70)

        for precision in self.configs.keys():
            metrics = self.calculate_metrics(precision)
            print(f"{precision:<15} {metrics['size_gb']:>6.1f} GB    "
                  f"{metrics['speed_tokens_s']:>6.1f} tok/s    "
                  f"{metrics['quality_percent']:>5.1f}%")

    def plot_comparison(self):
        """Visualiza la comparación"""
        precisions = list(self.configs.keys())
        metrics = [self.calculate_metrics(p) for p in precisions]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Tamaño del modelo
        sizes = [m['size_gb'] for m in metrics]
        colors = [m['color'] for m in metrics]

        ax1.barh(precisions, sizes, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Tamaño del Modelo (GB)', fontsize=11)
        ax1.set_title('Tamaño en Disco/Memoria', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')

        # Añadir valores
        for i, size in enumerate(sizes):
            ax1.text(size + 0.5, i, f'{size:.1f} GB',
                     va='center', fontsize=9)

        # Velocidad
        speeds = [m['speed_tokens_s'] for m in metrics]

        ax2.barh(precisions, speeds, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Velocidad (tokens/segundo)', fontsize=11)
        ax2.set_title('Throughput de Generación', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')

        for i, speed in enumerate(speeds):
            ax2.text(speed + 1, i, f'{speed:.0f}',
                     va='center', fontsize=9)

        # Calidad
        qualities = [m['quality_percent'] for m in metrics]

        ax3.barh(precisions, qualities, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Calidad Relativa (%)', fontsize=11)
        ax3.set_title('Calidad del Modelo', fontsize=12)
        ax3.set_xlim([95, 100.5])
        ax3.grid(True, alpha=0.3, axis='x')

        for i, quality in enumerate(qualities):
            ax3.text(quality + 0.1, i, f'{quality:.1f}%',
                     va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

# Análisis para Llama 2 7B
comparison = QuantizationComparison(model_size_billions=7)
comparison.print_comparison()
print()
comparison.plot_comparison()

# Análisis de trade-offs
print("\nAnálisis de Trade-offs:")
print("="*70)
print("float32: Máxima calidad, pero 4x el tamaño de fp16")
print("  → Usar solo si la precisión es crítica")
print()
print("float16: Excelente balance, estándar en la industria")
print("  → Recomendado para la mayoría de casos")
print()
print("INT8: 4x compresión vs fp16, degradación mínima (<1%)")
print("  → Ideal para deployment con memoria limitada")
print()
print("INT4 (AWQ): 8x compresión vs fp16, ~2% degradación")
print("  → Mejor para edge devices o cuando la memoria es crítica")
```

---

## Parte 6: Elección de Hardware

### GPU para Serving

```
RECOMENDACIONES POR CARGA:

Baja carga (< 10 req/s):
  ✓ RTX 4090 (24 GB): Cabe Llama 7B sin cuantización
  Costo: $1500 (una vez)
  Throughput: 100 tokens/s

Media carga (10-50 req/s):
  ✓ V100 (32 GB): Llama 13B sin cuantización
  ✓ 4x RTX 4090: Distribución de carga
  Costo: $10K-20K
  Throughput: 200-400 tokens/s

Alta carga (50+ req/s):
  ✓ A100 (80 GB): Llama 70B + KV-Cache
  ✓ Datacenter con varias GPUs
  Costo: $100K+
  Throughput: 1000+ tokens/s
```

### CPU vs GPU

```
CPU (Intel Xeon):
  Pro: Muchos núcleos (64+)
  Con: Lento para operaciones matriciales
  Uso: Solo si no puedes pagar GPU
  Velocidad: 1-5 tokens/s por proceso

GPU:
  Pro: Miles de núcleos CUDA optimizados para matrices
  Con: Costoso
  Velocidad: 50-500 tokens/s
```

---

## Parte 7: Pipeline Completo de Serving

```
Cliente HTTP
    ↓
[Servidor vLLM/TensorRT-LLM]
    ├─→ Request 1: "¿Hola?"
    ├─→ Request 2: "¿Cómo?"
    └─→ Request 3: "¿Qué hay?"
    ↓
[Continuous Batching]
    Paso 1: Procesa 1 token de cada request
    Paso 2: Procesa 1 token de cada request
    ...
    ↓
[GPU con Modelo Cuantizado (INT4)]
    KV-Cache: 16 GB (cabe)
    Pesos: 3.5 GB (cabe)
    Total: 19.5 GB (cabe en V100)
    ↓
[Speculative Decoding Auxiliar]
    Modelo 1.5B especula 4 tokens
    Modelo 70B valida
    ↓
[Respuestas Generadas]
    Request 1: "Hola, ¿cómo estás?" → 50 tokens
    Request 2: "Estoy bien, gracias" → 40 tokens
    Request 3: "Todo está funcionando" → 45 tokens
    ↓
[Retorna a Cliente]
    Throughput: ~135 tokens / ~2 segundos = 67.5 tokens/s
```

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **KV-Cache:** ¿Por qué es tan importante? ¿Qué pasaría sin él?

2. **Continuous Batching:** ¿Cómo afecta si las solicitudes tienen longitudes muy diferentes (1 solicitud de 1000 tokens, 10 solicitudes de 10 tokens)?

3. **Speculative Decoding + Constrained:** ¿Cómo interactúan? ¿Beneficio neto?

### Ejercicios Prácticos:

1. **Cálculo de KV-Cache:**
   ```
   Modelo: 70B parámetros, 80 capas
   Dim oculta: 8192
   Usuarios: 10 concurrentes, promedio 2000 tokens/usuario

   Calcula tamaño total de KV-Cache en GB
   (Asume 2 bytes por parámetro - float16)
   ```

2. **Throughput Analysis:**
   ```
   Escenario 1: Sin continuous batching, 1 usuario
     - Genera 100 tokens en 10 segundos
     - Throughput: 10 tokens/s

   Escenario 2: Con continuous batching, 20 usuarios
     - Todos generan 100 tokens
     - Total 2000 tokens en 12 segundos
     - Throughput: ? tokens/s

   Calcula mejora de throughput
   ```

3. **Cuantización: Trade-off:**
   ```
   Modelo: Llama 13B

   float32:  52 GB modelo, 100 tokens/s
   INT8:     13 GB modelo, 90 tokens/s
   INT4:     6.5 GB modelo, 70 tokens/s

   Si tu GPU tiene 24 GB:
   - ¿Qué configuración puedes usar?
   - ¿Cuál elegirías y por qué?
   ```

4. **Reflexión escrita (350 palabras):** "Los sistemas de producción usan múltiples técnicas simultáneamente: KV-Cache, continuous batching, cuantización, speculative decoding. ¿En qué orden las implementarías si tuvieras que priorizar? ¿Por qué?"

---

## Puntos Clave

- **KV-Cache:** Almacena Keys/Values previos; 10-100x más rápido pero consume mucha memoria
- **vLLM:** Framework popular con paged attention y continuous batching
- **Continuous Batching:** Acepta solicitudes nuevas mientras genera; mejora throughput
- **Speculative Decoding:** Modelo pequeño especula, modelo grande valida; 3-5x más rápido
- **Cuantización INT8/INT4:** Reduce tamaño modelo de 28 GB a 3.5 GB; <2% pérdida de calidad
- **Hardware:** GPU esencial (100x+ más rápido que CPU)
- **Pipeline integrado:** Combina todas las técnicas para máxima velocidad

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- KV-Cache es esencial para velocidad (10-100x) pero consume memoria proporcional a seq_len * batch_size
- Continuous batching mantiene GPU ocupada aceptando nuevas solicitudes mientras genera, maximizando throughput
- Speculative decoding con modelo draft pequeño logra 2-4x speedup si tasa de aceptación >60%
- Cuantización INT8 es sweet spot: 4x compresión con <1% pérdida de calidad
- Framework choice: vLLM para uso general, TensorRT-LLM para máxima velocidad, SGLang para constrained decoding

**Siguiente paso:** En la próxima lectura analizaremos **Token Economics**, calculando costos de APIs vs self-hosting, break-even points, y cómo optimizar costos en producción con prompt caching.
```

```{admonition} Verifica tu comprensión
:class: note
1. Calcula el tamaño de KV-Cache para Llama 2 70B (80 capas, hidden=8192) con 10 usuarios, 2K tokens cada uno
2. ¿Por qué continuous batching es superior a batching tradicional? Da un ejemplo numérico
3. Si tu modelo genera a 100 tok/s sin speculative decoding, ¿qué speedup esperas con 70% acceptance rate y k=4?
4. ¿Cuándo elegirías INT4 sobre INT8? Considera memoria, velocidad y calidad
```

---

## Referencias

- vLLM. [vLLM: Easy, Fast, and Cheap LLM Serving](https://docs.vllm.ai/). vLLM.
- Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP.
- Hugging Face. [Text Generation Inference](https://huggingface.co/docs/text-generation-inference). HuggingFace.

