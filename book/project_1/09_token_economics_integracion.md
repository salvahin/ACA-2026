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

# Lectura 8: Token Economics

```{admonition} Ejecutar en Google Colab
:class: tip

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salvahin/ACA-2026/blob/main/book/notebooks/09_token_economics_integracion.ipynb)
```

```{code-cell} ipython3
:tags: [remove-input, setup]

# Setup Colab Environment
!pip install -q numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate triton
print('Dependencies installed!')
```

```{admonition} Objetivos de Aprendizaje
:class: tip
Al finalizar esta lectura podrás:
- Calcular costos de APIs (OpenAI, Claude, etc.) basados en tokens input/output
- Estimar costos de self-hosting considerando hardware, electricidad y amortización
- Determinar break-even points entre API y self-hosting para diferentes volúmenes
- Aplicar prompt caching para reducir costos de entrada 60-80%
- Evaluar cuantización como estrategia de reducción de costos (INT8/INT4)
```

```{admonition} Milestone del Proyecto
:class: important
Después de esta lectura podrás: Calcular el costo total de operación de KernelAgent en producción, decidir entre usar APIs de LLM vs desplegar modelos propios, y optimizar costos con técnicas como prompt caching y cuantización. Sabrás cuándo escalar de API a self-hosting basado en volumen de tokens.
```

## Introducción

¿Cuánto cuesta generar un texto? Si usas OpenAI API, pagas por tokens. Si ejecutas modelos localmente, pagas por electricidad y hardware. Entender la economía de tokens es crítico para:

- Elegir si usar API vs auto-hosting
- Optimizar costos en producción
- Diseñar aplicaciones escalables
- Evaluar viabilidad económica de proyectos IA

:::{figure} diagrams/token_economics.png
:name: fig-token-economics
:alt: Análisis de costos API vs Self-Hosted y break-even points
:align: center
:width: 90%

**Figura 1:** Economía de tokens - comparación de costos entre APIs y self-hosting con punto de equilibrio.
:::

---

## Parte 1: Precios de API

### OpenAI Pricing (Enero 2024)

```
GPT-4 Turbo:
  Input:  $0.01 / 1K tokens
  Output: $0.03 / 1K tokens

GPT-3.5 Turbo:
  Input:  $0.0005 / 1K tokens
  Output: $0.0015 / 1K tokens

Claude 2 (Anthropic):
  Input:  $0.008 / 1K tokens
  Output: $0.024 / 1K tokens
```

```{code-cell} ipython3
# Calculadora de costos de API
import plotly.graph_objects as go
import numpy as np

# Precios por 1M tokens (input/output)
precios = {
    'GPT-4o': (5.0, 15.0),
    'GPT-4o-mini': (0.15, 0.60),
    'Claude 3.5 Sonnet': (3.0, 15.0),
    'Claude 3 Haiku': (0.25, 1.25),
    'Llama 3.1 405B': (3.0, 3.0),
}

# Escenario: 10K requests/día, 1K tokens input, 500 output
requests_dia = 10000
tokens_in = 1000
tokens_out = 500
dias = 30

fig = go.Figure()

modelos = list(precios.keys())
costos_mensuales = []

for modelo, (precio_in, precio_out) in precios.items():
    costo_dia = requests_dia * ((tokens_in * precio_in / 1e6) + (tokens_out * precio_out / 1e6))
    costos_mensuales.append(costo_dia * dias)

colors = ['#FF6B6B' if c > 1000 else '#4ECDC4' if c > 100 else '#95E1A3' for c in costos_mensuales]

fig.add_trace(go.Bar(x=modelos, y=costos_mensuales, marker_color=colors,
                     text=[f'${c:,.0f}' for c in costos_mensuales], textposition='outside'))

fig.update_layout(
    title=f"Costo Mensual: {requests_dia:,} req/día × {tokens_in}+{tokens_out} tokens",
    yaxis_title="USD/mes",
    height=400
)
fig.show()
```

### Ejemplo de Costo

```{code-cell} ipython3
# Calculadora de costos de API para LLMs
class TokenCostCalculator:
    """Calcula costos de uso de APIs de LLM"""

    # Precios actuales (Enero 2024) en USD por 1K tokens
    PRICING = {
        'GPT-4 Turbo': {'input': 0.01, 'output': 0.03},
        'GPT-3.5 Turbo': {'input': 0.0005, 'output': 0.0015},
        'Claude 2': {'input': 0.008, 'output': 0.024},
        'Llama 2 7B (self-hosted)': {'input': 0.0, 'output': 0.0}  # Sin costo de API
    }

    def __init__(self, model='GPT-3.5 Turbo'):
        self.model = model
        self.input_price = self.PRICING[model]['input']
        self.output_price = self.PRICING[model]['output']

    def words_to_tokens(self, words, factor=1.3):
        """Convierte palabras a tokens (aproximado)"""
        return int(words * factor)

    def calculate_cost(self, input_tokens, output_tokens):
        """Calcula el costo total"""
        input_cost = (input_tokens / 1000) * self.input_price
        output_cost = (output_tokens / 1000) * self.output_price
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost
        }

    def format_cost(self, amount):
        """Formatea el costo"""
        if amount < 0.01:
            return f"${amount:.4f}"
        return f"${amount:.2f}"

# Ejemplo: Resumir 100 artículos
print("Tarea: Resumir 100 artículos de 2000 palabras cada uno")
print("="*60)

calculator = TokenCostCalculator('GPT-3.5 Turbo')

# Calcular tokens
articles = 100
words_per_article = 2000
summary_words = 200

input_words = articles * words_per_article
output_words = articles * summary_words

input_tokens = calculator.words_to_tokens(input_words)
output_tokens = calculator.words_to_tokens(output_words)

print(f"Input:")
print(f"  {articles} artículos × {words_per_article} palabras")
print(f"  ≈ {input_words:,} palabras × 1.3 = {input_tokens:,} tokens")

print(f"\nOutput:")
print(f"  {articles} resúmenes × {summary_words} palabras")
print(f"  ≈ {output_words:,} palabras × 1.3 = {output_tokens:,} tokens")

# Calcular costos
costs = calculator.calculate_cost(input_tokens, output_tokens)

print(f"\nCostos con {calculator.model}:")
print(f"  Input:  {calculator.format_cost(costs['input_cost'])} "
      f"({input_tokens:,} tokens @ ${calculator.input_price}/1K)")
print(f"  Output: {calculator.format_cost(costs['output_cost'])} "
      f"({output_tokens:,} tokens @ ${calculator.output_price}/1K)")
print(f"  TOTAL:  {calculator.format_cost(costs['total_cost'])}")

# Comparar con otros modelos
print("\n" + "="*60)
print("Comparación entre modelos:")
print(f"{'Modelo':<25} {'Costo Total':<15} {'Costo/Artículo':<15}")
print("-"*60)

for model_name in TokenCostCalculator.PRICING.keys():
    calc = TokenCostCalculator(model_name)
    model_costs = calc.calculate_cost(input_tokens, output_tokens)
    cost_per_article = model_costs['total_cost'] / articles

    print(f"{model_name:<25} {calc.format_cost(model_costs['total_cost']):<15} "
          f"{calc.format_cost(cost_per_article):<15}")

print("\nConclusión: GPT-3.5 Turbo es muy económico para esta tarea (¡menos de $1!)")
```

---

## Parte 2: Costos de Auto-Hosting

### Hardware

```
Opción 1: RTX 4090 (24 GB)
  Costo inicial: $1,500
  Potencia: 450W
  Costo electricidad: $0.12/kWh (USA), ~$30/mes en uso continuo
  Vida útil: 3-5 años
  Costo amortizado: $1,500 / 48 meses = $31/mes + $30 electricidad = $61/mes

Opción 2: A100 (80 GB)
  Costo inicial: $9,000 + DC cooling/power
  Potencia: 250W
  Costo total amortizado: $300-400/mes

Opción 3: Cloud GPU (AWS p3.2xlarge)
  Costo: $3.06/hora = $0.051/minuto
  Uso intensivo (8 horas/día): $24.48/día = $735/mes
```

### Throughput

```
Llama 2 7B en RTX 4090:
  Throughput: ~50 tokens/segundo

Si usas 8 horas al día generando:
  8 horas * 3600 segundos * 50 tokens/s
  = 1,440,000 tokens/día
  = 43,200,000 tokens/mes

Costo por token: $61/mes / 43,200,000 tokens
                = $0.0000014 / token (input + output)
```

### Comparación: API vs Self-Hosted

```{code-cell} ipython3
# Comparación de costos: API vs Self-Hosted
import matplotlib.pyplot as plt
import numpy as np

class CostComparison:
    """Compara costos de API vs self-hosting"""

    def __init__(self):
        # Costos de API (promedio input + output)
        self.api_cost_per_token = 0.001  # $1 por millón de tokens

        # Costos de hardware self-hosted
        self.hardware_costs = {
            'RTX 4090': {
                'initial': 1500,
                'electricity_monthly': 30,
                'lifespan_months': 48,
                'tokens_per_second': 50
            },
            'A100 80GB': {
                'initial': 9000,
                'electricity_monthly': 50,
                'lifespan_months': 60,
                'tokens_per_second': 150
            }
        }

    def api_monthly_cost(self, tokens_per_month):
        """Calcula costo mensual de API"""
        return tokens_per_month * self.api_cost_per_token / 1000

    def self_hosted_monthly_cost(self, hardware):
        """Calcula costo mensual amortizado de self-hosted"""
        hw = self.hardware_costs[hardware]
        amortized = hw['initial'] / hw['lifespan_months']
        return amortized + hw['electricity_monthly']

    def break_even_point(self, hardware):
        """Calcula el punto de equilibrio en tokens/mes"""
        monthly_fixed = self.self_hosted_monthly_cost(hardware)
        # monthly_fixed = api_cost_per_token * break_even_tokens / 1000
        break_even_tokens = (monthly_fixed * 1000) / self.api_cost_per_token
        return break_even_tokens

    def print_comparison(self, tokens_per_month):
        """Imprime comparación para un volumen dado"""
        print(f"Comparación de costos para {tokens_per_month:,} tokens/mes")
        print("="*70)

        api_cost = self.api_monthly_cost(tokens_per_month)
        print(f"{'Opción':<25} {'Costo Mensual':<20} {'Costo/M tokens':<20}")
        print("-"*70)
        print(f"{'API (GPT-3.5)':<25} ${api_cost:>8.2f}             "
              f"${self.api_cost_per_token * 1000:>6.2f}")

        for hw_name in self.hardware_costs.keys():
            hw_cost = self.self_hosted_monthly_cost(hw_name)
            cost_per_m = (hw_cost / tokens_per_month) * 1_000_000 if tokens_per_month > 0 else 0
            print(f"{hw_name + ' (self-hosted)':<25} ${hw_cost:>8.2f}             "
                  f"${cost_per_m:>6.2f}")

    def plot_break_even(self):
        """Visualiza el punto de equilibrio"""
        tokens_range = np.logspace(6, 9, 50)  # 1M a 1B tokens

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Gráfico 1: Costo total vs volumen
        api_costs = [self.api_monthly_cost(t) for t in tokens_range]

        ax1.plot(tokens_range / 1e6, api_costs, label='API (GPT-3.5)',
                linewidth=2, color='blue')

        colors = ['orange', 'green']
        for i, hw_name in enumerate(self.hardware_costs.keys()):
            hw_cost = self.self_hosted_monthly_cost(hw_name)
            ax1.axhline(y=hw_cost, label=f'{hw_name} (self-hosted)',
                       linestyle='--', linewidth=2, color=colors[i])

            # Marcar punto de equilibrio
            break_even = self.break_even_point(hw_name)
            ax1.plot(break_even / 1e6, hw_cost, 'o', markersize=10,
                    color=colors[i], markeredgecolor='black', markeredgewidth=2)
            ax1.text(break_even / 1e6, hw_cost + 50,
                    f'Break-even:\n{break_even/1e6:.1f}M tokens',
                    ha='center', fontsize=9, fontweight='bold')

        ax1.set_xlabel('Tokens por Mes (Millones)', fontsize=11)
        ax1.set_ylabel('Costo Mensual (USD)', fontsize=11)
        ax1.set_title('Costo Mensual: API vs Self-Hosted', fontsize=12)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')

        # Gráfico 2: Ahorro relativo
        tokens_test = [10e6, 50e6, 100e6, 500e6, 1000e6]  # Diferentes volúmenes
        hardware_names = list(self.hardware_costs.keys())

        x = np.arange(len(tokens_test))
        width = 0.25

        for i, hw_name in enumerate(hardware_names):
            savings = []
            for tokens in tokens_test:
                api_cost = self.api_monthly_cost(tokens)
                hw_cost = self.self_hosted_monthly_cost(hw_name)
                saving_pct = ((api_cost - hw_cost) / api_cost * 100) if api_cost > hw_cost else 0
                savings.append(saving_pct)

            offset = width * (i - 0.5)
            ax2.bar(x + offset, savings, width, label=hw_name, alpha=0.8)

        ax2.set_xlabel('Volumen Mensual (Millones de tokens)', fontsize=11)
        ax2.set_ylabel('Ahorro vs API (%)', fontsize=11)
        ax2.set_title('Ahorro Relativo de Self-Hosted vs API', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{int(t/1e6)}M' for t in tokens_test])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='red', linestyle='-', linewidth=1)

        plt.tight_layout()
        plt.show()

# Análisis
comparison = CostComparison()

# Caso 1: Bajo volumen (1M tokens/mes)
print("CASO 1: Bajo Volumen")
comparison.print_comparison(1_000_000)
print()

# Caso 2: Medio volumen (50M tokens/mes)
print("\nCASO 2: Volumen Medio")
comparison.print_comparison(50_000_000)
print()

# Caso 3: Alto volumen (500M tokens/mes)
print("\nCASO 3: Alto Volumen")
comparison.print_comparison(500_000_000)
print()

# Puntos de equilibrio
print("\n" + "="*70)
print("PUNTOS DE EQUILIBRIO:")
print("-"*70)
for hw_name in comparison.hardware_costs.keys():
    break_even = comparison.break_even_point(hw_name)
    print(f"{hw_name}: {break_even/1e6:.1f}M tokens/mes "
          f"(${comparison.self_hosted_monthly_cost(hw_name):.2f}/mes)")

print("\n" + "="*70)
print("Conclusión:")
print("  - < 30M tokens/mes: Usa API (más simple y barato)")
print("  - 30-100M tokens/mes: RTX 4090 self-hosted empieza a tener sentido")
print("  - > 100M tokens/mes: Self-hosted es claramente más barato")
print("  - Considera también: privacidad, latencia, flexibilidad")

# Visualización
comparison.plot_break_even()
```

---

## Parte 3: Contexto Largo y Costo

### Tamaño de Contexto

```
Versión      Contexto    Costo/M tokens (entrada)
GPT-3.5      4K          $0.5 (normal)
GPT-4        8K          $3 (normal)
             32K         $6 (2x)
             128K        $3 (con descuento)  ← Nuevo
Claude 2     100K        $2.3
Llama 2      4K          $0 (self-hosted)
             + context extension: puedes forzar más, pero menos eficiente
```

### Ejemplo: Análisis de Documento Largo

```
Documento: 50,000 palabras = 65,000 tokens

Opción 1: GPT-3.5 + 4K contexto
  - Divide en chunks, procesa independientemente
  - Múltiples solicitudes
  - No hay contexto entre chunks

Opción 2: Claude 2 + 100K contexto
  - Una solicitud, todo el documento
  - Contexto completo
  - Costo: 65,000 tokens * $2.3 = $0.15

Costo Opción 2 << Costo Opción 1 si necesitas contexto
```

---

## Parte 4: Prompts Largos - Compression

### Problema

```
Patrón común:
1. Sistema prompt: 500 tokens
2. Ejemplos (few-shot): 2000 tokens
3. Documento usuario: 1000 tokens
4. Pregunta: 50 tokens

Total: 3550 tokens

Si generas 100 respuestas/día:
  3550 * 100 = 355,000 input tokens/día
  Solo prompts, sin respuestas
```

### Prompt Compression

#### Opción 1: Resumir Ejemplos

```
Antes (few-shot original):
  "Usuario: ¿Cuál es la capital de Francia?
   Asistente: París
   Usuario: ¿Cuál es la capital de España?
   Asistente: Madrid
   ..."
  = 50 ejemplos, 2000 tokens

Después (resumido):
  "Ejemplos previos: respuestas a preguntas geográficas"
  = 10 tokens

Pero: ¿Pierde la especificidad?
Riesgo: modelo menos preciso con prompt comprimido
```

#### Opción 2: LLM-based Compression

```
Tool: LLMCompress (comprime prompts sin perder significado)

Entrada:
  "El usuario pregunta sobre ciudades.
   Se han mostrado 50 ejemplos de ciudades españolas e internacionales.
   El usuario pregunta por la capital de Italia."

Salida comprimida:
  "Q: Italia?
   A: Roma"
  (De miles de tokens a cientos)
```

#### Opción 3: Smart Caching

```{code-cell} ipython3
# Simulación de ahorro con prompt caching
import matplotlib.pyplot as plt
import numpy as np

class PromptCachingAnalyzer:
    """Analiza el impacto del prompt caching en costos"""

    def __init__(self, input_cost_per_1k=0.0005, output_cost_per_1k=0.0015):
        self.input_cost = input_cost_per_1k
        self.output_cost = output_cost_per_1k

    def calculate_costs(self, system_tokens, examples_tokens, user_tokens,
                       question_tokens, output_tokens, num_requests,
                       cache_hit_rate=0.0):
        """Calcula costos con y sin caching"""

        # Sin caching: todos los tokens se envían cada vez
        total_input_no_cache = (system_tokens + examples_tokens +
                                user_tokens + question_tokens) * num_requests
        cost_no_cache = (total_input_no_cache / 1000 * self.input_cost +
                        output_tokens * num_requests / 1000 * self.output_cost)

        # Con caching: system + examples se cachean
        cacheable = system_tokens + examples_tokens
        non_cacheable = user_tokens + question_tokens

        # Primera request: todo se paga
        first_request_input = cacheable + non_cacheable

        # Requests subsecuentes: solo pagar por cache misses + non-cacheable
        cache_hits = int((num_requests - 1) * cache_hit_rate)
        cache_misses = (num_requests - 1) - cache_hits

        total_input_with_cache = (first_request_input +
                                 cache_hits * non_cacheable +
                                 cache_misses * (cacheable + non_cacheable))

        cost_with_cache = (total_input_with_cache / 1000 * self.input_cost +
                          output_tokens * num_requests / 1000 * self.output_cost)

        return {
            'no_cache': {
                'total_input_tokens': total_input_no_cache,
                'total_cost': cost_no_cache
            },
            'with_cache': {
                'total_input_tokens': total_input_with_cache,
                'total_cost': cost_with_cache
            },
            'savings': {
                'tokens': total_input_no_cache - total_input_with_cache,
                'cost': cost_no_cache - cost_with_cache,
                'percentage': ((cost_no_cache - cost_with_cache) / cost_no_cache * 100)
                             if cost_no_cache > 0 else 0
            }
        }

# Configuración del prompt
print("Arquitectura de Prompt:")
print("="*60)
system_tokens = 500
examples_tokens = 2000
user_tokens = 1000
question_tokens = 50
output_tokens = 200

print(f"  Sistema prompt: {system_tokens} tokens → CACHEABLE")
print(f"  Ejemplos (few-shot): {examples_tokens} tokens → CACHEABLE")
print(f"  Documento usuario: {user_tokens} tokens → NUEVO cada vez")
print(f"  Pregunta: {question_tokens} tokens → NUEVO cada vez")
print(f"  Respuesta: {output_tokens} tokens → OUTPUT")
print()

cacheable = system_tokens + examples_tokens
non_cacheable = user_tokens + question_tokens
print(f"Total cacheable: {cacheable} tokens")
print(f"Total no-cacheable: {non_cacheable} tokens")
print(f"Total por request (sin cache): {cacheable + non_cacheable} tokens")
print(f"Total por request (con cache, hit): {non_cacheable} tokens")

# Análisis
analyzer = PromptCachingAnalyzer()

print("\n" + "="*60)
print("ANÁLISIS DE COSTOS (100 requests, 95% cache hit rate):")
print("="*60)

results = analyzer.calculate_costs(
    system_tokens, examples_tokens, user_tokens, question_tokens,
    output_tokens, num_requests=100, cache_hit_rate=0.95
)

print("\nSIN caching:")
print(f"  Total input tokens: {results['no_cache']['total_input_tokens']:,}")
print(f"  Costo total: ${results['no_cache']['total_cost']:.4f}")

print("\nCON caching (95% hit rate):")
print(f"  Total input tokens: {results['with_cache']['total_input_tokens']:,}")
print(f"  Costo total: ${results['with_cache']['total_cost']:.4f}")

print("\nAHORRO:")
print(f"  Tokens ahorrados: {results['savings']['tokens']:,}")
print(f"  Dinero ahorrado: ${results['savings']['cost']:.4f}")
print(f"  Porcentaje de ahorro: {results['savings']['percentage']:.1f}%")

# Visualización: Impacto de cache hit rate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

cache_hit_rates = np.linspace(0, 1, 21)
num_requests = 100

costs_no_cache = []
costs_with_cache = []
savings_pct = []

for rate in cache_hit_rates:
    result = analyzer.calculate_costs(
        system_tokens, examples_tokens, user_tokens, question_tokens,
        output_tokens, num_requests, cache_hit_rate=rate
    )
    costs_no_cache.append(result['no_cache']['total_cost'])
    costs_with_cache.append(result['with_cache']['total_cost'])
    savings_pct.append(result['savings']['percentage'])

# Gráfico 1: Costo vs cache hit rate
ax1.plot(cache_hit_rates * 100, costs_no_cache, label='Sin caching',
        linewidth=2, linestyle='--', color='red')
ax1.plot(cache_hit_rates * 100, costs_with_cache, label='Con caching',
        linewidth=2, color='green')
ax1.fill_between(cache_hit_rates * 100, costs_with_cache, costs_no_cache,
                 alpha=0.3, color='green', label='Ahorro')
ax1.set_xlabel('Cache Hit Rate (%)', fontsize=11)
ax1.set_ylabel('Costo Total (USD)', fontsize=11)
ax1.set_title(f'Costo Total vs Cache Hit Rate\n({num_requests} requests)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Ahorro porcentual
ax2.plot(cache_hit_rates * 100, savings_pct, linewidth=2.5, color='purple')
ax2.axhline(y=70, color='orange', linestyle='--', alpha=0.7,
           label='70% ahorro típico')
ax2.fill_between(cache_hit_rates * 100, 0, savings_pct,
                alpha=0.3, color='purple')
ax2.set_xlabel('Cache Hit Rate (%)', fontsize=11)
ax2.set_ylabel('Ahorro (%)', fontsize=11)
ax2.set_title('Porcentaje de Ahorro vs Cache Hit Rate', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Recomendaciones:")
print("  - Cachea system prompts y ejemplos few-shot")
print("  - En producción, 80-95% cache hit rate es típico")
print("  - Ahorro de 60-80% en costos de input es común")
print("  - OpenAI, Anthropic, y otros soportan prompt caching")
```

---

## Parte 5: Análisis de Costo por Tarea

### Ejemplo 1: Chatbot de Servicio al Cliente

```
Estadísticas:
  - 1000 usuarios/día
  - Promedio 5 mensajes/usuario
  - 5000 mensajes/día

Análisis de tokens:
  Sistema prompt: 300 tokens (instrucciones)
  Historial chat: 500 tokens (promedio)
  Mensaje usuario: 50 tokens
  Total entrada: 850 tokens

  Respuesta: 100 tokens (corta)

Costo por mensaje:
  Entrada: 850 * $0.0005 / 1000 = $0.0004
  Salida: 100 * $0.0015 / 1000 = $0.00015
  Total: $0.00055

Costo diario: 5000 * $0.00055 = $2.75
Costo mensual: $82.50

Viabilidad: SÍ (muy barato, puede ser gratis con publicidad)
```

### Ejemplo 2: Análisis de Documentos Empresariales

```
Estadísticas:
  - 100 documentos/mes
  - Documentos grandes (10,000 palabras c/u)
  - Análisis profundo (extraer insights, generar resumen largo)

Análisis de tokens:
  Documento: 10,000 palabras = 13,000 tokens
  Sistema prompt: 500 tokens
  Total entrada: 13,500 tokens

  Análisis+respuesta: 1000 tokens

Costo por documento:
  Entrada: 13,500 * $0.0005 / 1000 = $0.0068
  Salida: 1000 * $0.0015 / 1000 = $0.0015
  Total: $0.0083

Costo mensual: 100 * $0.0083 = $0.83 (API)

Self-hosted:
  Costo hardware: $61/mes
  Con 100 documentos/mes: $0.61/doc
  Total: $61

Viabilidad con API: SÍ (muy barato)
Viabilidad con self-hosted: SÍ (hardware caro, pero reutilizable)
```

### Ejemplo 3: Generación Masiva (Síntesis de Datos)

```
Estadísticas:
  - Generar 1,000,000 registros de entrenamiento
  - Cada registro: 200 tokens entrada, 100 tokens salida

Total:
  Input: 1,000,000 * 200 = 200,000,000 tokens
  Output: 1,000,000 * 100 = 100,000,000 tokens

Costo con OpenAI GPT-3.5:
  Input: 200M * $0.0005 / 1000 = $100
  Output: 100M * $0.0015 / 1000 = $150
  Total: $250

Costo self-hosted (RTX 4090):
  Hardware: $61/mes
  Tiempo: 300M tokens / 50 tokens/s = 6,000,000 segundos = 70 días
  Entonces: ~3 GPUs, ~$200 amortizado en el mes
  Total: ~$200

Viabilidad: API es competitiva si lo necesitas rápido
Self-hosted si puedes esperar
```

---

## Parte 6: Cuantización y Ahorro

### Impacto de Cuantización en Velocidad/Costo

```
Modelo: Llama 70B

float32 (28 GPU A100):
  Costo hardware: $252K
  Velocidad: 500 tokens/s (distributed)
  Costo/token: $252K / (6 meses * 86400 * 500) ≈ $0.00097

INT8 (7 GPU A100):
  Costo hardware: $63K
  Velocidad: 450 tokens/s
  Costo/token: $63K / (6 meses * 86400 * 450) ≈ $0.00027

INT4 (2 GPU A100):
  Costo hardware: $18K
  Velocidad: 350 tokens/s
  Costo/token: $18K / (6 meses * 86400 * 350) ≈ $0.000025

Pérdida de calidad:
  float32: 0%
  INT8: <1%
  INT4: 2-3%

Recomendación: INT8 es sweet spot
```

---

## Parte 7: Decisión: API vs Self-Hosted

### Matriz de Decisión

```
Criterio                    API              Self-Hosted
─────────────────────────────────────────────────────
Tokens/mes                  < 30M            > 30M
Latencia requerida          < 500ms          flexible
Privacidad datos            No importante    Crítica
Flexibilidad modelo         Necesaria        No importante
Costo capital               No               Sí
Costo operacional           Sí               Bajo
Escalabilidad               Instantánea      Lenta
Control sobre salida        Poco             Completo
Facilidad de uso            Máxima           Mínima
```

### Ejemplo Práctico

```
APLICACIÓN: Servicio de traducción automática
Requisitos:
  - 1M traducciones/mes
  - Latencia < 2 segundos
  - Precisión > 95%
  - Datos sensibles (pueden ser privados)

ANÁLISIS:

Opción 1: API (Google Translate)
  Costo: ~$15/1M caracteres = $3-5/mes
  Latencia: 200-500ms ✓
  Privacidad: Google ve datos ✗
  Control: Ninguno ✗

Opción 2: API (OpenAI GPT-4)
  Costo: $1,000-2,000/mes (caría GPU)
  Latencia: 1-3 segundos (lento para traducción)
  Privacidad: OpenAI ve datos ✗
  Control: Alguno ✓

Opción 3: Self-hosted (Llama 7B cuantizado)
  Costo: $61/mes hardware + $50 electricidad = $111/mes
  Latencia: 500ms-1s ✓
  Privacidad: Completa ✓
  Control: Total ✓

DECISIÓN: Opción 3 (self-hosted) - mejor en casi todo
```

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **Break-even:** ¿A cuántos tokens mensuales cambia el break-even entre diferentes opciones? (Considera hardware de vida útil 3 años, electricidad $0.15/kWh)

2. **Escalabilidad:** Si necesitas 1B tokens/mes, ¿cómo cambiaría tu decisión?

3. **Experiencia del usuario:** Si la latencia es crítica (chatbot), ¿cómo afecta esto la decisión de costo?

### Ejercicios Prácticos:

1. **Cálculo de costo:**
   ```
   Tu aplicación:
   - 500,000 tokens entrada/mes
   - 100,000 tokens salida/mes

   Calcula costo mensual con:
   a) GPT-3.5 API
   b) Claude API
   c) Self-hosted RTX 4090 (amortizado a 3 años)

   ¿Cuál es más barato?
   ¿En cuántos meses se amortiza el hardware?
   ```

2. **Prompt Compression Simulation:**
   ```
   Sistema prompt: 1000 tokens
   Few-shot ejemplos: 5000 tokens
   Usuario input: 500 tokens
   Total: 6500 tokens por solicitud

   Si usas caching:
   - Cache hits: 95% del tiempo
   - Nuevo input por solicitud: 500 tokens

   Costo sin caching: 6500 tokens * $0.002 = $0.013
   Costo con caching: (6500 * 1 + 500 * 99) / 100 = 115 tokens prom = $0.00023

   ¿Ahorro?
   ```

3. **Break-even Analysis:**
   ```
   Hardware A100: $9000 inicial, $200/mes electricidad
   vs
   API a $0.002/token

   Si necesitas N tokens/mes:
   - Amortización hardware: $9000/36 + $200 = $450/mes
   - Costo API: N * $0.002 / 1000

   Break-even: 225M tokens/mes

   ¿Tiene sentido para tu caso de uso?
   ```

4. **Reflexión escrita (350 palabras):** "La economía de tokens sigue cambiando. Modelos se baratizan, hardware baja de precio. ¿Cómo predirías qué será más barato en 2 años? ¿Invertirías en hardware ahora o esperarías?"

---

## Puntos Clave

- **API:** Barata (<30M tokens/mes), fácil de usar, pero privacidad
- **Self-hosted:** Más barata para alto volumen (>30M tokens/mes), total control, pero overhead operacional
- **Break-even:** Típicamente 20-50M tokens/mes dependiendo de hardware y tarifa API
- **Contexto largo:** Crucial para documentos; algunas APIs cobran 2x por 128K vs 4K
- **Prompt caching:** Reduce tokens de entrada; invaluable para aplicaciones repetitivas
- **Cuantización:** INT8/INT4 reduce hardware 75-85%, impacto mínimo en calidad
- **Matriz de decisión:** Combina costo, latencia, privacidad, escalabilidad, control

```{admonition} Resumen
:class: important
**Lo que aprendiste:**
- APIs son económicas (<$100/mes) para volúmenes bajos (<30M tokens/mes) pero cuestan más a escala
- Self-hosting tiene costo fijo (hardware + electricidad) que se amortiza con volumen alto
- Break-even típico: 20-50M tokens/mes (RTX 4090: ~30M, A100: ~200M)
- Prompt caching ahorra 60-80% en costos de input para prompts repetitivos (system + examples)
- Cuantización INT8 reduce costos de hardware 4x con pérdida <1% de calidad

**Siguiente paso:** En la lectura final exploraremos **Sistemas Agénticos**, donde aprenderás sobre tool use, RAG, multi-agent architectures, y cómo construir sistemas de IA que razonan, planifican y actúan de forma autónoma.
```

```{admonition} Verifica tu comprensión
:class: note
1. Tu app genera 500K tokens/mes. Calcula costos con: a) GPT-3.5 API, b) RTX 4090 self-hosted. ¿Cuál es más barato?
2. Si implementas prompt caching con 90% hit rate en prompts de 2K tokens, ¿cuánto ahorras mensualmente en 100K requests?
3. ¿En cuántos meses se amortiza una A100 ($9K) si procesas 100M tokens/mes vs usar API ($200/mes)?
4. ¿Cuándo recomendarías API sobre self-hosting incluso con volumen alto (>50M tokens/mes)?
```

```{code-cell} ipython3
# Calculadora interactiva de costos para proyectos reales
import matplotlib.pyplot as plt
import numpy as np

class ProjectCostEstimator:
    """Estima costos de proyectos IA basados en token economics"""

    def __init__(self):
        self.api_pricing = {
            'GPT-4 Turbo': 0.01,      # USD por 1K tokens (promedio)
            'GPT-3.5 Turbo': 0.001,
            'Claude 2': 0.016,
            'Gemini Pro': 0.0005
        }

        self.hardware_costs = {
            'RTX 4090': {'initial': 1500, 'monthly': 30},
            'A100 80GB': {'initial': 9000, 'monthly': 50},
            'Cloud GPU': {'initial': 0, 'monthly': 700}
        }

    def estimate_project_cost(self, project_type: str,
                             users_per_month: int,
                             use_case: str) -> Dict:
        """Estima costos para diferentes tipos de proyectos"""

        # Definir perfiles de uso por tipo de proyecto
        profiles = {
            'Chatbot': {
                'tokens_per_interaction': 2000,
                'interactions_per_user': 10,
                'description': 'Chatbot de servicio al cliente'
            },
            'Document Analysis': {
                'tokens_per_interaction': 15000,
                'interactions_per_user': 5,
                'description': 'Análisis y resumen de documentos'
            },
            'Code Generation': {
                'tokens_per_interaction': 5000,
                'interactions_per_user': 20,
                'description': 'Asistente de programación'
            },
            'Data Synthesis': {
                'tokens_per_interaction': 1000,
                'interactions_per_user': 1000,
                'description': 'Generación masiva de datos'
            }
        }

        profile = profiles.get(project_type, profiles['Chatbot'])
        total_tokens = (users_per_month *
                       profile['interactions_per_user'] *
                       profile['tokens_per_interaction'])

        # Calcular costos para diferentes opciones
        costs = {}

        # API options
        for api_name, price_per_1k in self.api_pricing.items():
            costs[api_name] = (total_tokens / 1000) * price_per_1k

        # Self-hosted options
        for hw_name, hw_costs in self.hardware_costs.items():
            monthly = hw_costs['monthly']
            if hw_costs['initial'] > 0:
                # Amortizar a 36 meses
                monthly += hw_costs['initial'] / 36
            costs[f'{hw_name} (self-hosted)'] = monthly

        return {
            'project_type': project_type,
            'description': profile['description'],
            'users_per_month': users_per_month,
            'total_tokens': total_tokens,
            'costs': costs,
            'cheapest_option': min(costs, key=costs.get),
            'most_expensive': max(costs, key=costs.get)
        }

    def visualize_comparison(self, project_estimates: List[Dict]):
        """Visualiza comparación de costos entre proyectos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, estimate in enumerate(project_estimates[:4]):
            ax = axes[idx]

            options = list(estimate['costs'].keys())
            costs = list(estimate['costs'].values())

            # Separar API vs Self-hosted
            api_options = [opt for opt in options if 'self-hosted' not in opt]
            hw_options = [opt for opt in options if 'self-hosted' in opt]

            api_costs = [estimate['costs'][opt] for opt in api_options]
            hw_costs = [estimate['costs'][opt] for opt in hw_options]

            x_api = np.arange(len(api_options))
            x_hw = np.arange(len(api_options), len(api_options) + len(hw_options))

            bars1 = ax.bar(x_api, api_costs, color='#3498db', alpha=0.8,
                          label='API', edgecolor='black')
            bars2 = ax.bar(x_hw, hw_costs, color='#2ecc71', alpha=0.8,
                          label='Self-hosted', edgecolor='black')

            ax.set_ylabel('Costo Mensual (USD)', fontsize=10)
            ax.set_title(f"{estimate['project_type']}\n"
                        f"{estimate['users_per_month']:,} usuarios/mes",
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(options)))
            ax.set_xticklabels(options, rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_yscale('log')

            # Marcar la opción más barata
            cheapest = estimate['cheapest_option']
            cheapest_idx = options.index(cheapest)
            ax.plot(cheapest_idx, estimate['costs'][cheapest],
                   marker='*', markersize=20, color='gold',
                   markeredgecolor='black', markeredgewidth=2)

        plt.tight_layout()
        plt.show()

# Crear estimador
estimator = ProjectCostEstimator()

# Definir 4 proyectos diferentes
projects = [
    {'type': 'Chatbot', 'users': 1000, 'use_case': 'customer_service'},
    {'type': 'Document Analysis', 'users': 100, 'use_case': 'legal_docs'},
    {'type': 'Code Generation', 'users': 500, 'use_case': 'dev_assistant'},
    {'type': 'Data Synthesis', 'users': 10, 'use_case': 'training_data'}
]

print("ANÁLISIS DE COSTOS POR TIPO DE PROYECTO")
print("="*70)

estimates = []
for project in projects:
    estimate = estimator.estimate_project_cost(
        project['type'],
        project['users'],
        project['use_case']
    )
    estimates.append(estimate)

    print(f"\n{estimate['project_type'].upper()}")
    print(f"Descripción: {estimate['description']}")
    print(f"Usuarios/mes: {estimate['users_per_month']:,}")
    print(f"Tokens totales/mes: {estimate['total_tokens']:,}")
    print("-"*70)

    # Ordenar por costo
    sorted_costs = sorted(estimate['costs'].items(), key=lambda x: x[1])

    print(f"{'Opción':<30} {'Costo Mensual':>15}")
    print("-"*70)
    for option, cost in sorted_costs:
        marker = " ← MEJOR" if option == estimate['cheapest_option'] else ""
        print(f"{option:<30} ${cost:>13,.2f}{marker}")

# Visualización comparativa
print("\n" + "="*70)
print("Generando visualización comparativa...")
estimator.visualize_comparison(estimates)

# Recomendaciones finales
print("\n" + "="*70)
print("RECOMENDACIONES POR TIPO DE PROYECTO:")
print("="*70)
print("""
1. CHATBOT (bajo volumen):
   → API (GPT-3.5 o Gemini): <$100/mes
   → Escalable, fácil de implementar
   → Considera self-hosted si privacidad es crítica

2. DOCUMENT ANALYSIS (volumen medio):
   → Claude 2 para contexto largo
   → Self-hosted si >100 usuarios
   → Balance entre capacidad y costo

3. CODE GENERATION (alto volumen):
   → Self-hosted (RTX 4090) si >500 usuarios
   → API para prototipado rápido
   → Considera fine-tuning de modelo pequeño

4. DATA SYNTHESIS (volumen masivo):
   → SIEMPRE self-hosted
   → Break-even en días, no meses
   → Considera GPU cloud para bursts
""")

# Tabla de decisión
print("\n" + "="*70)
print("TABLA DE DECISIÓN RÁPIDA:")
print("="*70)
print(f"{'Tokens/Mes':<20} {'Usuarios':<15} {'Recomendación':<30}")
print("-"*70)
print(f"{'<10M':<20} {'<1K':<15} {'API (cualquiera)':<30}")
print(f"{'10-50M':<20} {'1K-5K':<15} {'API o RTX 4090':<30}")
print(f"{'50-200M':<20} {'5K-20K':<15} {'RTX 4090 (self-hosted)':<30}")
print(f"{'>200M':<20} {'>20K':<15} {'A100 o Cloud GPU':<30}")
```

---

## Referencias

- OpenAI. [Pricing](https://openai.com/pricing). OpenAI.
- Anthropic. [Claude API Pricing](https://www.anthropic.com/pricing). Anthropic.
- Together AI. [Pricing](https://www.together.ai/pricing). Together AI.

