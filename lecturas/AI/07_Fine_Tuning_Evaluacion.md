# Lectura 7: Fine-Tuning y Evaluación

## Contexto
Aprenderás técnicas de fine-tuning desde full fine-tuning hasta métodos eficientes (LoRA, QLoRA). Dominarás evaluación rigurosa y detección de benchmark contamination.

## Introducción

Un modelo pre-entrenado es un punto de partida. Para optimizarlo para tu tarea específica, necesitas **fine-tuning**. ¿Pero cómo evalúas si la mejora es real o solo memoria memorizada?

Esta lectura cubre fine-tuning (full, LoRA, QLoRA), evaluación rigurosa y métricas confiables.

---

## Parte 1: Full Fine-Tuning

### La Idea

```
Modelo pre-entrenado: Entrenado en 1 billón de tokens generales
Tu tarea específica: 10,000 ejemplos de instrucciones en tu dominio

Full Fine-Tuning:
  - Toma el modelo pre-entrenado
  - Continúa el entrenamiento con TUS datos
  - Actualiza TODOS los pesos (por eso "full")
```

### Proceso

```
Paso 1: Prepare datos
  ```
  [
    {"instruction": "Traduce a español", "input": "hello", "output": "hola"},
    {"instruction": "Traduce a español", "input": "good morning", "output": "buenos días"},
    ...
  ]
  ```

Paso 2: Configura entrenamiento
  ```
  modelo = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

  optimizer = AdamW(modelo.parameters(), lr=1e-5)

  para epoch in range(3):  # 3 pasadas por los datos
    para batch in data_loader:
      logits = modelo(batch["input_ids"])
      loss = cross_entropy(logits, batch["labels"])
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
  ```

Paso 3: Evalúa
  - Genera respuestas en datos de validación
  - Compara con respuestas esperadas
  - Calcula métricas

Paso 4: Guarda modelo
  - Guardas los NUEVOS pesos (7B parámetros)
  - ~28 GB en float32, ~7 GB en INT8
```

### Costos de Full Fine-Tuning

```
Hardware requerido:
  Llama 7B: GPU A100 (80GB) mínimo
  Llama 70B: 8 x A100

Datos: 10,000 ejemplos * 1,000 tokens = 10M tokens
Tiempo: ~30 minutos en 8x A100
Costo: ~$50-100 (AWS p3 instances)

Ventaja: Máxima adaptación a tu dominio
Desventaja: Muy caro, requiere mucho dato, riesgo de overfitting
```

### Cuándo Usar Full Fine-Tuning

```
SÍ si:
  ✓ Dominio muy específico (medicina, derecho, muy diferente del general)
  ✓ Tienes 50,000+ ejemplos
  ✓ Presupuesto disponible
  ✓ Necesitas máxima precisión

NO si:
  ✗ Tienes < 10,000 ejemplos (overfitting probable)
  ✗ Presupuesto limitado
  ✗ Dominio similar al general
```

**Advertencia: Catastrophic Forgetting** - Full fine-tuning puede hacer que el modelo "olvide" capacidades generales. Mitigación: mezclar datos generales (20%) con específicos (80%), o usar LoRA.

---

## Parte 2: PEFT - Parameter Efficient Fine-Tuning

![AI_07_01_LoRA_Descomposicion.jpeg](../../assets/imagenes/AI_07_01_LoRA_Descomposicion.jpeg)

> **Descomposición de rango bajo en LoRA**
>
> LoRA reduce drásticamente el número de parámetros entrenables descomponiendo las actualizaciones de peso como productos de dos matrices de rango bajo (A y B). Este diagrama ilustra cómo una matriz grande se aproxima eficientemente mediante esta descomposición, permitiendo fine-tuning en hardware limitado sin sacrificar significativamente la calidad.

### El Problema de Full Fine-Tuning

```
Llama 7B: 7,000,000,000 parámetros = 28 GB en float32
Fine-tuning requiere guardar:
  - Pesos originales
  - Gradientes (28 GB)
  - Optimizer states (28 GB más)
  - Pesos actualizados
  Total: ~112 GB de memoria

Solo un 1% de A100 puede hacer esto
```

### La Solución: PEFT (LoRA)

En lugar de actualizar TODOS los pesos, actualiza solo una **matriz de rango bajo**:

```
Peso original (forma 4096 x 4096): 16M parámetros
  W

LoRA: Descompone update como producto de matrices pequeñas
  W' = W + ΔW
  ΔW = A (4096 x r) @ B (r x 4096)

donde r es rank pequeño (típicamente 4, 8, 16)

Si r = 8:
  A: 4096 x 8 = 32K parámetros
  B: 8 x 4096 = 32K parámetros
  Total: 64K parámetros (en lugar de 16M)

Ratio: 64K / 16M = 0.4% de parámetros

Con LoRA en todas las capas:
  Total parámetros trainables: ~1-2% del modelo original
  Costo de memoria: 1/50 del full fine-tuning
```

### LoRA en Práctica

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=8,                           # Rank bajo
    lora_alpha=32,                 # Escala de actualización
    target_modules=["q_proj", "v_proj"],  # Qué capas actualizar
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

modelo = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
modelo = get_peft_model(modelo, config)

# Ahora entrenar como normal, pero:
# - Memoria: 10x más eficiente
# - Tiempo: 5x más rápido
# - Guardas solo: 1-2% de parámetros (300 MB en lugar de 28 GB)
```

### QLoRA: Aún Más Eficiente

```
LoRA: Modelo en float16 (13 GB) + LoRA en float16 (100 MB)
      Requiere: 13 GB memoria

QLoRA: Modelo cuantizado INT4 (1.5 GB) + LoRA en float16 (100 MB)
       Requiere: 2 GB memoria

Trade-off:
  - Pérdida de precisión por cuantización: ~1-2%
  - Mejor eficiencia: 6.5x menos memoria
  - Mismo resultado de fine-tuning
```

### Comparación

```
Método              Memoria    Tiempo    Guardas   Precisión
─────────────────────────────────────────────────────────────
Full Fine-Tuning    100 GB     1.0x      28 GB     100%
LoRA                13 GB      0.2x      300 MB    99.8%
QLoRA               2 GB       0.15x     300 MB    99%
```

![Full Fine-Tuning vs LoRA vs QLoRA](./diagrams/fullft_lora_qlora.png)

> **Comparación Visual: Full Fine-Tuning vs LoRA vs QLoRA**
>
> El diagrama ilustra cómo cada método maneja los pesos del modelo. Full FT actualiza toda la matriz W (16M parámetros/capa), requiriendo ~100GB de RAM. LoRA congela W y añade dos matrices pequeñas A×B de rango r=8, reduciendo los parámetros entrenables al 0.4% y la RAM a ~13GB. QLoRA va más lejos: cuantiza W a INT4 (4-bit, ~1.5GB/capa) antes de aplicar LoRA, bajando la RAM total a ~2GB y habilitando fine-tuning en GPUs de consumo, con solo ~1% de pérdida de precisión.

---

![Arquitectura LoRA en detalle](./diagrams/lora_architecture.png)

> **Arquitectura LoRA: Descomposición de Bajo Rango**
>
> LoRA reemplaza la actualización ΔW (d×k parámetros) por el producto de dos matrices pequeñas: B (d×r) y A (r×k), donde r≪min(d,k). Durante el entrenamiento solo se actualizan A y B; los pesos originales W₀ quedan congelados. En inferencia, la adaptación se fusiona: W = W₀ + BA, sin overhead adicional.

## Parte 3: Instruction Tuning

Un tipo especial de fine-tuning enfocado en seguir instrucciones:

### Dataset de Instruction Tuning

```
{
  "instruction": "Clasifica el sentimiento de este review",
  "input": "El producto es excelente, muy satisfecho",
  "output": "positivo"
}

vs

{
  "instruction": "Traduce al español",
  "input": "hello",
  "output": "hola"
}

vs

{
  "instruction": "Resuelve esta ecuación",
  "input": "2x + 3 = 7",
  "output": "x = 2"
}
```

### Ventaja de Instruction Tuning

```
Sin instruction tuning:
  Entrada: "Traduce al español: hello"
  Salida: "El hello es un saludo"  ← Modelo no entiende que traduce

Con instruction tuning:
  Entrada: "Traduce al español: hello"
  Salida: "hola"  ← Modelo entiende la instrucción
```

Hace al modelo general sobre **tipos de instrucciones**, no solo una tarea.

---

## Parte 4: Evaluación - El Desafío

![AI_07_03_Quality_vs_Constraint_Strictness.jpeg](../../assets/imagenes/AI_07_03_Quality_vs_Constraint_Strictness.jpeg)

> **Evaluación de calidad versus rigor de restricciones**
>
> En fine-tuning, la evaluación debe considerar múltiples dimensiones de rendimiento. Este diagrama ilustra la relación entre diferentes criterios de evaluación y cómo el énfasis en calidad pura puede diferir del que considere restricciones adicionales, informando decisiones sobre qué modelo fine-tuned es mejor para una aplicación específica.

### El Problema: Cómo Saber si Mejoraste

```
Pre-entrenado (GPT):
  Prompt: "¿Cuál es 2 + 2?"
  Salida: "4"

Fine-tuned en mi dominio:
  Prompt: "¿Cuál es 2 + 2?"
  Salida: "El resultado es 4"

¿Mejora? Depende...
```

### Problema 1: Benchmark Contamination

```
Entrenaste el modelo en datos que podrían incluir el benchmark
Entonces el modelo "recuerda" la respuesta, no aprendió

Ejemplo:
  Fine-tuning data accidentalmente incluye ejemplos de SQuAD (benchmark QA)
  Evalúas en SQuAD
  Resultado: 95% (pero es memorización, no comprensión)

Cómo prevenirlo:
  ✓ Verifica que fine-tuning data NO incluya benchmark
  ✓ Usa benchmarks nuevos, no publicados
  ✓ Examina ejemplos de fine-tuning manualmente
```

### Problema 2: Overfitting

```
Pequeño fine-tuning set (100 ejemplos):
  Training loss: 0.01 (muy bajo, modelo memorizó)
  Validation loss: 2.5 (muy alto, no generaliza)

Problema: Modelo funciona en training pero falla en producción
```

---

## Parte 5: Evaluación Rigurosa

### Métricas Automáticas

#### BLEU (Machine Translation)

```
Traducción esperada: "The cat is black"
Traducción generada: "The cat is black"
BLEU: 100%

Traducción esperada: "The cat is black"
Traducción generada: "The black cat"
BLEU: 50% (n-gramas parcialmente coinciden)
```

Propiedad: Simple pero no captura semántica.

#### ROUGE (Resumen)

```
Resumen de referencia: "El producto es excelente"
Resumen generado: "El producto es muy bueno"

ROUGE-L (longest common subsequence):
  LCS: "El producto es"
  ROUGE-L = 3 / 5 = 60%
```

Propiedad: Mejor para resúmenes, pero aún imperfecto.

#### METEOR, CIDEr, etc.

Variaciones de las anteriores, cada una captura algo diferente.

### LLM-as-Judge

Una alternativa moderna: **usa otro LLM para evaluar**

```
Prompt de evaluación:
  ```
  Evalúa la siguiente respuesta en una escala 1-10.

  Pregunta: ¿Cuál es la capital de Francia?
  Respuesta esperada: París
  Respuesta generada: Francia tiene muchas ciudades importantes, París siendo la capital.

  Criterios:
  - Exactitud (es correcta?)
  - Completitud (contiene toda la información?)
  - Claridad (es clara?)

  Puntuación: ?
  ```

Ventaja: Comprende semántica, es flexible
Desventaja: Sesgo hacia modelo evaluador, requiere ejecución LLM
```

### Metrics Específicas por Dominio

```
QA (Question Answering):
  - Exact Match (EM): ¿Es la respuesta exacta?
  - F1 Score: Overlap de palabras

Clasificación:
  - Accuracy: % respuestas correctas
  - F1 (macro/micro): Balance entre precisión y recall

Generación:
  - Perplexity: Confianza del modelo en respuesta
  - BLEU/ROUGE/METEOR
  - Human evaluation (gold standard)
```

---

## Parte 6: Pass@k Metrics

![AI_07_02_Pass_k_Metricas_Evaluacion.jpeg](../../assets/imagenes/AI_07_02_Pass_k_Metricas_Evaluacion.jpeg)

> **Métricas Pass@k para evaluación de generación**
>
> La métrica Pass@k reconoce que algunos problemas tienen múltiples soluciones válidas. Este diagrama muestra cómo evaluar la probabilidad de que al menos una de k generaciones sea correcta, proporcionando una evaluación más realista de la capacidad del modelo comparada con métricas de una sola generación.

Métrica importante para generación de código y problemas complejos:

### Idea

```
Una pregunta puede tener múltiples respuestas correctas.
En lugar de 1 intento, generas k intentos.
Alguno es correcto?

Pass@1: Generaste 1 respuesta, fue correcta?
Pass@5: Generaste 5 respuestas, al menos 1 fue correcta?
Pass@100: Generaste 100 respuestas, al menos 1 fue correcta?
```

### Ejemplo

```
Pregunta: "Escribe una función que ordena un array"

Pass@1 = 70%   (de 100 preguntas, 70 veces el primer intento fue correcto)
Pass@5 = 85%   (de 100 preguntas, 85 veces alguno de los 5 intentos fue correcto)
Pass@100 = 92% (de 100 preguntas, 92 veces alguno de 100 intentos fue correcto)
```

### Fórmula

```
Pass@k = 1 - (N-c)! / ((N-k)! * (N-c-k+1)!)

donde:
  N = número total de intentos generados
  c = número de intentos correctos
  k = número que usamos para computar métrica

O más simplemente (si puedes generar muchos intentos):
  Pass@k ≈ 1 - (1 - Pass@1)^k
```

---

## Parte 7: Benchmark Contamination y Cómo Evitarlo

![AI_07_04_Benchmark_Contamination_Detection.jpeg](../../assets/imagenes/AI_07_04_Benchmark_Contamination_Detection.jpeg)

> **Detección y análisis de contaminación de benchmarks**
>
> La contaminación de benchmarks es un riesgo crítico que puede llevar a sobrestimar significativamente el rendimiento del modelo. Este diagrama ilustra los diferentes tipos de contaminación y métodos para detectarlos, desde comparación directa de conjuntos hasta análisis de similitud semántica, asegurando que las evaluaciones reflejen capacidades genuinas.

### Tipos de Contaminación

```
Tipo 1: Datos de fine-tuning incluyen benchmark
  Problema: Modelo memorizó respuestas
  Solución: Excluir benchmark de fine-tuning

Tipo 2: Pre-entrenamiento vio benchmark
  Problema: Modelo ya conocía respuestas
  Solución: Entrenar de cero (impractical) o usar benchmarks nuevos

Tipo 3: Leakage por semejanza
  Problema: Fine-tuning data es muy similar al benchmark
  Solución: Análisis cuidadoso de semejanza
```

### Cómo Detectar

```
Test 1: Analiza conjunto de fine-tuning
  ```
  for example in fine_tuning_set:
    if example in benchmark:
      print("¡CONTAMINACIÓN!")
  ```

Test 2: Verifica por hash
  ```
  hash(fine_tuning_text) == hash(benchmark_text)
  ```

Test 3: Simil coseno
  ```
  similarity(embedding(fine_tuning_text), embedding(benchmark_text))
  if similarity > 0.9:
    print("¡Posible contaminación!")
  ```
```

---

## Parte 8: Recomendaciones Prácticas

![AI_07_05_Correctness_Rate_Heatmap.jpeg](../../assets/imagenes/AI_07_05_Correctness_Rate_Heatmap.jpeg)

> **Heatmap de tasas de corrección por configuración**
>
> Los resultados de fine-tuning varían según múltiples dimensiones: hiperparámetros, configuraciones de generación, y características del benchmark. Este heatmap permite identificar rápidamente qué combinaciones de configuración producen los mejores resultados, facilitando optimización y comparación sistemática de enfoques de fine-tuning.

### Flujo de Fine-Tuning y Evaluación

```
1. Prepara datos
   ├─ 80% training
   ├─ 10% validation
   └─ 10% test

2. Fine-tune (LoRA si presupuesto limitado)
   ├─ Monitorea training loss
   ├─ Detén cuando validation loss empiece a subir (early stopping)
   └─ Guarda mejor checkpoint

3. Evalúa en test set
   ├─ Usa métricas automáticas
   ├─ Evalúa manualmente 100-200 ejemplos
   └─ Busca patterns de error

4. Compara vs baseline
   ├─ Modelo pre-entrenado sin fine-tuning
   ├─ Otro modelo especializado
   └─ Verifica que mejora es significativa

5. Valida en datos nuevos
   ├─ Datos que el modelo NUNCA vio
   ├─ Simula distribución de producción
   └─ Mide drop en rendimiento (es esperado)

6. Despliega con monitoreo
   ├─ Monitorea métricas en producción
   ├─ Recolecta feedback de usuarios
   └─ Reentrana si rendimiento baja
```

### Checklists Finales

```
Antes de confiar en el modelo:

Benchmark Contamination:
  ☐ Verificaste que fine-tuning ≠ benchmark?
  ☐ Validaste en datos que nunca viste?
  ☐ Usaste evaluación humana en sample?

Overfitting:
  ☐ Training loss ≠ Validation loss?
  ☐ Early stopping implementado?
  ☐ Data augmentation considerado?

Fairness y Bias:
  ☐ Rendimiento consistente en subgrupos?
  ☐ Buscaste ejemplos donde falla?
  ☐ Documentaste limitaciones?
```

---

## Reflexión y Ejercicios

### Preguntas para Reflexionar:

1. **Full vs LoRA:** ¿Cuándo la pérdida de precisión de LoRA es inaceptable?

2. **Evaluación:** ¿Qué es mejor, BLEU score de 0.8 u aprobación humana del 70%?

3. **Contaminación:** ¿Cómo sabrías si contaminaste sin intención?

### Ejercicios Prácticos:

1. **Fine-tuning Budget:**
   ```
   Tienes presupuesto de $500 para fine-tuning.
   Opciones:
   a) Full fine-tuning en 4x A100 ($200)
   b) LoRA en 1x A100 ($50)
   c) QLoRA en 1x 24GB GPU ($10)

   ¿Cuál elegirías?
   ¿Cómo usarías el presupuesto restante?
   ```

2. **Dataset Split:**
   ```
   Tienes 10,000 ejemplos etiquetados manualmente.
   ¿Cuál es mejor split?

   a) 8000 train, 1000 val, 1000 test
   b) 9000 train, 500 val, 500 test
   c) 7000 train, 2000 val, 1000 test

   ¿Por qué?
   ```

3. **Benchmark Analysis:**
   ```
   Tu modelo:
   - SQuAD (benchmark público): 90% F1
   - Preguntas internas: 65% F1

   ¿Posible contaminación? ¿Qué investigarías?
   ```

4. **Reflexión escrita (350 palabras):** "La métrica perfecta es evaluación humana, pero es cara. ¿Cuáles son las 3 métricas automáticas más importantes para tu dominio? ¿Cómo verificarías que correlacionan con calidad humana?"

---

## Puntos Clave

- **Full Fine-Tuning:** Actualiza todos los pesos; caro pero máxima adaptación
- **LoRA:** Actualiza matriz de rango bajo (~1% parámetros); 50x más eficiente
- **QLoRA:** LoRA + cuantización; cabe en GPU de consumidor
- **Instruction Tuning:** Fine-tuning en pares (instrucción, salida)
- **Benchmark Contamination:** Riesgo crítico de sobreestimar rendimiento
- **LLM-as-Judge:** Evalúa con otro LLM; más semántica que métricas
- **Pass@k:** Generas k respuestas; alguna es correcta?
- **Flujo:** Entrenar en train, valida en validation, test en datos nuevos, despliega con monitoreo

