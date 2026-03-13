# Power Analysis y Diseño Experimental
## Semana 4 - Estadística para Generación de Kernels GPU

Aquí es donde la planificación científica se convierte en poder estadístico. Antes de ejecutar tu experimento, necesitas saber: **¿Cuántas muestras necesito para detectar un efecto que me importa?** Eso es power analysis.

## Conceptos Fundamentales del Diseño

### Variables Independientes y Dependientes

Una **variable independiente (IV)** es lo que tú manipulas. En tu proyecto:
- IV: Tipo de decoding (baseline vs. con restricciones gramaticales)
- IV: Temperatura del LLM (0.0 vs. 0.5)
- IV: Arquitectura del kernel (simple vs. compleja)

Una **variable dependiente (DV)** es lo que mides como resultado:
- DV: Número de kernels válidos (compilables)
- DV: Iteraciones hasta convergencia
- DV: Tiempo de ejecución
- DV: Consumo de memoria

**Variables de control** son factores que intentas mantener constantes:
- Mismo LLM base (ej. GPT-3.5)
- Mismo hardware
- Mismos prompts de instrucción
- Mismo conjunto de kernels de prueba

Controlar variables de confusión mejora tu capacidad de atribuir resultados a tu IV.

## Diseños Experimentales: Las Cuatro Configuraciones


![**Figura 1:** Proceso del diseño experimental desde hipótesis hasta conclusiones.](diagrams/experimental_design.png)

***Figura 1:** Proceso del diseño experimental desde hipótesis hasta conclusiones.*

### Configuración A: Between-Subjects, Simple

Diferentes participantes/muestras experimentan diferentes condiciones.

```
Grupo 1 (Baseline): 50 kernels generados con baseline
↓ Mide: tasa de validez

Grupo 2 (Restricciones): 50 kernels generados con restricciones
↓ Mide: tasa de validez

Compare grupos
```

**Ventajas**: Simple, menos riesgo de sesgo por aprendizaje
**Desventajas**: Necesitas más muestras, mayor variación entre grupos

### Configuración B: Within-Subjects, Simple

Los mismos kernels se generan con ambos métodos.

```
Set de 50 kernels
↓
Genera con Baseline → mide: validez A
↓
Genera con Restricciones → mide: validez B
↓
Compara A vs B para cada kernel
```

**Ventajas**: Controlas variabilidad del kernel, necesitas menos muestras
**Desventajas**: Riesgo de orden/aprendizaje, el segundo intento puede ser diferente

### Configuración C: Múltiples Factores, Between-Subjects

```
         Restricciones: No | Restricciones: Sí
Temp=0   Grupo 1 (n=20)  | Grupo 3 (n=20)
Temp=0.5 Grupo 2 (n=20)  | Grupo 4 (n=20)

Total: 80 muestras
```

Mides efectos de:
- Restricciones (promedio Grupo 1&2 vs. Grupo 3&4)
- Temperatura (promedio Grupo 1&3 vs. Grupo 2&4)
- **Interacción**: ¿El efecto de restricciones depende de temperatura?

**Ventaja**: Eficiente, obtiene múltiples efectos
**Desventaja**: Más complejo de analizar

### Configuración D: Múltiples Factores, Within-Subjects

Los mismos kernels experimentan todas las combinaciones.

```
50 kernels
↓
Cada kernel se genera 4 veces:
- Baseline, Temp=0
- Baseline, Temp=0.5
- Restricciones, Temp=0
- Restricciones, Temp=0.5
```

**Ventaja**: Máximo control, necesitas menos kernels
**Desventaja**: Riesgo de orden/fatiga, complejidad analítica

## Power Analysis

**El poder** es tu capacidad de detectar un efecto real cuando existe.

```
Poder = 1 - β

β = probabilidad de Error Tipo II (no detectar cuando existe)
```

Típicamente buscamos poder ≥ 0.80, significando 80% de probabilidad de detectar un efecto real.

### Factores que Afectan el Poder

1. **Tamaño del efecto (d)**: Qué tan grande es la diferencia que esperas
2. **Tamaño muestral (n)**: Cuántas observaciones tienes
3. **Nivel de significancia (α)**: Tu umbral de p-valor
4. **Dirección de la prueba**: Una cola vs. dos colas

### Ejemplo: Calculando Tamaño Muestral Requerido

Quieres comparar tasa de validez:
- Baseline: esperas 75% validez
- Con restricciones: esperas 85% validez
- Diferencia esperada: 10 puntos porcentuales
- Quieres poder = 0.80, α = 0.05

Usando fórmulas o software (G*Power):

```
n = 240 por grupo (480 total)
```

Significa que necesitas 240 intentos con baseline y 240 con restricciones.

Si n = 100 por grupo:
```
Poder resultante ≈ 0.45
```

Muy bajo. Solo 45% de probabilidad de detectar tu efecto de 10%.

**Insight**: Detectar pequeñas diferencias (2-5%) requiere muestras grandes. Diferencias grandes (15-20%) requieren menos.

## Tamaño del Efecto

El tamaño del efecto cuantifica la magnitud de una diferencia.

### Cohen's d (para variables continuas)

```
d = (μ₁ - μ₂) / σ_pooled

Donde σ_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]
```

Interpretación:
- d = 0.2: efecto pequeño
- d = 0.5: efecto mediano
- d = 0.8: efecto grande

Ejemplo: Baseline toma 5.0s ± 1.2s, Restricciones toma 4.2s ± 1.1s

```
d = (5.0 - 4.2) / √[((29×1.2² + 29×1.1²) / 58)]
  = 0.8 / 1.15
  ≈ 0.70 (mediano a grande)
```

### Diferencia Proporcional (para conteos)

Para tasa de validez:

```
p₁ = 0.75 (75% baseline)
p₂ = 0.85 (85% restricciones)
h = 2 × arcsin(√p₂) - 2 × arcsin(√p₁)
  ≈ 0.42 (mediano)
```

## Órdenes y Contrabalanceo

En diseños within-subjects, el **orden importa**.

Si ejecutas "Baseline primero, luego Restricciones", podrías ver:
- **Efecto de aprendizaje**: El segundo intento se beneficia del primero
- **Efecto de fatiga**: El segundo intento es peor porque el modelo está "cansado"
- **Efecto de sesgo**: El modelo recuerda lo que hizo antes

### Solución: Contrabalanceo

```
Mitad de participantes: Baseline → Restricciones
Otra mitad: Restricciones → Baseline
```

Esto distribuye cualquier efecto de orden entre condiciones.

Para múltiples condiciones (>2), usa **cuadrado latino**:

```
Participante 1: A → B → C
Participante 2: B → C → A
Participante 3: C → A → B
```

Cada condición aparece en cada posición exactamente una vez.

## Validez Interna vs. Externa

### Validez Interna

¿Tus resultados se deben realmente a tu variable independiente, o a algo más?

**Amenazas**:
- **Confusión**: Variable no controlada correlaciona con IV
- **Selección sesgada**: Cómo eliges muestras afecta resultados
- **Maduración**: Los participantes cambian con el tiempo
- **Artefactos experimentales**: Comportamiento artificial en el laboratorio

En tu proyecto:
- ¿Usas el mismo LLM/versión para ambas condiciones?
- ¿Los kernels de prueba son representativos?
- ¿Hay cambios en el código entre ejecuciones que podrían confundir?

### Validez Externa

¿Tus resultados generalizan más allá de tu experimento?

**Amenazas**:
- **Especificidad de población**: Solo estudiantes en la clase
- **Especificidad de contexto**: Solo en un sistema específico
- **Especificidad de operacionalización**: Solo con esta implementación de restricciones

En tu proyecto:
- ¿Tus restricciones son generales o específicas para GPUs?
- ¿Generalizaría a otros LLMs?
- ¿Generalizaría a otros tipos de kernels?

## Plan Experimental Robusto

Aquí hay un plan general para tu proyecto:

```
1. ESPECIFICA
   - IV: Decoding method (baseline vs. restricciones)
   - DV: Tasa de validez, iteraciones, tiempo
   - Controles: Mismo LLM, mismo dataset, mismo hardware
   - α = 0.05, poder objetivo = 0.80

2. POWER ANALYSIS
   - Esperas diferencia de 10% en validez
   - Usas G*Power: necesitas n=240 por grupo

3. DISEÑO
   - Within-subjects (mismo kernels ambos métodos)
   - Contrabalanceo: mitad baseline primero, mitad restricciones primero
   - Total: 240 kernels × 2 métodos = 480 ejecuciones

4. RECOLECTA DATOS
   - Controla orden
   - Registra todas las variables de interés

5. ANALIZA
   - Prueba de hipótesis de dos muestras pareadas
   - Reporta p-valor, efecto tamaño, IC 95%

6. INTERPRETA
   - ¿Significante y prácticamente importante?
```

## Ejercicios y Reflexión

### Ejercicio 1: Identifica Variables
Para tu proyecto, especifica:
- ¿Cuáles son tus IVs?
- ¿Cuáles son tus DVs?
- ¿Qué variables de control necesitas?

### Ejercicio 2: Elige Diseño
¿Cuál configuración es mejor para tu proyecto (A/B/C/D)? Justifica:
- ¿Ventajas del tuyo?
- ¿Desventajas?
- ¿Cómo controlarías confusores?

### Ejercicio 3: Power Analysis
Supongamos:
- Esperas Cohen's d = 0.6
- α = 0.05, dos colas
- Poder = 0.80

¿Cuál es el tamaño muestral requerido?

(Usa online calculator de G*Power si no lo sabes)

### Ejercicio 4: Validez
Para tu diseño experimental:
- ¿Cuáles son 3 amenazas a validez interna? ¿Cómo las mitigas?
- ¿Cuáles son 3 amenazas a validez externa?

### Reflexión
1. **Trade-offs**: Si aumentas poder a 0.95 (vs. 0.80), ¿cómo cambia tu n requerido? ¿Vale la pena?
2. **Efectos prácticos vs. estadísticos**: Si encuentras p<0.001 pero el efecto es d=0.15, ¿es tu método mejor realmente?
3. **Orden**: ¿Cómo planificarías contrabalanceo en tu proyecto? ¿Es feasible?

---

**Próxima semana**: Veremos cómo hacer que nuestros experimentos sean reproducibles controlando la aleatoriedad y documentando bien.
