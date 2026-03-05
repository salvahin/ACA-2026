# Plantilla: Tabla de Experimentos

> **Instrucciones:** Usa esta tabla para registrar cada experimento ANTES de ejecutarlo (disciplina de pre-registro) y llena los resultados después. Esto evita el "HARKing" (Hypothesizing After Results are Known). Borra estas instrucciones antes de entregar.

---

## Tabla de Experimentos

| ID | Hipótesis | Condición | N | Métrica Principal | Resultado Esperado | Resultado Real | p-valor | Efecto |
|----|-----------|-----------|---|-------------------|-------------------|----------------|---------|--------|
| E1 | [Grammar constraints aumentan validez sintáctica] | Baseline vs Grammar | [50] | Validez Sintáctica (%) | [↑ >20pp] | [...] | [...] | [...] |
| E2 | [Efecto de beam_width en calidad] | bw=1 vs 3 vs 5 | [30] | Correctitud Funcional (%) | [bw=3 > bw=1] | [...] | [...] | [...] |
| E3 | [...] | [...] | [...] | [...] | [...] | [...] | [...] | [...] |

> **pp** = puntos porcentuales. **bw** = beam width. **N** = runs por condición.

---

## Registro Detallado por Experimento

### Experimento E1

**Hipótesis:** Agregar grammar constraints al decoding aumenta la validez sintáctica de los kernels generados en al menos 20 puntos porcentuales sobre el baseline no restringido.

**Variables:**
- **Independiente:** Presencia/ausencia de grammar constraints
- **Dependiente:** % de kernels válidos (pasan el parser Triton)
- **Controladas:** Modelo (DeepSeek-Coder-7B), temperatura (0.7), prompts (mismos 50)

**Protocolo:**
1. Generar 50 kernels con modelo baseline (sin constraints)
2. Generar 50 kernels con modelo + XGrammar constraints
3. Parsear todos los kernels con `triton.compile()` o parser equivalente
4. Aplicar prueba chi-cuadrado o z-test de proporciones (α=0.05)

**Resultado registrado (fecha: [DD/MM/AAAA]):**

| Condición | N válidos / N total | % válidos |
|-----------|---------------------|-----------|
| Baseline  | [...] / 50          | [...]%    |
| Grammar   | [...] / 50          | [...]%    |

```python
# Código de análisis estadístico (ver stats/uso-en-el-proyecto.md)
from scipy import stats
import numpy as np

n_baseline_valid = ...   # rellenar
n_grammar_valid  = ...   # rellenar
N = 50

# Test de proporciones (z-test)
p1 = n_baseline_valid / N
p2 = n_grammar_valid  / N
p_pool = (n_baseline_valid + n_grammar_valid) / (2 * N)

z = (p2 - p1) / np.sqrt(p_pool * (1 - p_pool) * (1/N + 1/N))
p_val = stats.norm.sf(abs(z)) * 2  # two-tailed

print(f"Baseline: {p1*100:.1f}%  Grammar: {p2*100:.1f}%")
print(f"z={z:.3f}, p={p_val:.4f}")
print(f"Significativo: {'Sí' if p_val < 0.05 else 'No'}")
```

**Interpretación:** [Escribe aquí qué significa el resultado en términos del dominio. ¿Apoya la hipótesis?]

---

### Experimento E2

**Hipótesis:** [...]

*(Repite la estructura de E1 para cada experimento)*

---

## Checklist de Rigor Experimental

- [ ] Cada hipótesis fue registrada **antes** de ejecutar el experimento
- [ ] El tamaño de muestra N fue calculado o justificado (power analysis, ver Stats L07)
- [ ] Se especificó α=0.05 antes de correr las pruebas
- [ ] El test estadístico elegido es apropiado para el tipo de datos
- [ ] Los resultados negativos también están documentados
- [ ] Las semillas de aleatoriedad están fijadas para reproducibilidad

---

## Resumen Estadístico Final

| Experimento | Hipótesis apoyada | p-valor | Tamaño de efecto | Conclusión |
|-------------|-------------------|---------|------------------|------------|
| E1 | [Sí/No/Parcialmente] | [...] | Cohen's d=[...] | [...] |
| E2 | [...] | [...] | [...] | [...] |

---

*Este archivo es parte del módulo Research del curso ACA-2026. Ver también `stats/uso-en-el-proyecto.md`.*
