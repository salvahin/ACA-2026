# Plantilla: Metodología

> **Instrucciones:** Reemplaza cada `[elemento]`. La sección de Metodología debe ser reproducible: otro investigador debería poder ejecutar tus experimentos con solo leer esta sección. Borra estas instrucciones antes de entregar.

---

## 3. Metodología

### 3.1 Descripción del Sistema Propuesto

[Describe en 1–2 párrafos **qué hace tu sistema**, no cómo lo implementaste. Incluye un diagrama de pipeline si es posible.]

```
[DIAGRAMA ASCII o referencia a figura]
Ejemplo:
Prompt → LLM → Grammar FSM → Token Filter → Valid Kernel
```

**Componentes principales:**

| Componente | Función | Librería/Herramienta |
|---|---|---|
| [Nombre] | [Qué hace] | [torch / xgrammar / triton] |
| [Nombre] | [Qué hace] | [...] |
| [Nombre] | [Qué hace] | [...] |

---

### 3.2 Dataset y Configuración Experimental

**Dataset de evaluación:**
- **Fuente:** [¿De dónde vienen los datos? (sintético, benchmark público, custom)]
- **Tamaño:** [N ejemplos de entrenamiento] / [M ejemplos de evaluación]
- **Formato:** [Descripción de cada entrada/salida]
- **Split:** [X% train / Y% val / Z% test] — justificación: [...]

**Modelos base evaluados:**

| Modelo | Parámetros | Fuente |
|---|---|---|
| [DeepSeek-Coder-7B] | 7B | HuggingFace |
| [CodeLlama-13B] | 13B | Meta / HuggingFace |
| [Tu modelo fine-tuned] | [N]B | Este trabajo |

**Hardware:**
- GPU: [NVIDIA A100 / T4 / etc.], VRAM: [N]GB
- RAM: [N]GB
- SO: [Ubuntu 22.04 / etc.]

---

### 3.3 Métricas de Evaluación

Define cada métrica con su fórmula o referencia:

| Métrica | Descripción | Fórmula | Rango |
|---|---|---|---|
| Validez Sintáctica | % kernels que parsean sin error | `valid / total × 100` | 0–100% |
| Correctitud Funcional | % kernels que producen el output correcto | `correct / total × 100` | 0–100% |
| [Métrica 3] | [...] | [...] | [...] |

**Estadística inferencial:** Usaremos [t-test / Wilcoxon / ANOVA] con α=0.05 para comparar condiciones. Ver módulo Stats L03.

---

### 3.4 Procedimiento Experimental

```
Paso 1: Preparar prompts de evaluación (N=120)
  ↓
Paso 2: Generar kernels — condición baseline (sin restricciones)
  ↓
Paso 3: Generar kernels — condición experimental (con grammar constraints)
  ↓
Paso 4: Validar sintaxis con parser Triton
  ↓
Paso 5: Ejecutar tests funcionales en GPU
  ↓
Paso 6: Aplicar prueba estadística para comparar condiciones
  ↓
Paso 7: Analizar ejemplos de error cualitativamente
```

**Reproducibilidad:**
- Seeds fijas: `torch.manual_seed(42)`, `np.random.seed(42)`
- Temperatura de generación: `T=[valor]`
- Código disponible en: [URL de repositorio]

---

### 3.5 Hiperparámetros

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Beam width | [3] | [Balance calidad/velocidad] |
| Max tokens | [512] | [Longitud máxima de kernels] |
| Temperature | [0.7] | [Mayor creatividad sin ruido excesivo] |
| [Otro] | [...] | [...] |

---

## Checklist de Metodología

- [ ] Alguien que lea esta sección puede reproducir exactamente tus experimentos
- [ ] Todas las métricas están definidas con fórmula o referencia
- [ ] El dataset está descrito (fuente, tamaño, formato)
- [ ] Los hiperparámetros están documentados
- [ ] La prueba estadística elegida está justificada (ver `stats/uso-en-el-proyecto.md`)
- [ ] Se menciona cómo se controla la aleatoriedad (seeds)

---

*Este archivo es parte del módulo Research del curso ACA-2026.*
