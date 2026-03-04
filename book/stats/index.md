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

# Módulo de Estadística
## Grammar-Constrained GPU Kernel Generation - ACA

Este módulo proporciona 10 lecturas sobre estadística diseñadas específicamente para estudiantes investigando la generación de kernels GPU con restricciones gramaticales.

### Estructura del Curso

#### Semana 1: Fundamentos de Probabilidad
**01-Fundamentos-Probabilidad.md** (1,433 palabras)
- Espacios muestrales y eventos
- Axiomas de probabilidad
- Probabilidad condicional y Teorema de Bayes
- Variables aleatorias discretas
- Función de masa de probabilidad (PMF)
- Valor esperado y varianza
- Ejercicios contextualizados al proyecto

#### Semana 2: Distribuciones y Estadística Descriptiva
**02-Distribuciones-Descriptiva.md** (1,520 palabras)
- Distribución de Bernoulli y Binomial
- Distribución de Poisson
- Distribución Normal (Gaussiana)
- Teorema Central del Límite
- Media, mediana, desviación estándar
- Percentiles e IQR
- Box plots y detección de outliers

#### Semana 3: Pruebas de Hipótesis
**03-Pruebas-Hipotesis.md** (1,446 palabras)
- Estructura de hipótesis nula vs. alternativa
- Errores Tipo I y Tipo II
- P-valores y niveles de significancia
- Prueba t de una muestra
- Prueba t de dos muestras independientes
- Pruebas pareadas
- Interpretación de resultados

#### Semana 4: Power Analysis y Diseño Experimental
**04-PowerAnalisis-Diseño.md** (1,295 palabras)
- Variables independientes y dependientes
- Cuatro configuraciones experimentales (A/B/C/D)
- Concepto de poder estadístico
- Cálculo de tamaño muestral requerido
- Tamaño del efecto (Cohen's d)
- Órdenes y contrabalanceo
- Validez interna vs. externa

#### Semana 5: Reproducibilidad
**05-Reproducibilidad.md** (1,316 palabras)
- Importancia de la reproducibilidad científica
- Controlando estocacidad con semillas
- Temperature=0 para determinismo
- Documentación exhaustiva de experimentos
- Amenazas a validez (constructo, interna, externa)
- Pre-registro de análisis
- Checklists prácticas

#### Semana 6: Pruebas No Paramétricas
**06-Pruebas-NoParametricas.md** (1,312 palabras)
- Cuándo usar pruebas no paramétricas
- Prueba de normalidad Shapiro-Wilk
- Mann-Whitney U (alternativa a t-test)
- Prueba de Wilcoxon de rangos pareados
- Kruskal-Wallis (alternativa a ANOVA)
- Tamaño del efecto en pruebas no paramétricas
- Flujos de decisión para elegir prueba

#### Semana 7: Tamaño del Efecto
**07-Tamano-Efecto.md** (1,234 palabras)
- Por qué tamaño de efecto es crítico
- Cohen's d para diferencias de medias
- Intervalos de confianza 95%
- Bootstrap para estimación no paramétrica
- Reportando resultados completos
- Efecto tamaño para proporciones
- Número Necesario a Tratar (NNT)

#### Semana 8: Comparaciones Múltiples
**08-Comparaciones-Multiples.md** (1,241 palabras)
- El problema de comparaciones múltiples
- Tasa de error familia-wise (FWER)
- Correcciones de Bonferroni y Holm
- ANOVA para 3+ grupos
- Pruebas post-hoc (Tukey, Dunn)
- Tasa de Descubrimiento Falso (FDR)
- Análisis pre-registrado

#### Semana 9: MLOps y Visualización
**09-MLOps-Visualizacion.md** (1,282 palabras)
- Rastreo de experimentos con Weights & Biases
- Qué loguear: hiperparámetros y métricas
- MLflow como alternativa
- Eligiendo visualizaciones correctas
- Distribuciones, comparaciones, relaciones
- Series temporales
- Checklists de accesibilidad

#### Semana 10: Reporte Estadístico
**10-Reporte-Estadistico.md** (1,828 palabras)
- Estándares APA y IEEE
- Estructura de sección de Resultados
- Qué reportar vs. qué no
- Significancia estadística vs. práctica
- Sección de Limitaciones
- Tablas efectivas
- Ejemplos completos profesionales

### Características Comunes

Cada lectura incluye:
- **Tono conversacional y mentorizado**: Explicaciones intuitivas
- **Ejemplos contextualizados**: Todos relacionados a generación de kernels GPU
- **Ecuaciones claras**: Con interpretación práctica
- **Código Python**: Ejemplos reproducibles
- **Tablas y diagramas**: Para síntesis visual
- **Ejercicios prácticos**: Con soluciones esperadas
- **Preguntas de reflexión**: Para pensamiento crítico profundo

### Cómo Usar Este Módulo

1. **Lectura lineal**: Sigue semanas 1-10 en orden
2. **Referencia**: Usa como consulta cuando apliques cada concepto
3. **Ejercicios**: Completa todos los ejercicios con tus datos de proyecto
4. **Integración**: Conecta cada concepto con tu investigación sobre restricciones gramaticales

### Recursos Adicionales Recomendados

- **G*Power 3**: Calculadora de poder estadístico
- **Weights & Biases**: wandb.ai para rastreo de experimentos
- **SciPy/NumPy/Pandas**: Librerías Python para análisis
- **Open Science Framework**: Para pre-registro osf.io

### Total de Contenido

- **10 lecturas**: Una por semana del semestre
- **13,907 palabras**: Aproximadamente 1,400 palabras por lectura
- **Rango**: 1,234 a 1,828 palabras por archivo
- **Promedio**: ~5-7 páginas por lectura

### Notas Pedagógicas

Este módulo fue diseñado para:
- Estudiantes de maestría en CS/Ingeniería sin formación estadística previa
- Investigadores trabajando en proyectos de generación con LLMs
- Énfasis en aplicación práctica sobre teoría pura
- Reproducibilidad y rigor científico como temas transversales

Cada lectura asume que el estudiante ha completado las anteriores pero proporciona revisiones cuando conceptos previos son críticos.

---

*Módulo creado para ACA: Grammar-Constrained GPU Kernel Generation*
