# Diagrama de Bode - Sistema de Segundo Orden

## Descripción
Este proyecto genera un diagrama de Bode educativo para un sistema de segundo orden con los siguientes parámetros:

- **Frecuencia natural (ωn):** 10 rad/s
- **Factor de amortiguamiento (ζ):** 0.5
- **Función de transferencia:** H(s) = 100 / (s² + 10s + 100)

## Características del diagrama

### Diagrama de Magnitud
- Escala semilogarítmica (eje X logarítmico, eje Y lineal)
- Rango de frecuencia: 1 a 1000 rad/s
- Línea de referencia en -3 dB
- Línea vertical indicando la frecuencia natural (ωn = 10 rad/s)
- Etiquetas en español

### Diagrama de Fase
- Escala semilogarítmica (eje X logarítmico, eje Y lineal)
- Rango de frecuencia: 1 a 1000 rad/s
- Línea de referencia en -90°
- Línea vertical indicando la frecuencia natural (ωn = 10 rad/s)
- Etiquetas en español

## Características del Sistema

| Característica | Valor |
|---|---|
| DC Gain | 0.04 dB |
| Magnitud en ωn | 0.0 dB |
| Fase en ωn | -90.0° |
| Factor de calidad (Q) | 1.0 |
| Pico de resonancia | No presente |
| Clasificación | Subamortiguado (ζ < 1) |

## Archivos generados

1. **bode_diagram.png** - Imagen del diagrama (372 KB, 300 DPI, 3562x2980 píxeles)
2. **diagram_info.json** - Metadatos del diagrama generado
3. **system_analysis.json** - Análisis de características del sistema
4. **metrics.json** - Métricas de ejecución
5. **README.md** - Este archivo

## Tecnologías utilizadas

- Python 3
- NumPy - Cálculos numéricos
- Matplotlib - Visualización
- SciPy - Análisis de sistemas y señales

## Notas técnicas

- El sistema con ζ = 0.5 es un sistema subamortiguado (underdamped) pero sin pico de resonancia evidente
- La fase cambia de 0° a -180° conforme aumenta la frecuencia
- El diagrama cruza -3 dB en las frecuencias de corte
- La resolución de 300 DPI es adecuada para impresión profesional
