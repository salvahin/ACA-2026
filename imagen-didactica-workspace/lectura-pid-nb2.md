# Control PID: Fundamentos y Aplicaciones

**Asignatura:** Sistemas de Control
**Tema:** Controlador Proporcional-Integral-Derivativo (PID)
**Nivel:** Ingeniería universitaria

---

## Introducción

El controlador PID es uno de los algoritmos de control más utilizados en la industria.
Su popularidad se debe a su simplicidad conceptual y su capacidad para manejar una
amplia variedad de procesos industriales.

---

## Estructura del controlador PID

El controlador PID combina tres acciones de control: proporcional, integral y derivativa.
Cada una aporta características distintas a la respuesta del sistema.

![Diagrama de bloques del sistema de control PID mostrando la señal de referencia r(t), el error e(t), los tres bloques Kp, Ki y Kd en paralelo, la suma de las tres acciones, la planta G(s), la salida y(t) y el lazo de retroalimentación.](/sessions/vibrant-compassionate-dijkstra/mnt/TC3002B-2026/imagen-didactica-workspace/imagenes-pid-nb2/diagrama-de-bloques-del-sistema-de-control-pid-mostrando-la.png)

*Diagrama de bloques del sistema de control PID mostrando la señal de referencia r(t), el error e(t), los tres bloques Kp, Ki y Kd en paralelo, la suma de las tres acciones, la planta G(s), la salida y(t) y el lazo de retroalimentación.*

La acción proporcional genera una salida proporcional al error actual. La acción integral
acumula el error a lo largo del tiempo para eliminar el error en estado estacionario.
La acción derivativa anticipa el comportamiento futuro del error.

---

## Respuesta temporal

La sintonización de los parámetros Kp, Ki y Kd afecta directamente la respuesta
temporal del sistema en lazo cerrado.

![Gráfica de respuesta al escalón unitario comparando tres configuraciones: solo P (oscilatorio con offset), PI (sin offset pero con sobrepaso) y PID (respuesta óptima con mínimo sobrepaso). Ejes: tiempo (s) en x, amplitud en y. Incluir la referencia como línea punteada.](/sessions/vibrant-compassionate-dijkstra/mnt/TC3002B-2026/imagen-didactica-workspace/imagenes-pid-nb2/grafica-de-respuesta-al-escalon-unitario-comparando-tres-con.png)

*Gráfica de respuesta al escalón unitario comparando tres configuraciones: solo P (oscilatorio con offset), PI (sin offset pero con sobrepaso) y PID (respuesta óptima con mínimo sobrepaso). Ejes: tiempo (s) en x, amplitud en y. Incluir la referencia como línea punteada.*

---

## Aplicación industrial

Los controladores PID se encuentran en prácticamente todas las plantas industriales,
desde control de temperatura hasta regulación de flujo.

![Fotografía de un panel de control industrial moderno con pantallas HMI mostrando lazos de control PID en tiempo real, en una planta de procesamiento químico.](/sessions/vibrant-compassionate-dijkstra/mnt/TC3002B-2026/imagen-didactica-workspace/imagenes-pid-nb2/fotografia-de-un-panel-de-control-industrial-moderno-con-pan.png)

*Fotografía de un panel de control industrial moderno con pantallas HMI mostrando lazos de control PID en tiempo real, en una planta de procesamiento químico.*

---

## Síntesis

El controlador PID sigue siendo la piedra angular del control automático industrial
gracias a su balance entre simplicidad y efectividad.
