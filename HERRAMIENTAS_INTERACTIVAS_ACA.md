# Plataformas y Visualizadores Interactivos: ACA-2026

Este documento recopila las mejores plataformas web, simuladores y visualizadores interactivos para complementar el aprendizaje teórico y práctico del curso **Aplicaciones Computacionales Avanzadas (ACA-2026)**. A diferencia de los videos, estas herramientas te permiten *jugar* con los hiperparámetros, arquitecturas y datos en tiempo real directamente en tu navegador.

---

## 1. Módulo de Inteligencia Artificial

Para entender cómo las matemáticas se transforman en "razonamiento" a través de redes neuronales y Transformers.

*   **[TensorFlow Neural Network Playground](https://playground.tensorflow.org/)**
    *   *Uso:* Fundamentos de Deep Learning.
    *   *Descripción:* Una herramienta clásica e imprescindible. Te permite construir una red neuronal básica en el navegador, añadiendo capas ocultas y neuronas. Útil para visualizar interactivamente cómo la red aprende a separar datos complejos ajustando los pesos y sesgos en tiempo real.
*   **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)**
    *   *Uso:* Arquitectura Transformer y LLMs.
    *   *Descripción:* Una visualización interactiva brutal que te permite explorar la arquitectura de un LLM moderno (como GPT-2) en el navegador. Puedes ingresar tu propio texto y ver exactamente cómo el mecanismo de *Self-Attention* procesa la información token por token.
*   **[BertViz](https://github.com/jessevig/bertviz)**
    *   *Uso:* Mecanismos de Atención.
    *   *Descripción:* Aunque requiere correrse en un notebook (Jupyter/Colab), es la herramienta por excelencia para visualizar cómo los *Attention Heads* de modelos como BERT o GPT conectan las palabras e infieren contexto.

---

## 2. Módulo de Teoría de Compiladores

Simuladores visuales para entender cómo las máquinas procesan lenguajes formales y gramáticas.

*   **[AutomataVerse](https://automataverse.com/)**
    *   *Uso:* Lenguajes Regulares y Automátas (DFAs, NFAs).
    *   *Descripción:* Una plataforma interactiva open-source donde puedes dibujar grafos de máquinas de estados y simular paso a paso cómo una cadena de texto es procesada, aceptada o rechazada. Muestra visualmente la conversión de NFA a DFA.
*   **[FSM Simulator (Ivan Zuzak)](https://ivanzuzak.info/noam/webapps/fsm_simulator/)**
    *   *Uso:* Máquinas de Estados Finitos.
    *   *Descripción:* Permite definir expresiones regulares o estados finitos y generar automáticamente la visualización del grafo correspondiente, simulando entradas.
*   **[AST Explorer](https://astexplorer.net/)**
    *   *Uso:* Parsing y Árboles de Sintaxis Abstracta (ASTs).
    *   *Descripción:* Una herramienta web fantástica donde pegas cualquier código fuente y te genera en tiempo real el Árbol de Sintaxis Abstracta. Vital para entender cómo los parsers estructuran el código antes de compilarlo.

---

## 3. Módulo de Estadística de Investigación

Para perderle el miedo a la probabilidad y el diseño experimental a través del juego visual.

*   **[Seeing Theory (Brown University)](https://seeing-theory.brown.edu/)**
    *   *Uso:* Probabilidad, Distribuciones, Inferencia y Regresión.
    *   *Descripción:* Posiblemente el mejor libro de texto interactivo de estadística en internet. Creado con D3.js, hace que conceptos como el Teorema del Límite Central, las distribuciones continuas o la Inferencia Bayesiana sean completamente intuitivos mediante minijuegos analíticos interactivos.
*   **[Probability Playground](https://www.distributome.org/V3/)**
    *   *Uso:* Distribuciones y Descriptiva.
    *   *Descripción:* Explora cómo se interrelacionan docenas de distribuciones de probabilidad diferentes alterando sus parámetros y viendo cómo cambia la campana estadística en vivo.

---

## 4. Módulos de Proyectos (GPU, CUDA, Triton y Code Gen)

Interactuar con el silicio y el paralelismo no es sencillo en un navegador web, pero existen aproximaciones visuales invaluables para crear un modelo mental.

*   **[Cuda Grid Visualizer (Hugging Face)](https://huggingface.co/spaces/ThomasSimonini/CUDA-Grid-Visualizer)**
    *   *Uso:* Modelo de ejecución CUDA (Hilos, Bloques, Grids).
    *   *Descripción:* Una aplicación web interactiva de Hugging Face donde puedes modificar los tamaños del Grid y de los Bloques (X, Y, Z) para entender tridimensionalmente cómo se organizan y paralelizan los hilos en una GPU antes de programar un kernel Triton/CUDA.
*   **[NVIDIA Nsight Systems / Compute (Desktop Tools)](https://developer.nvidia.com/nsight-systems)**
    *   *Uso:* Debugging, Profiling y Análisis de Rendimiento.
    *   *Descripción:* Aunque son programas de escritorio instalables (no web), son **la** herramienta visual por excelencia para el curso. Proveen líneas de tiempo gráficas masivas y detalladas (Timelines) para ver exactamente cuándo tu kernel de GPU se ejecuta, dónde hay cuellos de botella de memoria y cómo se comunican la CPU y la GPU.
*   **[WebGPU Examples / Use.GPU](https://usegpu.live/)**
    *   *Uso:* Simulaciones in-browser de paralelismo GPU.
    *   *Descripción:* Para palpar el futuro del GPU computing en la web. Permite ver ejemplos visuales en vivo (Live Graphs) de simulaciones fluidodinámicas calculadas directamente en los shaders de la GPU de tu computadora a través del navegador.
