# Sugerencias de Mejora Visual y de Estilo

A partir de la revisión navegada sobre el build en vivo de Jupyter Book, se han identificado las siguientes áreas de oportunidad y mejoras ("low-hanging fruit") para darle al curso ACA-2026 una estética más pulida, legible y "premium".

## 1. Bloques de Código y Sintaxis
- **Números de línea:** Implementar números de línea opcionales en bloques de código extensos. Al explicar código complejo (como en Compiladores o en `causal_attention`), poder decir "en la línea 14..." es metodológicamente invaluable.
- **Etiquetas de lenguaje:** Añadir un pequeño badge o label visual (ej: "PYTHON", "C++", "TRITON") en la esquina superior derecha del contenedor de los bloques de código para diferenciar el stack tecnológico rápidamente, algo visualmente moderno (ej. el estilo de la documentación de Tailwind o Vercel).

## 2. Admonitions (Note, Warning, Tip)
- **Consistencia Visual:** Los iconos personalizados (🎬 para videos, 🔧 para herramientas interactiva) resaltan muy bien. Se podría modificar el archivo de configuración CSS personalizado del Jupyter Book (`_static/custom.css`) para crear clases Admonition 100% nativas para ellos, dándoles sombras o un color de borde lateral muy distintivo.
- **Vibrancia en Modo Oscuro:** Al utilizar el tema oscuro, los contenedores verdes ("Tip") y azules ("Note") pierden saturación y lucen algo apagados. Escalar la intensidad o utilizar un sistema de colores pastel neón mejoraría la jerarquía visual del contenido.

## 3. Navegación y Estructura (ToC)
- **Organización de la Barra Lateral (Sidebar):** El índice primario funciona, pero dado que hay 5 grandes módulos diferentes (Introducción, AI, Compilers, Stats, Research), agregar un divisor sutil o un margen adicional (espaciado vertical) entre las "Partes" del libro en el sidebar reduciría la fatiga visual.
- **Scroll Spy en Índice Derecho:** Para lecturas largas (como Arquitectura Transformer), sugerir auditar el "scroll spy" (el indicador negrita en el Table of Contents derecho). En ocasiones de alta densidad gráfica, puede perder el foco.

## 4. Tipografía y Espaciado
- **Interlineado (Line-Height):** La tipografía base es limpia, pero subir el atributo CSS `line-height` a `1.6` o `1.65` permitiría que los párrafos técnicos más densos "respiren", mejorando contundentemente la velocidad de lectura.
- **Anclaje de Encabezados:** Aumentar el margen superior de los encabezados de nivel 2 y nivel 3 (`margin-top`). Un margen superior más grande agrupa perceptualmente al bloque que le precede, indicando sin ambigüedades el inicio de una nueva sección.

## 5. Accesibilidad Móvil (Responsividad)
- **Controles Prominentes:** Al encoger el viewport, los botones institucionales (como "Open in Colab") pueden dominar el primer scroll por completo. Podrían estar encapsulados en un contenedor más delgado.
- **Tablas Desbordadas:** Revisar la inclusión opcional de CSS `overflow-x: auto; white-space: nowrap;` sobre los wrappers de las tablas más complejas (especialmente las de Metodología de Research o los ASTs tabulares) para que en celular puedan deslizarse con el dedo (swipe) sin descolocar el margen padre del documento.

## 6. Toques "Premium" (Interactividad UI)
- **Efectos de Transición:** Modificar la carga de la página añadiendo una mínima transición fade-in CSS (unos ~200ms a 300ms) a la opacidad del `<main>`.
- **Medium Zoom para Imágenes:** Los diagramas estáticos generados con Python (histogramas estáticos, distribuciones matemáticas, grafos AST) serían más inmersivos si permitieran hacerles click para ampliar (efecto "lightbox"). Existen extensiones de Sphinx (`sphinx-design` o integraciones JS puras) para aportar una función de zoom muy fácilmente.
