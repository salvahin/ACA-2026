# ACA - Aplicaciones Computacionales Avanzadas

Repositorio del curso TC3002B. El contenido se publica automáticamente en GitHub Pages.

**Sitio web:** https://salvahin.github.io/ACA-2026

---

## Guía para Profesores: Editar Contenido

### Estructura del Repositorio

```
book/
├── ai/                    # Módulo: Inteligencia Artificial
├── compilers/             # Módulo: Compiladores
├── project_1/             # Módulo: Proyecto 1
├── project_2/             # Módulo: Proyecto 2
├── research/              # Módulo: Investigación
├── stats/                 # Módulo: Estadística
├── notebooks/             # Notebooks para Colab
├── _config.yml            # Configuración del libro
└── _toc.yml               # Tabla de contenidos
```

### Cómo Editar una Lectura

1. **Localiza el archivo** en `book/<módulo>/`
   - Los archivos de lectura son `.md` (Markdown)
   - Ejemplo: `book/ai/01_ia_clasica_vs_generativa.md`

2. **Edita el archivo** con cualquier editor de texto
   - VSCode (recomendado)
   - GitHub web (para cambios rápidos)

3. **Formato del contenido:**
   - Usa Markdown estándar
   - Matemáticas con LaTeX: `$E = mc^2$` o bloques `$$...$$`
   - Imágenes: `![descripción](diagrams/nombre.png)`
   - Código: usa bloques con triple backtick y el lenguaje

### Cómo Publicar los Cambios

#### Opción A: Desde Terminal (Recomendado)

```bash
# 1. Ver qué archivos cambiaron
git status

# 2. Agregar los cambios
git add book/

# 3. Crear commit con mensaje descriptivo
git commit -m "Actualizar lectura X del módulo Y"

# 4. Subir a GitHub
git push
```

#### Opción B: Desde VSCode

1. Abre la pestaña **Source Control** (Ctrl+Shift+G)
2. Revisa los archivos modificados
3. Escribe un mensaje de commit
4. Click en **Commit** y luego **Push**

#### Opción C: Desde GitHub Web

1. Navega al archivo en github.com/salvahin/ACA-2026
2. Click en el ícono de lápiz (Edit)
3. Realiza los cambios
4. Click en **Commit changes**

### Publicación Automática

Después de hacer `push`:

1. **GitHub Actions** detecta el cambio automáticamente
2. Compila el libro con Jupyter Book (~2-3 minutos)
3. Publica en GitHub Pages

**Ver estado:** [Actions](https://github.com/salvahin/ACA-2026/actions)

### Agregar Imágenes/Diagramas

1. Coloca la imagen en `book/<módulo>/diagrams/`
2. Referencia en el Markdown:
   ```markdown
   ![Descripción del diagrama](diagrams/mi_imagen.png)
   ```
3. Commit y push

### Agregar una Nueva Lectura

1. Crea el archivo `.md` en el módulo correspondiente
2. Edita `book/_toc.yml` para agregarlo a la tabla de contenidos
3. Commit y push

### Solución de Problemas

| Problema | Solución |
|----------|----------|
| El sitio no se actualiza | Revisa [Actions](https://github.com/salvahin/ACA-2026/actions) para ver si hay errores |
| Error de sintaxis Markdown | Verifica que los bloques de código estén cerrados |
| Imagen no aparece | Verifica la ruta relativa y que el archivo exista |
| Conflicto de git | Ejecuta `git pull` antes de hacer cambios |

---

## Desarrollo Local (Opcional)

Para previsualizar cambios localmente:

```bash
# Instalar dependencias
pip install jupyter-book

# Compilar el libro
jupyter-book build book/

# Abrir en navegador
open book/_build/html/index.html
```

---

## Contacto

- **Autor:** Salvador Hinojosa
- **Curso:** TC3002B - 2026
