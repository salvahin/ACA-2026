#!/usr/bin/env python3
"""
Revisor de Convención de Idioma — ACA-2026
==========================================
Convención: comentarios/docstrings en español, código (variables/funciones) en inglés.

Uso:
    python3 scripts/revisar_idioma.py              # Revisa todo el repo
    python3 scripts/revisar_idioma.py book/ai/     # Revisa solo un módulo
"""

import ast
import json
import re
import sys
from pathlib import Path

# Palabras en español que no deberían aparecer en nombres de variables/funciones
# (palabras comunes del español que indican que el nombre está en español)
PALABRAS_ESPANOL = {
    'calcular', 'crear', 'obtener', 'verificar', 'procesar', 'generar',
    'guardar', 'cargar', 'mostrar', 'imprimir', 'lista', 'arreglo',
    'numero', 'valor', 'tasa', 'tamaño', 'promedio', 'suma', 'resultado',
    'entrada', 'salida', 'archivo', 'directorio', 'nombre', 'nombre',
    'tiempo', 'paso', 'contador', 'indice', 'longitud', 'total',
    'datos', 'modelo', 'entrenamiento', 'prediccion', 'capas', 'neuronas',
}

# Patrón para detectar comentarios en inglés que deberían estar en español
PATRON_COMENTARIO_INGLES = re.compile(
    r'#\s*(the|this|get|set|check|create|load|save|process|calculate|return|initialize)\s',
    re.IGNORECASE
)


def revisar_archivo_python(ruta: Path) -> list[dict]:
    """Analiza un archivo .py y detecta violaciones de convención."""
    violaciones = []

    try:
        codigo = ruta.read_text(encoding='utf-8')
        tree = ast.parse(codigo)
    except (SyntaxError, UnicodeDecodeError) as e:
        return [{'tipo': 'parse_error', 'archivo': str(ruta), 'mensaje': str(e)}]

    lineas = codigo.split('\n')

    for nodo in ast.walk(tree):
        # Verificar nombres de funciones y variables
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nombre = nodo.name
            nombre_lower = nombre.lower()
            for palabra in PALABRAS_ESPANOL:
                if palabra in nombre_lower:
                    violaciones.append({
                        'tipo': 'nombre_en_espanol',
                        'archivo': str(ruta),
                        'linea': nodo.lineno,
                        'elemento': nombre,
                        'sugerencia': f'Renombrar a inglés: ej. calculate_*, get_*, process_*'
                    })
                    break

        # Verificar docstrings en inglés
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (nodo.body and isinstance(nodo.body[0], ast.Expr) and
                    isinstance(nodo.body[0].value, ast.Constant) and
                    isinstance(nodo.body[0].value.value, str)):
                docstring = nodo.body[0].value.value
                # Detectar palabras muy comunes en inglés que indican docstring en inglés
                palabras_ingles = re.findall(
                    r'\b(returns?|calculates?|generates?|initializes?|creates?|loads?)\b',
                    docstring, re.IGNORECASE
                )
                if palabras_ingles:
                    violaciones.append({
                        'tipo': 'docstring_en_ingles',
                        'archivo': str(ruta),
                        'linea': nodo.lineno,
                        'elemento': nodo.name,
                        'muestra': docstring[:80].replace('\n', ' ') + '...',
                        'sugerencia': 'Traducir docstring al español'
                    })

    # Verificar comentarios en inglés
    for i, linea in enumerate(lineas, 1):
        comentario = re.search(r'#(.+)', linea)
        if comentario and PATRON_COMENTARIO_INGLES.search(comentario.group(1)):
            violaciones.append({
                'tipo': 'comentario_en_ingles',
                'archivo': str(ruta),
                'linea': i,
                'muestra': linea.strip(),
                'sugerencia': 'Traducir comentario al español'
            })

    return violaciones


def revisar_notebook(ruta: Path) -> list[dict]:
    """Analiza un archivo .ipynb extrayendo las celdas de código."""
    violaciones = []
    try:
        nb = json.loads(ruta.read_text(encoding='utf-8'))
        for idx, celda in enumerate(nb.get('cells', [])):
            if celda.get('cell_type') == 'code':
                codigo_celda = ''.join(celda.get('source', []))
                # Crear archivo temporal virtual para analizarlo
                tmp = Path(f'/tmp/__celda_{idx}.py')
                tmp.write_text(codigo_celda, encoding='utf-8')
                viol = revisar_archivo_python(tmp)
                for v in viol:
                    v['archivo'] = str(ruta)
                    v['celda'] = idx + 1
                    v.pop('linea', None)  # Las líneas cambian en notebooks
                violaciones.extend(viol)
                tmp.unlink(missing_ok=True)
    except (json.JSONDecodeError, KeyError):
        violaciones.append({'tipo': 'parse_error', 'archivo': str(ruta)})
    return violaciones


def generar_reporte(directorio: str = '.') -> None:
    """Genera reporte completo de violaciones de convención de idioma."""
    base = Path(directorio)
    archivos_py  = list(base.rglob('*.py'))
    archivos_nb  = list(base.rglob('*.ipynb'))

    # Excluir directorios de herramientas
    excluir = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.ipynb_checkpoints'}
    archivos_py = [f for f in archivos_py if not any(p in excluir for p in f.parts)]
    archivos_nb = [f for f in archivos_nb if not any(p in excluir for p in f.parts)]

    print(f"🔍 Revisando convención de idioma en: {directorio}")
    print(f"   Archivos Python: {len(archivos_py)}")
    print(f"   Notebooks:       {len(archivos_nb)}")
    print(f"   Convención:      Comentarios/docstrings en ESPAÑOL, código en INGLÉS")
    print("=" * 70)

    total_violaciones = []

    for ruta in archivos_py:
        viols = revisar_archivo_python(ruta)
        total_violaciones.extend(viols)

    for ruta in archivos_nb:
        viols = revisar_notebook(ruta)
        total_violaciones.extend(viols)

    # Agrupar por tipo
    por_tipo = {}
    for v in total_violaciones:
        tipo = v['tipo']
        por_tipo.setdefault(tipo, []).append(v)

    if not total_violaciones:
        print("\n✅ Sin violaciones encontradas. ¡Convención de idioma correcta!")
        return

    print(f"\n⚠️  {len(total_violaciones)} violaciones encontradas:\n")

    resumen = {
        'nombre_en_espanol':   ('🏷️  Nombres de función/variable en español', 'Usar nombres en inglés'),
        'docstring_en_ingles': ('📝 Docstrings en inglés',                     'Traducir docstrings al español'),
        'comentario_en_ingles':('💬 Comentarios en inglés',                    'Traducir comentarios al español'),
    }

    for tipo, (titulo, accion) in resumen.items():
        viols = por_tipo.get(tipo, [])
        if not viols:
            continue
        print(f"{titulo}: {len(viols)} casos")
        print(f"   Acción: {accion}")
        for v in viols[:3]:  # Mostrar máximo 3 ejemplos
            archivo = Path(v['archivo']).name
            linea = v.get('linea', v.get('celda', '?'))
            elemento = v.get('elemento', v.get('muestra', ''))[:50]
            print(f"   • {archivo}:{linea}  →  {elemento}")
        if len(viols) > 3:
            print(f"   ... y {len(viols) - 3} más.")
        print()

    print(f"Total: {len(total_violaciones)} violaciones en {len(set(v['archivo'] for v in total_violaciones))} archivos")


if __name__ == '__main__':
    directorio = sys.argv[1] if len(sys.argv) > 1 else 'book'
    generar_reporte(directorio)
