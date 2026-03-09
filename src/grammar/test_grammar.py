import os
import sys

try:
    import xgrammar as xg
except ImportError:
    print("Error: xgrammar no está instalado. Ejecuta: pip install xgrammar")
    sys.exit(1)

def main():
    grammar_path = os.path.join(os.path.dirname(__file__), "triton.ebnf")
    
    with open(grammar_path, "r", encoding="utf-8") as f:
        ebnf_string = f.read()

    print("Contenido de la gramática:")
    print("-" * 40)
    print(ebnf_string)
    print("-" * 40)
    
    print("\nCompilando la gramática con xgrammar...")
    try:
        # Intentando cargar la gramática desde EBNF
        grammar = xg.Grammar.from_ebnf(ebnf_string)
        print("✅ Gramática compilada exitosamente!")
        
        # Opcional: imprimir algo sobre la gramática si es posible
        # print(grammar)
        
    except Exception as e:
        print(f"❌ Error al compilar la gramática: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
