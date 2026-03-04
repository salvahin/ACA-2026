#!/usr/bin/env python3
"""
Generate FSM and grammar diagrams for the LLM Code Generation project.
Uses graphviz with improved styling for clarity.

Dependencies:
    pip install graphviz railroad-diagrams cairosvg matplotlib Pillow
"""

import os, sys, tempfile, shutil, io
from pathlib import Path

import graphviz
import railroad as rr
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR / 'diagrams'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR    = Path(tempfile.mkdtemp(prefix='p1_diagrams_'))

# ── Theme (shared with compilers module) ───────────────────────────────
C = {
    'bg':        '#FFFFFF',
    'dark':      '#1B2A4A',
    'blue':      '#2563EB',
    'sky':       '#38BDF8',
    'orange':    '#F59E0B',
    'green':     '#10B981',
    'red':       '#EF4444',
    'purple':    '#8B5CF6',
    'gray':      '#94A3B8',
    'light':     '#F1F5F9',
}

FONT = 'Helvetica,Arial,sans-serif'

def render_gv(dot, name):
    """Render a graphviz object to PNG."""
    tmp = TEMP_DIR / name
    dot.render(str(tmp), cleanup=True, format='png')
    src = Path(f'{tmp}.png')
    dst = OUTPUT_DIR / f'{name}.png'
    shutil.copy2(str(src), str(dst))
    return dst

def svg_to_png(svg_content, output_path, scale=2.0):
    cairosvg.svg2png(bytestring=svg_content.encode('utf-8'),
                     write_to=str(output_path), scale=scale)


# ═══════════════════════════════════════════════════════════════════════
#  1. FSM para Gramática JSON
# ═══════════════════════════════════════════════════════════════════════

def generate_fsm_json_states():
    print("  [1/5] FSM — JSON grammar states")
    dot = graphviz.Digraph('fsm_json', format='png', engine='dot')
    dot.attr('graph', rankdir='LR', dpi='200', bgcolor='white',
             pad='0.6', nodesep='0.9', ranksep='1.3',
             fontname=FONT, fontsize='14', fontcolor=C['dark'],
             label='FSM para Gramática JSON\n'
                   'Cada estado determina los tokens válidos para la máscara de logits',
             labelloc='t', labeljust='c')
    dot.attr('node', fontname=FONT, fontsize='12', style='filled',
             penwidth='2.5', fixedsize='true', width='1.2', height='1.2')
    dot.attr('edge', fontname=FONT, fontsize='12', penwidth='2', color=C['gray'])

    # ── Structural states ──
    dot.node('START',     'START',              fillcolor=C['dark'],   fontcolor='white')
    dot.node('OBJ_OPEN',  'OBJ_OPEN\n{',       fillcolor=C['blue'],   fontcolor='white')
    dot.node('COLON',     'COLON\n:',           fillcolor=C['blue'],   fontcolor='white')
    dot.node('COMMA',     'COMMA\n,',           fillcolor=C['blue'],   fontcolor='white')
    dot.node('OBJ_CLOSE', 'OBJ_CLOSE\n}',      fillcolor=C['green'],  fontcolor='white', shape='doublecircle')

    # ── Key states ──
    with dot.subgraph(name='cluster_key') as s:
        s.attr(style='rounded,dashed', color=C['sky'], penwidth='2',
               label='Key States', fontname=FONT, fontsize='12', fontcolor=C['sky'])
        s.node('KEY',     'KEY\nExpecting "',    fillcolor=C['sky'],    fontcolor=C['dark'])
        s.node('KEY_STR', 'KEY_STR\nInside "…"', fillcolor=C['sky'],    fontcolor=C['dark'])

    # ── Value routing ──
    dot.node('VALUE', 'VALUE\nRouter', fillcolor=C['orange'], fontcolor='white',
             shape='diamond', width='1.4', height='1.4')

    # ── Value type states ──
    with dot.subgraph(name='cluster_val') as s:
        s.attr(style='rounded,dashed', color=C['purple'], penwidth='2',
               label='Value Types', fontname=FONT, fontsize='12', fontcolor=C['purple'])
        s.node('VAL_STR',  'VAL_STR\nString',   fillcolor=C['purple'], fontcolor='white')
        s.node('VAL_NUM',  'VAL_NUM\nNumber',   fillcolor=C['purple'], fontcolor='white')
        s.node('VAL_BOOL', 'VAL_BOOL\nBoolean', fillcolor=C['purple'], fontcolor='white')
        s.node('VAL_ARR',  'VAL_ARR\nArray',    fillcolor=C['purple'], fontcolor='white')
        s.node('VAL_OBJ',  'VAL_OBJ\nNested {}', fillcolor=C['purple'], fontcolor='white')

    # ── Start arrow ──
    dot.node('_s', '', shape='point', width='0')
    dot.edge('_s', 'START')

    # ── Main flow ──
    dot.edge('START',    'OBJ_OPEN',  label=' { ',  color=C['blue'],   fontcolor=C['blue'])
    dot.edge('OBJ_OPEN','KEY',        label=' " ',  color=C['sky'],    fontcolor=C['sky'])
    dot.edge('KEY',      'KEY_STR',   label=' chars ', color=C['sky'], fontcolor=C['sky'])
    dot.edge('KEY_STR',  'COLON',     label=' " ',  color=C['blue'],   fontcolor=C['blue'])
    dot.edge('COLON',   'VALUE',      label=' any ', color=C['orange'], fontcolor=C['orange'])

    # ── Value routing ──
    dot.edge('VALUE', 'VAL_STR',  label=' " ',   color=C['purple'], fontcolor=C['purple'])
    dot.edge('VALUE', 'VAL_NUM',  label=' [0-9] ', color=C['purple'], fontcolor=C['purple'])
    dot.edge('VALUE', 'VAL_BOOL', label=' t/f ',  color=C['purple'], fontcolor=C['purple'])
    dot.edge('VALUE', 'VAL_OBJ',  label=' { ',    color=C['purple'], fontcolor=C['purple'])
    dot.edge('VALUE', 'VAL_ARR',  label=' [ ',    color=C['purple'], fontcolor=C['purple'])

    # ── Completion → COMMA ──
    for src in ['VAL_STR', 'VAL_NUM', 'VAL_BOOL', 'VAL_ARR', 'VAL_OBJ']:
        dot.edge(src, 'COMMA', label=' end ', color=C['gray'], fontcolor=C['gray'], style='dashed')

    # ── Comma routes ──
    dot.edge('COMMA', 'KEY',       label=' " (next key) ', color=C['sky'],   fontcolor=C['sky'])
    dot.edge('COMMA', 'OBJ_CLOSE', label=' } ',            color=C['green'], fontcolor=C['green'])

    render_gv(dot, 'fsm_json_states')


# ═══════════════════════════════════════════════════════════════════════
#  2. Ciclo de Decodificación Restringida
# ═══════════════════════════════════════════════════════════════════════

def generate_constrained_decoding_flow():
    print("  [2/5] Constrained decoding cycle")
    dot = graphviz.Digraph('decoding', format='png', engine='dot')
    dot.attr('graph', rankdir='TB', dpi='200', bgcolor='white',
             pad='0.6', nodesep='0.7', ranksep='0.9',
             fontname=FONT, fontsize='14', fontcolor=C['dark'],
             label='Ciclo de Decodificación Restringida\n'
                   'Per-token latency: ~ms (GPU optimizado)',
             labelloc='t', labeljust='c')
    dot.attr('node', fontname=FONT, fontsize='12', style='filled,rounded',
             shape='box', penwidth='2', margin='0.3,0.2')
    dot.attr('edge', fontname=FONT, fontsize='11', penwidth='2.5',
             color=C['gray'], arrowsize='1.0')

    # Steps
    dot.node('s1', '1. LLM produce\nLogits',                       fillcolor=C['dark'],   fontcolor='white')
    dot.node('s2', '2. FSM determina\ntokens válidos',              fillcolor=C['orange'], fontcolor='white')
    dot.node('s3', '3. Genera bitmask\n(válido → 1, inválido → 0)', fillcolor=C['purple'], fontcolor='white')
    dot.node('s4', '4. Aplica máscara\na logits (inválido → −∞)',   fillcolor=C['red'],    fontcolor='white')
    dot.node('s5', '5. Softmax &\nSample token',                    fillcolor=C['blue'],   fontcolor='white')
    dot.node('s6', '6. Actualiza estado\nFSM con token',            fillcolor=C['green'],  fontcolor='white')

    # Decision
    dot.node('dec', '¿Token END\no max length?', shape='diamond',
             fillcolor=C['light'], fontcolor=C['dark'], penwidth='2.5',
             width='2.2', height='1.2')

    # Output
    dot.node('out', 'Salida: Texto\nGramaticalmente Válido', fillcolor=C['green'],
             fontcolor='white', shape='box', style='filled,rounded,bold', penwidth='3')

    # Edges
    dot.edge('s1', 's2', label=' logits ')
    dot.edge('s2', 's3', label=' valid tokens ')
    dot.edge('s3', 's4', label=' bitmask ')
    dot.edge('s4', 's5', label=' masked logits ')
    dot.edge('s5', 's6', label=' token ')
    dot.edge('s6', 'dec')

    dot.edge('dec', 's1', label=' No → siguiente token ',
             color=C['orange'], fontcolor=C['orange'], style='dashed')
    dot.edge('dec', 'out', label=' Sí ',
             color=C['green'], fontcolor=C['green'], penwidth='3')

    render_gv(dot, 'constrained_decoding_flow')


# ═══════════════════════════════════════════════════════════════════════
#  3. Pipeline de Compilación XGrammar
# ═══════════════════════════════════════════════════════════════════════

def generate_xgrammar_compilation_pipeline():
    print("  [3/5] XGrammar compilation pipeline")
    dot = graphviz.Digraph('xg_pipeline', format='png', engine='dot')
    dot.attr('graph', rankdir='LR', dpi='200', bgcolor='white',
             pad='0.6', nodesep='0.7', ranksep='1.0',
             fontname=FONT, fontsize='14', fontcolor=C['dark'],
             label='Pipeline de Compilación XGrammar\n'
                   'EBNF → Objetos Python → Árbol → Autómata → Máscaras de Tokens',
             labelloc='t', labeljust='c')
    dot.attr('node', fontname=FONT, fontsize='12', style='filled,rounded',
             shape='box', penwidth='2', margin='0.3,0.2')
    dot.attr('edge', fontname=FONT, fontsize='11', penwidth='2.5',
             color=C['gray'], arrowsize='1.2')

    stages = [
        ('s1', 'BNF / JSON\nSchema',           C['light'],  C['dark']),
        ('s2', 'Grammar\nParser',               C['blue'],   'white'),
        ('s3', 'AST\nConstruction',             C['purple'], 'white'),
        ('s4', 'FSM\nCompiler',                 C['orange'], 'white'),
        ('s5', 'DFA\nMinimization',             C['green'],  'white'),
        ('s6', 'Token Mask\nCache',             C['blue'],   'white'),
        ('s7', 'LogitsProcessor',               C['purple'], 'white'),
        ('s8', 'Generation\nLoop',              C['green'],  'white'),
    ]

    for nid, lbl, bg, fg in stages:
        extra = {}
        if nid == 's1':
            extra['shape'] = 'note'
        elif nid == 's8':
            extra['shape'] = 'box'
            extra['style'] = 'filled,rounded,bold'
            extra['penwidth'] = '3'
        dot.node(nid, lbl, fillcolor=bg, fontcolor=fg, **extra)

    for i in range(len(stages) - 1):
        dot.edge(stages[i][0], stages[i+1][0])

    # Feedback loop
    dot.edge('s8', 's6', label=' autoregressive\n iteration ',
             style='dashed', color=C['orange'], fontcolor=C['orange'])

    render_gv(dot, 'xgrammar_compilation_pipeline')


# ═══════════════════════════════════════════════════════════════════════
#  4. Railroad — JSON Grammar
# ═══════════════════════════════════════════════════════════════════════

def _fix_svg(svg_str):
    """Inline styles for cairosvg compatibility."""
    import re
    svg_str = svg_str.replace('<svg ', '<svg style="background:white" ', 1)
    svg_str = re.sub(r'<path d="',
        '<path style="stroke:#2563EB;stroke-width:2.5;fill:none" d="', svg_str)
    svg_str = re.sub(r'<rect (.*?)></rect>',
        lambda m: f'<rect style="fill:white;stroke:#2563EB;stroke-width:2" {m.group(1)}></rect>',
        svg_str)
    svg_str = re.sub(r'<text (.*?)>',
        lambda m: f'<text style="font-family:Helvetica,Arial,sans-serif;font-size:14px;fill:#1B2A4A" {m.group(1)}>',
        svg_str)
    svg_str = re.sub(
        r'(<g class="non-terminal[^"]*".*?)<rect style="fill:white;stroke:#2563EB;stroke-width:2"',
        r'\1<rect style="fill:#F5F3FF;stroke:#8B5CF6;stroke-width:2"',
        svg_str, flags=re.DOTALL)
    return svg_str

def generate_railroad_json():
    """Railroad diagram for JSON grammar."""
    print("  [4/5] Railroad — JSON grammar")
    from PIL import Image

    # JSON value
    value_diag = rr.Diagram(
        rr.Choice(0,
            rr.NonTerminal('object'),
            rr.NonTerminal('array'),
            rr.Terminal('string'),
            rr.Terminal('number'),
            rr.Terminal('true'),
            rr.Terminal('false'),
            rr.Terminal('null'),
        ),
        type='complex',
    )

    # JSON object
    obj_diag = rr.Diagram(
        rr.Sequence(
            rr.Terminal('{'),
            rr.Optional(
                rr.Sequence(
                    rr.Terminal('string'),
                    rr.Terminal(':'),
                    rr.NonTerminal('value'),
                    rr.ZeroOrMore(
                        rr.Sequence(
                            rr.Terminal(','),
                            rr.Terminal('string'),
                            rr.Terminal(':'),
                            rr.NonTerminal('value'),
                        ),
                    ),
                ),
            ),
            rr.Terminal('}'),
        ),
        type='complex',
    )

    # JSON array
    arr_diag = rr.Diagram(
        rr.Sequence(
            rr.Terminal('['),
            rr.Optional(
                rr.Sequence(
                    rr.NonTerminal('value'),
                    rr.ZeroOrMore(
                        rr.Sequence(
                            rr.Terminal(','),
                            rr.NonTerminal('value'),
                        ),
                    ),
                ),
            ),
            rr.Terminal(']'),
        ),
        type='complex',
    )

    imgs = []
    for name, diag in [('value', value_diag), ('object', obj_diag), ('array', arr_diag)]:
        sio = io.StringIO()
        diag.writeSvg(sio.write)
        svg_str = _fix_svg(sio.getvalue())
        tmp = TEMP_DIR / f'rr_json_{name}.png'
        svg_to_png(svg_str, tmp, scale=3.0)
        imgs.append((name, Image.open(str(tmp))))

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=150,
                              gridspec_kw={'height_ratios': [1.2, 1.5, 1]})
    fig.patch.set_facecolor('white')
    fig.suptitle('Railroad Diagrams — Gramática JSON (EBNF)',
                 fontsize=18, fontweight='bold', color=C['dark'], fontfamily='sans-serif')

    for ax, (name, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_ylabel(name, fontsize=14, fontweight='bold', color=C['purple'],
                      rotation=0, labelpad=60, va='center', fontfamily='sans-serif')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    plt.tight_layout(rect=[0.07, 0, 1, 0.94])
    out = OUTPUT_DIR / 'triton_grammar_levels.png'  # replaces the grammar levels diagram
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  5. Railroad — Triton Kernel Grammar (EBNF)
# ═══════════════════════════════════════════════════════════════════════

def generate_railroad_triton():
    """Railroad diagram for simplified Triton kernel grammar."""
    print("  [5/5] Railroad — Triton kernel grammar")
    from PIL import Image

    # Kernel definition
    kernel_diag = rr.Diagram(
        rr.Sequence(
            rr.Terminal('@triton.jit'),
            rr.Terminal('def'),
            rr.NonTerminal('name'),
            rr.Terminal('('),
            rr.NonTerminal('params'),
            rr.Terminal(')'),
            rr.Terminal(':'),
            rr.NonTerminal('body'),
        ),
        type='complex',
    )

    # Load/Store
    loadstore_diag = rr.Diagram(
        rr.Choice(0,
            rr.Sequence(
                rr.Terminal('tl.load'),
                rr.Terminal('('),
                rr.NonTerminal('ptr'),
                rr.Optional(
                    rr.Sequence(rr.Terminal(','), rr.Terminal('mask='), rr.NonTerminal('mask')),
                ),
                rr.Terminal(')'),
            ),
            rr.Sequence(
                rr.Terminal('tl.store'),
                rr.Terminal('('),
                rr.NonTerminal('ptr'),
                rr.Terminal(','),
                rr.NonTerminal('value'),
                rr.Optional(
                    rr.Sequence(rr.Terminal(','), rr.Terminal('mask='), rr.NonTerminal('mask')),
                ),
                rr.Terminal(')'),
            ),
        ),
        type='complex',
    )

    imgs = []
    for name, diag in [('kernel', kernel_diag), ('load/store', loadstore_diag)]:
        sio = io.StringIO()
        diag.writeSvg(sio.write)
        svg_str = _fix_svg(sio.getvalue())
        tmp = TEMP_DIR / f'rr_triton_{name.replace("/","_")}.png'
        svg_to_png(svg_str, tmp, scale=3.0)
        imgs.append((name, Image.open(str(tmp))))

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle('Railroad Diagrams — Gramática Triton Kernel (simplificada)',
                 fontsize=18, fontweight='bold', color=C['dark'], fontfamily='sans-serif')

    for ax, (name, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_ylabel(name, fontsize=13, fontweight='bold', color=C['purple'],
                      rotation=0, labelpad=70, va='center', fontfamily='sans-serif')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    plt.tight_layout(rect=[0.08, 0, 1, 0.93])
    # Save as a separate file — this doesn't replace an existing one
    out = OUTPUT_DIR / 'triton_railroad.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  Generating Project 1 Diagrams (improved)")
    print("=" * 60)

    generate_fsm_json_states()
    generate_constrained_decoding_flow()
    generate_xgrammar_compilation_pipeline()
    generate_railroad_json()
    generate_railroad_triton()

    print("\n  Generated files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        kb = f.stat().st_size / 1024
        print(f"    ✓ {f.name:<45} {kb:>7.1f} KB")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60 + "\n")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
