#!/usr/bin/env python3
"""
Generate grammar and automata diagrams for the Compilers module.
Uses graphviz (FSMs, parse trees, pipelines), railroad-diagrams (BNF/EBNF),
and matplotlib (Chomsky hierarchy).

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR / 'diagrams'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR    = Path(tempfile.mkdtemp(prefix='diagrams_'))

# ── Theme ──────────────────────────────────────────────────────────────
C = {
    'bg':        '#FFFFFF',
    'dark':      '#1B2A4A',   # titles, text
    'blue':      '#2563EB',   # primary / accept states
    'sky':       '#38BDF8',   # secondary highlights
    'orange':    '#F59E0B',   # transitions / operators
    'green':     '#10B981',   # terminals / accept
    'red':       '#EF4444',   # reject / errors
    'purple':    '#8B5CF6',   # non-terminals
    'gray':      '#94A3B8',   # neutral
    'light':     '#F1F5F9',   # backgrounds
}

FONT = 'Helvetica,Arial,sans-serif'

# ── Helpers ────────────────────────────────────────────────────────────

def svg_to_png(svg_content: str, output_path: Path, scale: float = 2.0):
    """Convert SVG string to high-res PNG."""
    cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        write_to=str(output_path),
        scale=scale,
    )

def render_gv(dot: graphviz.Digraph, name: str) -> Path:
    """Render a graphviz Digraph to PNG in OUTPUT_DIR."""
    tmp = TEMP_DIR / name
    dot.render(str(tmp), cleanup=True, format='png')
    src = Path(f'{tmp}.png')
    dst = OUTPUT_DIR / f'{name}.png'
    shutil.copy2(str(src), str(dst))
    return dst


# ═══════════════════════════════════════════════════════════════════════
#  1. DFA: Números binarios divisibles por 3
# ═══════════════════════════════════════════════════════════════════════

def generate_dfa_binary():
    print("  [1/7] DFA — binarios divisibles por 3")
    dot = graphviz.Digraph('dfa_binary', format='png', engine='dot')
    dot.attr('graph', rankdir='LR', dpi='200', bgcolor='white',
             pad='0.5', nodesep='1.2', ranksep='1.5',
             label=r'DFA: Números binarios divisibles por 3\n'
                   r'Σ = {0, 1}  ·  Estado inicial: q₀  ·  Aceptación: {q₀}',
             labelloc='t', labeljust='c',
             fontname=FONT, fontsize='16', fontcolor=C['dark'])
    dot.attr('node', fontname=FONT, fontsize='14', style='filled',
             penwidth='2.5', fixedsize='true', width='1.1', height='1.1')
    dot.attr('edge', fontname=FONT, fontsize='14', penwidth='2',
             color=C['gray'], fontcolor=C['dark'])

    # States
    dot.node('q0', 'q₀\nresto 0', fillcolor=C['blue'],   fontcolor='white', shape='doublecircle')
    dot.node('q1', 'q₁\nresto 1', fillcolor=C['orange'],  fontcolor='white', shape='circle')
    dot.node('q2', 'q₂\nresto 2', fillcolor=C['red'],     fontcolor='white', shape='circle')

    # Start arrow
    dot.node('start', '', shape='point', width='0', height='0')
    dot.edge('start', 'q0', arrowsize='1.2', penwidth='2')

    # Transitions with colored labels
    dot.edge('q0', 'q0', label=' 0 ', fontcolor=C['blue'],   color=C['blue'])
    dot.edge('q0', 'q1', label=' 1 ', fontcolor=C['orange'], color=C['orange'])
    dot.edge('q1', 'q2', label=' 0 ', fontcolor=C['red'],    color=C['red'])
    dot.edge('q1', 'q0', label=' 1 ', fontcolor=C['green'],  color=C['green'])
    dot.edge('q2', 'q1', label=' 0 ', fontcolor=C['orange'], color=C['orange'])
    dot.edge('q2', 'q2', label=' 1 ', fontcolor=C['red'],    color=C['red'])

    render_gv(dot, 'dfa_binary')


# ═══════════════════════════════════════════════════════════════════════
#  2. Conversión NFA → DFA (Construcción de subconjuntos)
# ═══════════════════════════════════════════════════════════════════════

def _build_nfa():
    dot = graphviz.Digraph('nfa', format='png', engine='dot')
    dot.attr('graph', rankdir='LR', dpi='200', bgcolor='white',
             pad='0.4', nodesep='1', ranksep='1.2',
             label='NFA Original', labelloc='t', labeljust='c',
             fontname=FONT, fontsize='15', fontcolor=C['dark'])
    dot.attr('node', fontname=FONT, fontsize='14', style='filled',
             penwidth='2.5', fixedsize='true', width='0.9', height='0.9')
    dot.attr('edge', fontname=FONT, fontsize='14', penwidth='2', color=C['gray'])

    dot.node('p0', 'p₀', fillcolor=C['blue'],   fontcolor='white')
    dot.node('p1', 'p₁', fillcolor=C['orange'],  fontcolor='white')
    dot.node('p2', 'p₂', fillcolor=C['green'],   fontcolor='white', shape='doublecircle')
    dot.node('s', '', shape='point', width='0')
    dot.edge('s', 'p0')

    dot.edge('p0', 'p0', label=' a ', color=C['blue'],   fontcolor=C['blue'])
    dot.edge('p0', 'p1', label=' a ', color=C['orange'],  fontcolor=C['orange'])
    dot.edge('p1', 'p2', label=' b ', color=C['green'],   fontcolor=C['green'])
    return dot

def _build_dfa_converted():
    dot = graphviz.Digraph('dfa_conv', format='png', engine='dot')
    dot.attr('graph', rankdir='LR', dpi='200', bgcolor='white',
             pad='0.4', nodesep='1', ranksep='1.2',
             label='DFA Resultante (subconjuntos)', labelloc='t', labeljust='c',
             fontname=FONT, fontsize='15', fontcolor=C['dark'])
    dot.attr('node', fontname=FONT, fontsize='13', style='filled',
             penwidth='2.5', fixedsize='true', width='1.1', height='1.1')
    dot.attr('edge', fontname=FONT, fontsize='14', penwidth='2', color=C['gray'])

    dot.node('A', '{p₀}',       fillcolor=C['blue'],   fontcolor='white')
    dot.node('B', '{p₀,p₁}',    fillcolor=C['orange'],  fontcolor='white')
    dot.node('C', '{p₀,p₁,p₂}', fillcolor=C['green'],   fontcolor='white', shape='doublecircle')
    dot.node('D', '∅',           fillcolor=C['gray'],    fontcolor='white')
    dot.node('s', '', shape='point', width='0')
    dot.edge('s', 'A')

    dot.edge('A', 'B', label=' a ', color=C['orange'], fontcolor=C['orange'])
    dot.edge('A', 'D', label=' b ', color=C['gray'],   fontcolor=C['gray'])
    dot.edge('B', 'B', label=' a ', color=C['orange'], fontcolor=C['orange'])
    dot.edge('B', 'C', label=' b ', color=C['green'],  fontcolor=C['green'])
    dot.edge('C', 'B', label=' a ', color=C['orange'], fontcolor=C['orange'])
    dot.edge('C', 'C', label=' b ', color=C['green'],  fontcolor=C['green'])
    dot.edge('D', 'D', label=' a,b ', color=C['gray'], fontcolor=C['gray'])
    return dot

def generate_nfa_to_dfa():
    print("  [2/7] NFA → DFA conversion")
    from PIL import Image

    nfa_path = render_gv(_build_nfa(), 'nfa_original')
    dfa_path = render_gv(_build_dfa_converted(), 'dfa_converted')

    nfa_img = Image.open(str(nfa_path))
    dfa_img = Image.open(str(dfa_path))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle('Conversión NFA → DFA: Construcción de Subconjuntos',
                 fontsize=18, fontweight='bold', color=C['dark'], fontfamily='sans-serif')

    for ax, img, title, col in [
        (axes[0], nfa_img, 'NFA Original',      C['blue']),
        (axes[1], dfa_img, 'DFA Resultante',     C['green']),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', color=col, fontfamily='sans-serif')
        ax.axis('off')

    # Mapping table
    fig.text(0.5, 0.02,
             'Mapeo:  {p₀} → A   ·   {p₀,p₁} → B   ·   {p₀,p₁,p₂} → C   ·   ∅ → D (trampa)',
             ha='center', fontsize=12, fontstyle='italic', fontfamily='sans-serif',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=C['light'], edgecolor=C['gray'], alpha=0.9))

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    out = OUTPUT_DIR / 'nfa_to_dfa_conversion.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  3. Parse Trees — 2 + 3 * 4  (correcto vs ambiguo)
# ═══════════════════════════════════════════════════════════════════════

def _parse_tree(name, label, edges, node_styles):
    """Build a clean parse tree with graphviz."""
    dot = graphviz.Digraph(name, format='png', engine='dot')
    dot.attr('graph', rankdir='TB', dpi='200', bgcolor='white',
             pad='0.3', nodesep='0.6', ranksep='0.7',
             label=label, labelloc='t', labeljust='c',
             fontname=FONT, fontsize='14', fontcolor=C['dark'])
    dot.attr('node', fontname=FONT, fontsize='13', style='filled', penwidth='2')
    dot.attr('edge', penwidth='2', color=C['gray'], arrowhead='none')

    for nid, nlabel, ntype in node_styles:
        if ntype == 'nonterm':
            dot.node(nid, nlabel, shape='ellipse', fillcolor=C['purple'], fontcolor='white')
        elif ntype == 'op':
            dot.node(nid, nlabel, shape='diamond', fillcolor=C['orange'], fontcolor='white',
                     width='0.6', height='0.6', fixedsize='true')
        else:  # terminal / number
            dot.node(nid, nlabel, shape='box', fillcolor=C['green'], fontcolor='white',
                     style='filled,rounded')

    for src, dst in edges:
        dot.edge(src, dst)
    return dot

def generate_parse_trees():
    print("  [3/7] Parse trees — 2 + 3 * 4")
    from PIL import Image

    # ── Correct tree: E → E + T, T → T * F ──
    correct_nodes = [
        ('E1', 'E', 'nonterm'),
        ('E2', 'E', 'nonterm'), ('plus', '+', 'op'), ('T1', 'T', 'nonterm'),
        ('T2', 'T', 'nonterm'),                        # E2 → T2
        ('F1', 'F', 'nonterm'),                        # T2 → F1
        ('n2', '2', 'term'),                           # F1 → 2
        ('T3', 'T', 'nonterm'), ('mult', '×', 'op'), ('F2', 'F', 'nonterm'),  # T1 → T3 * F2
        ('F3', 'F', 'nonterm'),                        # T3 → F3
        ('n3', '3', 'term'),                           # F3 → 3
        ('n4', '4', 'term'),                           # F2 → 4
    ]
    correct_edges = [
        ('E1', 'E2'), ('E1', 'plus'), ('E1', 'T1'),
        ('E2', 'T2'),
        ('T2', 'F1'),
        ('F1', 'n2'),
        ('T1', 'T3'), ('T1', 'mult'), ('T1', 'F2'),
        ('T3', 'F3'),
        ('F3', 'n3'),
        ('F2', 'n4'),
    ]
    correct = _parse_tree('parse_correct',
                          'Correcto: (2 + (3 × 4)) = 14\nPrecedencia respetada',
                          correct_edges, correct_nodes)

    # ── Ambiguous tree: left-to-right, no precedence ──
    ambig_nodes = [
        ('E1', 'E', 'nonterm'),
        ('E2', 'E', 'nonterm'), ('mult', '×', 'op'), ('T1', 'T', 'nonterm'),
        ('E3', 'E', 'nonterm'), ('plus', '+', 'op'), ('T2', 'T', 'nonterm'),
        ('T3', 'T', 'nonterm'),   # E3 → T3
        ('F1', 'F', 'nonterm'),   # T3 → F1
        ('n2', '2', 'term'),
        ('F2', 'F', 'nonterm'),   # T2 → F2
        ('n3', '3', 'term'),
        ('F3', 'F', 'nonterm'),   # T1 → F3
        ('n4', '4', 'term'),
    ]
    ambig_edges = [
        ('E1', 'E2'), ('E1', 'mult'), ('E1', 'T1'),
        ('E2', 'E3'), ('E2', 'plus'), ('E2', 'T2'),
        ('E3', 'T3'),
        ('T3', 'F1'), ('F1', 'n2'),
        ('T2', 'F2'), ('F2', 'n3'),
        ('T1', 'F3'), ('F3', 'n4'),
    ]
    ambig = _parse_tree('parse_ambiguous',
                        'Ambiguo: ((2 + 3) × 4) = 20\nSin precedencia',
                        ambig_edges, ambig_nodes)

    cp = render_gv(correct, 'parse_correct')
    ap = render_gv(ambig,   'parse_ambiguous')

    ci = Image.open(str(cp))
    ai = Image.open(str(ap))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor('white')
    fig.suptitle('Árboles de Análisis Sintáctico: 2 + 3 × 4',
                 fontsize=18, fontweight='bold', color=C['dark'], fontfamily='sans-serif')

    axes[0].imshow(ci); axes[0].axis('off')
    axes[0].set_title('✓ Correcto', fontsize=14, fontweight='bold', color=C['green'])
    axes[1].imshow(ai); axes[1].axis('off')
    axes[1].set_title('✗ Ambiguo', fontsize=14, fontweight='bold', color=C['red'])

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C['purple'], label='No-terminal'),
        mpatches.Patch(facecolor=C['orange'], label='Operador'),
        mpatches.Patch(facecolor=C['green'],  label='Terminal'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=3, fontsize=12,
               frameon=True, fancybox=True, shadow=False, edgecolor=C['gray'])

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    out = OUTPUT_DIR / 'parse_tree_precedence.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  4. Jerarquía de Chomsky
# ═══════════════════════════════════════════════════════════════════════

def generate_chomsky_hierarchy():
    print("  [4/7] Jerarquía de Chomsky")

    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    fig.patch.set_facecolor('white')

    levels = [
        ('Tipo 0 — Recursivamente Enumerable', 'Máquina de Turing',   'α → β',      C['red'],    0.95),
        ('Tipo 1 — Sensible al Contexto',      'Autómata Acotado',    'αAβ → αγβ',  C['orange'], 0.72),
        ('Tipo 2 — Libre de Contexto',          'Autómata de Pila',    'A → γ',      C['purple'], 0.49),
        ('Tipo 3 — Regular',                     'DFA / NFA',           'A → aB | a', C['blue'],   0.26),
    ]

    for name, automaton, grammar, color, r in levels:
        box = FancyBboxPatch(
            (-r, -r), 2*r, 2*r,
            boxstyle="round,pad=0.04",
            edgecolor=color, facecolor=color, alpha=0.12,
            linewidth=3, zorder=5 - levels.index((name, automaton, grammar, color, r))
        )
        ax.add_patch(box)
        # Border (solid)
        border = FancyBboxPatch(
            (-r, -r), 2*r, 2*r,
            boxstyle="round,pad=0.04",
            edgecolor=color, facecolor='none',
            linewidth=2.5, zorder=10
        )
        ax.add_patch(border)

        # Type label (top-left inside)
        ax.text(-r + 0.05, r - 0.04, name,
                fontsize=12, fontweight='bold', va='top', ha='left',
                color=color, fontfamily='sans-serif')

        # Grammar (left)
        ax.text(-r - 0.06, -r * 0.3, grammar,
                fontsize=11, fontweight='bold', ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1.5),
                color=color, fontfamily='monospace')

        # Automaton (right)
        ax.text(r + 0.06, -r * 0.3, automaton,
                fontsize=11, fontweight='bold', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1.5),
                color=color, fontfamily='sans-serif')

    # Labels
    ax.text(-0.01, 0.98, 'Gramática', fontsize=10, ha='right', va='top',
            transform=ax.transAxes, color=C['gray'], fontstyle='italic')
    ax.text(1.01, 0.98, 'Reconocedor', fontsize=10, ha='left', va='top',
            transform=ax.transAxes, color=C['gray'], fontstyle='italic')

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.suptitle('Jerarquía de Chomsky', fontsize=20, fontweight='bold',
                 color=C['dark'], fontfamily='sans-serif', y=0.97)
    fig.text(0.5, 0.02,
             'Tipo 3 ⊂ Tipo 2 ⊂ Tipo 1 ⊂ Tipo 0  —  cada nivel es estrictamente más expresivo',
             ha='center', fontsize=11, fontstyle='italic', color=C['gray'],
             fontfamily='sans-serif')

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = OUTPUT_DIR / 'chomsky_detail.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  5. XGrammar Pipeline
# ═══════════════════════════════════════════════════════════════════════

def generate_xgrammar_pipeline():
    print("  [5/7] XGrammar pipeline")
    dot = graphviz.Digraph('xgrammar', format='png', engine='dot')
    dot.attr('graph', rankdir='LR', dpi='200', bgcolor='white',
             pad='0.6', nodesep='0.8', ranksep='1.0',
             fontname=FONT, fontsize='14', fontcolor=C['dark'],
             label='Pipeline de XGrammar: Gramática EBNF → Máscara de Tokens',
             labelloc='t', labeljust='c')
    dot.attr('node', fontname=FONT, fontsize='12', style='filled,rounded',
             shape='box', penwidth='2', margin='0.25,0.15')
    dot.attr('edge', fontname=FONT, fontsize='11', penwidth='2.5',
             color=C['gray'], arrowsize='1.2')

    stages = [
        ('ebnf',     'Gramática\nEBNF',       C['blue']),
        ('parser',   'Parser\nPython',         C['purple']),
        ('ast',      'AST\n(Árbol Sintáctico)', C['orange']),
        ('compiler', 'Compilador\nFSM',         C['green']),
        ('dfa',      'Estados\nDFA',            C['blue']),
        ('mask',     'Generador\nMáscaras',     C['purple']),
        ('logits',   'LogitsProcessor\n(Integración)', C['green']),
    ]

    for nid, lbl, col in stages:
        dot.node(nid, lbl, fillcolor=col, fontcolor='white')

    for i in range(len(stages) - 1):
        dot.edge(stages[i][0], stages[i+1][0])

    render_gv(dot, 'xgrammar_detail')


# ═══════════════════════════════════════════════════════════════════════
#  6. Railroad Diagram — BNF de expresiones aritméticas
# ═══════════════════════════════════════════════════════════════════════

def _fix_railroad_svg(svg_str: str) -> str:
    """Post-process railroad SVG to add inline styles for cairosvg compatibility."""
    import re
    # Add white background
    svg_str = svg_str.replace('<svg ', '<svg style="background:white" ', 1)
    # Style paths: blue stroke, no fill
    svg_str = re.sub(
        r'<path d="',
        '<path style="stroke:#2563EB;stroke-width:2.5;fill:none" d="',
        svg_str)
    # Style rects for terminals: white fill, blue border, rounded
    svg_str = re.sub(
        r'<rect (.*?)></rect>',
        lambda m: f'<rect style="fill:white;stroke:#2563EB;stroke-width:2" {m.group(1)}></rect>',
        svg_str)
    # Style text
    svg_str = re.sub(
        r'<text (.*?)>',
        lambda m: f'<text style="font-family:Helvetica,Arial,sans-serif;font-size:14px;fill:#1B2A4A" {m.group(1)}>',
        svg_str)
    # Fix non-terminal rects (they use class="non-terminal")
    svg_str = svg_str.replace(
        'class="non-terminal "',
        'class="non-terminal " ')
    # Re-style non-terminal rects with purple
    svg_str = re.sub(
        r'(<g class="non-terminal[^"]*".*?)<rect style="fill:white;stroke:#2563EB;stroke-width:2"',
        r'\1<rect style="fill:#F5F3FF;stroke:#8B5CF6;stroke-width:2"',
        svg_str, flags=re.DOTALL)
    return svg_str

def generate_railroad_arithmetic():
    """Railroad / syntax diagram for arithmetic expression grammar."""
    print("  [6/7] Railroad diagram — expresiones aritméticas")

    # Expr → Term (('+' | '-') Term)*
    expr_diag = rr.Diagram(
        rr.Sequence(
            rr.NonTerminal('Term'),
            rr.ZeroOrMore(
                rr.Sequence(
                    rr.Choice(0, rr.Terminal('+'), rr.Terminal('-')),
                    rr.NonTerminal('Term'),
                ),
            ),
        ),
        type='complex',
    )

    # Term → Factor (('*' | '/') Factor)*
    term_diag = rr.Diagram(
        rr.Sequence(
            rr.NonTerminal('Factor'),
            rr.ZeroOrMore(
                rr.Sequence(
                    rr.Choice(0, rr.Terminal('*'), rr.Terminal('/')),
                    rr.NonTerminal('Factor'),
                ),
            ),
        ),
        type='complex',
    )

    # Factor → NUMBER | '(' Expr ')'
    factor_diag = rr.Diagram(
        rr.Choice(
            0,
            rr.Terminal('NUMBER'),
            rr.Sequence(
                rr.Terminal('('),
                rr.NonTerminal('Expr'),
                rr.Terminal(')'),
            ),
        ),
        type='complex',
    )

    from PIL import Image
    imgs = []
    for name, diag in [('Expr', expr_diag), ('Term', term_diag), ('Factor', factor_diag)]:
        sio = io.StringIO()
        diag.writeSvg(sio.write)
        svg_str = _fix_railroad_svg(sio.getvalue())
        tmp_path = TEMP_DIR / f'rr_{name}.png'
        svg_to_png(svg_str, tmp_path, scale=3.0)
        imgs.append((name, Image.open(str(tmp_path))))

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), dpi=150,
                              gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.patch.set_facecolor('white')
    fig.suptitle('Railroad Diagrams — Gramática de Expresiones Aritméticas',
                 fontsize=18, fontweight='bold', color=C['dark'], fontfamily='sans-serif')

    for ax, (name, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_ylabel(name, fontsize=14, fontweight='bold', color=C['purple'],
                      rotation=0, labelpad=60, va='center', fontfamily='sans-serif')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(rect=[0.08, 0, 1, 0.94])
    out = OUTPUT_DIR / 'll_vs_lr_parsing.png'  # replaces old parsing diagram
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  7. Compiler Pipeline (multi-pass)
# ═══════════════════════════════════════════════════════════════════════

def generate_compiler_pipeline():
    print("  [7/7] Compiler pipeline (multi-pass)")
    dot = graphviz.Digraph('compiler_pipeline', format='png', engine='dot')
    dot.attr('graph', rankdir='TB', dpi='200', bgcolor='white',
             pad='0.5', nodesep='0.6', ranksep='0.8',
             fontname=FONT, fontsize='14', fontcolor=C['dark'],
             label='Pipeline de Compilación Multi-Paso',
             labelloc='t', labeljust='c')
    dot.attr('node', fontname=FONT, fontsize='12', style='filled,rounded',
             shape='box', penwidth='2', margin='0.3,0.15')
    dot.attr('edge', fontname=FONT, fontsize='10', penwidth='2',
             color=C['gray'], arrowsize='1.0')

    # Frontend
    with dot.subgraph(name='cluster_frontend') as s:
        s.attr(label='Frontend', style='dashed', color=C['blue'],
               fontname=FONT, fontsize='13', fontcolor=C['blue'], penwidth='2')
        s.node('lex',   'Análisis Léxico\n(Tokenizer)',     fillcolor=C['blue'],   fontcolor='white')
        s.node('parse', 'Análisis Sintáctico\n(Parser)',    fillcolor=C['blue'],   fontcolor='white')
        s.node('sem',   'Análisis Semántico\n(Type Check)', fillcolor=C['sky'],    fontcolor=C['dark'])

    # Middle
    with dot.subgraph(name='cluster_middle') as s:
        s.attr(label='Middle-End', style='dashed', color=C['purple'],
               fontname=FONT, fontsize='13', fontcolor=C['purple'], penwidth='2')
        s.node('ir',  'Representación\nIntermedia (IR)', fillcolor=C['purple'], fontcolor='white')
        s.node('opt', 'Optimización',                    fillcolor=C['purple'], fontcolor='white')

    # Backend
    with dot.subgraph(name='cluster_backend') as s:
        s.attr(label='Backend', style='dashed', color=C['green'],
               fontname=FONT, fontsize='13', fontcolor=C['green'], penwidth='2')
        s.node('codegen', 'Generación\nde Código', fillcolor=C['green'], fontcolor='white')
        s.node('target',  'Código\nObjeto',        fillcolor=C['green'], fontcolor='white')

    # Input/output
    dot.node('src', 'Código\nFuente', fillcolor=C['light'], fontcolor=C['dark'],
             shape='note', penwidth='1.5')
    dot.node('exe', 'Ejecutable', fillcolor=C['light'], fontcolor=C['dark'],
             shape='note', penwidth='1.5')

    for a, b in [('src','lex'),('lex','parse'),('parse','sem'),
                 ('sem','ir'),('ir','opt'),('opt','codegen'),('codegen','target'),('target','exe')]:
        dot.edge(a, b)

    render_gv(dot, 'compiler_multipass')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  Generating Compiler Module Diagrams (improved)")
    print("=" * 60)

    generate_dfa_binary()
    generate_nfa_to_dfa()
    generate_parse_trees()
    generate_chomsky_hierarchy()
    generate_xgrammar_pipeline()
    generate_railroad_arithmetic()
    generate_compiler_pipeline()

    print("\n  Generated files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        kb = f.stat().st_size / 1024
        print(f"    ✓ {f.name:<40} {kb:>7.1f} KB")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60 + "\n")

    # Cleanup
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
