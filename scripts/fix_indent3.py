import os

path = "/Users/salvahin/TC3002B-2026/book/notebooks/02_fundamentos_deep_learning.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

BAD = '    "    ax.text(0.9, 0.65, \'Salida\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\\\n",\\n'
GOOD = '    "ax.text(0.9, 0.65, \'Salida\\\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\\\n",\\n'

if BAD in c:
    c = c.replace(BAD, GOOD)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(c)
    print("Fixed indentation via raw string replace.")
else:
    print("BAD string not found directly.")

# Also handle the other variation where the newline \n is not escaped
BAD2 = '    "    ax.text(0.9, 0.65, \'Salida\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\n",\n'
GOOD2 = '    "ax.text(0.9, 0.65, \'Salida\\n(Sigmoid)\', ha=\'center\', fontsize=9)\\n",\n'
if BAD2 in c:
    c = c.replace(BAD2, GOOD2)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(c)
    print("Fixed indentation via raw string replace (v2).")
    
# Third variation, what if it's missing the comma or ending? Just replace the core code
BAD3 = "    ax.text(0.9, 0.65, 'Salida\\n(Sigmoid)', ha='center', fontsize=9)\\n"
GOOD3 = "ax.text(0.9, 0.65, 'Salida\\n(Sigmoid)', ha='center', fontsize=9)\\n"
if BAD3 in c:
     c = c.replace(BAD3, GOOD3)
     with open(path, 'w', encoding='utf-8') as f:
        f.write(c)
     print("Fixed indentation via raw string replace (v3).")
