import json
import os
import glob

notebook_dir = "/Users/salvahin/TC3002B-2026/book/notebooks"
book_dir = "/Users/salvahin/TC3002B-2026/book"

def convert_ipynb_to_myst(ipynb_path, original_md_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    myst_content = []
    
    # Preserve the yaml header from the original .md file
    original_header_lines = []
    with open(original_md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines and lines[0].strip() == "---":
            original_header_lines.append(lines[0].rstrip())
            for line in lines[1:]:
                original_header_lines.append(line.rstrip())
                if line.strip() == "---":
                    break
        else:
            original_header_lines = [
                "---",
                "jupytext:",
                "  text_representation:",
                "    extension: .md",
                "    format_name: myst",
                "kernelspec:",
                "  display_name: Python 3",
                "  language: python",
                "  name: python3",
                "---"
            ]

    myst_content.extend(original_header_lines)
    myst_content.append("")
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = "".join(cell.get('source', []))
            myst_content.append(source)
            myst_content.append("")
        elif cell['cell_type'] == 'code':
            myst_content.append("```{code-cell} ipython3")
            
            # Handle tags
            tags = cell.get('metadata', {}).get('tags', [])
            if tags:
                myst_content.append(f":tags: [{', '.join(tags)}]")
                
            myst_content.append("")
            source = "".join(cell.get('source', []))
            
            if source:
                myst_content.append(source.rstrip('\n'))
            
            myst_content.append("```")
            myst_content.append("")
            
    return "\n".join(myst_content)

md_files = glob.glob(f"{book_dir}/**/*.md", recursive=True)
md_map = {os.path.basename(f).replace('.md', ''): f for f in md_files if 'notebooks' not in f and '.gemini' not in f and '_build' not in f}

success_count = 0
failed_syncs = []

for ipynb_path in glob.glob(f"{notebook_dir}/*.ipynb"):
    base_name = os.path.basename(ipynb_path).replace('.ipynb', '')
    if base_name in md_map:
        md_path = md_map[base_name]
        try:
            myst_content = convert_ipynb_to_myst(ipynb_path, md_path)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(myst_content)
            success_count += 1
            print(f"Synced {base_name}.ipynb -> {md_path}")
        except Exception as e:
            failed_syncs.append(f"{base_name}: {str(e)}")
            
print(f"\nSuccessfully synced {success_count} files.")
if failed_syncs:
    print("Failed to sync:")
    for f in failed_syncs:
        print(f)
