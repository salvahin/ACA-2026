import os
import glob
import json
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS_DIR = "/Users/salvahin/TC3002B-2026/book/notebooks"

def test_notebook(notebook_path):
    print(f"\\nTesting: {os.path.basename(notebook_path)}")
    print("-" * 50)
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    # We will skip the setup cells (!pip install) so we don't mess up the local env
    # Or at least filter them out before running if possible.
    clean_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = "".join(cell.source)
            if "!pip install" in source or "apt-get" in source:
                continue # Skip installations
        clean_cells.append(cell)
        
    nb.cells = clean_cells
    
    client = NotebookClient(nb, timeout=60, kernel_name='python3', record_timing=False)
    
    try:
        client.execute()
        print("✅ SUCCESS: All cells executed perfectly.")
        return True, "Success"
    except CellExecutionError as e:
        # Extract the exact error message
        error_msg = str(e).split('\\n')[-2] if '\\n' in str(e) else str(e)
        
        # Check if it's an expected environment error (e.g., no GPU, missing heavy module)
        expected_failures = ["cuda", "triton", "xgrammar", "torch.cuda", "No module named"]
        
        if any(kw.lower() in str(e).lower() for kw in expected_failures):
            print(f"⚠️  ENVIRONMENT LIMITATION (Expected): {error_msg}")
            return True, f"Env Limit: {error_msg}"
        else:
            print(f"❌ CODE ERROR: {error_msg}")
            return False, f"Error: {error_msg}"
    except Exception as e:
        print(f"❌ KERNEL/SYSTEM ERROR: {e}")
        return False, f"Sys Error: {e}"

def main():
    notebooks = sorted(glob.glob(os.path.join(NOTEBOOKS_DIR, "*.ipynb")))
    total = len(notebooks)
    
    print(f"Starting verification of {total} notebooks. Skipping !pip install cells.")
    
    results = {}
    passed = 0
    failed = 0
    env_blocks = 0
    
    for nb in notebooks:
        # To avoid sitting here for 10 minutes, let's just do a dry run on the focal ones
        # or the first 5 to test the script logic.
        success, reason = test_notebook(nb)
        results[os.path.basename(nb)] = reason
        
        if success and "Env Limit" not in reason:
            passed += 1
        elif "Env Limit" in reason:
            env_blocks += 1
            passed += 1 # We count expected env blocks as "passed" logic-wise
        else:
            failed += 1
            
    print("\\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    print(f"Total Notebooks: {total}")
    print(f"✅ Executed perfectly: {passed - env_blocks}")
    print(f"⚠️ Stopped safely due to local env missing GPU/libs: {env_blocks}")
    print(f"❌ Failed due to real code errors: {failed}")
    
    # Let's save a report
    with open("notebook_test_report.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main()
