# VS Code Editor Extension Fix

## Issue
- Notebook shows blank cells (but file is valid JSON with 20 cells)
- Python files may also appear corrupted
- Root cause: VS Code Jupyter/Notebook extension rendering issue

## Quick Fixes (Try in Order)

### Fix 1: Reload VS Code Window
1. Press `Cmd+Shift+P` (Command Palette)
2. Type: "Developer: Reload Window"
3. Press Enter
4. Reopen the notebook

### Fix 2: Disable/Re-enable Jupyter Extension
1. Press `Cmd+Shift+X` (Extensions)
2. Search: "Jupyter"
3. Click "Disable" on Jupyter extension
4. Click "Enable" again
5. Reload window (`Cmd+Shift+P` → "Developer: Reload Window")

### Fix 3: Clear VS Code Cache
```bash
# Close VS Code completely first
rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage/*
rm -rf ~/Library/Application\ Support/Code/Cache/*
rm -rf ~/Library/Application\ Support/Code/CachedData/*
# Reopen VS Code
```

### Fix 4: Reinstall Jupyter Extension
1. `Cmd+Shift+X` → Search "Jupyter"
2. Uninstall the Jupyter extension
3. Restart VS Code
4. Reinstall Jupyter extension from marketplace
5. Restart VS Code again

### Fix 5: Open in JupyterLab (Verify File is Valid)
```bash
# Install JupyterLab if needed
pip install jupyterlab

# Launch JupyterLab
cd /Users/vivekdurairaj/Projects/Cogumi-LLM/notebooks
jupyter lab Benchmark_Diagnostic_v2.ipynb
```
This will prove the file itself is fine - it's a VS Code rendering issue.

## File Validation (Already Passed ✅)
```bash
# Verify notebook is valid JSON with proper structure
python3 -c "
import json
nb = json.load(open('Benchmark_Diagnostic_v2.ipynb'))
print(f'✅ Valid notebook: {len(nb[\"cells\"])} cells')
print(f'✅ Has kernelspec: {\"kernelspec\" in nb[\"metadata\"]}')
"
```

Output:
```
✅ Valid notebook: 20 cells
✅ Has kernelspec: True
```

## Alternative: Use Colab or Vast.ai
If VS Code continues having issues:
1. Upload `Benchmark_Diagnostic_v2.ipynb` to Google Colab
2. Or use on Vast.ai where it will work with JupyterLab

## Prevention
After fixing, update VS Code and extensions:
1. Check for VS Code updates (Code → Check for Updates)
2. Update all extensions (Extensions → "..." → Check for Extension Updates)
