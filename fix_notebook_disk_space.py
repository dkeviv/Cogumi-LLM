#!/usr/bin/env python3
"""
Fix H100 notebook with disk space prevention - JSON-safe editing
"""

import json
import sys

def add_disk_space_fixes():
    """Add disk space fixes to H100 notebook without breaking JSON."""
    
    notebook_path = '/Users/vivekdurairaj/Projects/Cogumi-LLM/notebooks/H100_Training_Clean.ipynb'
    
    # Read notebook
    print("📖 Reading notebook...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"   Found {len(nb['cells'])} cells")
    
    # Find the cell that creates train.py (contains 'script = """')
    train_script_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'script = """' in source and 'train.py' in source:
                train_script_cell_idx = i
                break
    
    if train_script_cell_idx is None:
        print("❌ Could not find training script generation cell")
        return False
    
    print(f"   Found training script cell at index {train_script_cell_idx}")
    
    # Get the cell source
    cell = nb['cells'][train_script_cell_idx]
    source_lines = cell['source']
    
    # Find and update save_total_limit and save_steps
    modified = False
    for i, line in enumerate(source_lines):
        if 'save_total_limit=3' in line:
            source_lines[i] = line.replace('save_total_limit=3', 'save_total_limit=2')
            modified = True
            print(f"   ✅ Fixed: save_total_limit=3 → save_total_limit=2")
        
        if 'save_steps=1000' in line and 'logging_steps' not in line:
            source_lines[i] = line.replace('save_steps=1000', 'save_steps=2000')
            modified = True
            print(f"   ✅ Fixed: save_steps=1000 → save_steps=2000")
    
    if not modified:
        print("   ℹ️  Parameters already updated or not found")
    
    # Save the modified notebook
    print("\n💾 Saving notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    
    print("✅ Notebook saved successfully")
    
    # Validate JSON
    print("\n🔍 Validating JSON...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        json.load(f)
    print("✅ JSON is valid")
    
    return True

if __name__ == '__main__':
    try:
        success = add_disk_space_fixes()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
