#!/usr/bin/env python3
"""
Add deprecation warnings to all __init__.py files in archive_old_src subdirectories.

This script prepends a warning message to each __init__.py file to prevent accidental use.
"""

import os
from pathlib import Path

WARNING_HEADER = '''"""
⚠️ ARCHIVED - DO NOT USE ⚠️
================================

**Status**: DEPRECATED
**Reason**: Old implementation superseded by current src/ directory
**Use Instead**: See archive_old_src/README_ARCHIVE.md for replacements

---

ORIGINAL DOCSTRING (for historical reference):

'''

def add_warning_to_init(init_path: Path):
    """Add deprecation warning to __init__.py if not already present."""
    
    if not init_path.exists():
        return
    
    content = init_path.read_text()
    
    # Skip if warning already present
    if "⚠️ ARCHIVED - DO NOT USE ⚠️" in content:
        print(f"✓ Already warned: {init_path}")
        return
    
    # If starts with docstring, insert warning
    if content.startswith('"""'):
        # Find end of first docstring
        end_quote = content.find('"""', 3)
        if end_quote != -1:
            # Extract original docstring
            original_doc = content[3:end_quote]
            # Build new content
            new_content = WARNING_HEADER + original_doc + '"""\n' + content[end_quote+4:]
            init_path.write_text(new_content)
            print(f"✅ Updated: {init_path}")
        else:
            print(f"⚠️  Skipped (malformed docstring): {init_path}")
    else:
        # No docstring, prepend warning
        new_content = WARNING_HEADER + 'No original docstring.\n"""\n\n' + content
        init_path.write_text(new_content)
        print(f"✅ Updated: {init_path}")

def main():
    """Process all __init__.py files in archive_old_src subdirectories."""
    
    archive_root = Path("archive_old_src")
    
    if not archive_root.exists():
        print("❌ archive_old_src not found")
        return
    
    print("🔍 Scanning for __init__.py files in archive_old_src...\n")
    
    init_files = list(archive_root.rglob("__init__.py"))
    
    if not init_files:
        print("No __init__.py files found")
        return
    
    print(f"Found {len(init_files)} __init__.py files\n")
    
    for init_path in init_files:
        add_warning_to_init(init_path)
    
    print(f"\n✅ Processed {len(init_files)} files")

if __name__ == "__main__":
    main()
