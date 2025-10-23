#!/usr/bin/env python3
"""
Phase 2 Compression - Pre-flight Validation
Run this to ensure Phase 2 is ready before Phase 1 completes
"""

import sys
import importlib
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        version = ""
        try:
            mod = sys.modules[module_name]
            if hasattr(mod, '__version__'):
                version = f" v{mod.__version__}"
        except:
            pass
        print(f"✅ {package_name or module_name}{version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - {str(e)}")
        return False

def check_file(file_path):
    """Check if a file exists"""
    path = Path(file_path)
    if path.exists():
        size = path.stat().st_size
        print(f"✅ {file_path} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {file_path} - NOT FOUND")
        return False

def check_syntax(file_path):
    """Check Python file for syntax errors"""
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️  {file_path} - File not found")
        return False
    
    try:
        with open(path, 'r') as f:
            compile(f.read(), path, 'exec')
        print(f"✅ {file_path} - Syntax OK")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path} - Syntax Error at line {e.lineno}: {e.msg}")
        return False

def main():
    print("=" * 60)
    print("🔍 PHASE 2 COMPRESSION - PRE-FLIGHT VALIDATION")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    print("\n📋 Python Environment:")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✅ Python {py_version}")
    
    # Check core dependencies
    print("\n📦 Core Dependencies:")
    all_good &= check_import("torch")
    all_good &= check_import("transformers")
    all_good &= check_import("peft")
    all_good &= check_import("accelerate")
    
    # Check compression libraries
    print("\n🗜️  Compression Libraries:")
    sparseml_ok = check_import("sparseml", "SparseML")
    autoawq_ok = check_import("awq", "AutoAWQ")
    
    if not sparseml_ok:
        print("   💡 Install: pip install sparseml[transformers]")
        all_good = False
    
    if not autoawq_ok:
        print("   💡 Install: pip install autoawq")
        all_good = False
    
    # Check Phase 2 scripts
    print("\n📄 Phase 2 Scripts:")
    all_good &= check_syntax("src/phase2_compression/neural_magic_prune.py")
    all_good &= check_syntax("src/phase2_compression/awq_quantize.py")
    
    # Check Phase 2 notebook
    print("\n📓 Phase 2 Notebook:")
    all_good &= check_file("notebooks/Phase2_Compression_Colab.ipynb")
    
    # Check configuration files
    print("\n⚙️  Configuration Files:")
    all_good &= check_file("configs/compression.yaml")
    
    # Check directories
    print("\n📂 Required Directories:")
    dirs = [
        "data/checkpoints",
        "models/compressed",
        "models/pruned",
        "models/quantized"
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ - Will be created")
            path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_path}/")
    
    # Final summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ PHASE 2 READY - All checks passed!")
        print("=" * 60)
        print("\n🚀 Next steps:")
        print("1. Wait for Phase 1 training to complete")
        print("2. Open notebooks/Phase2_Compression_Colab.ipynb")
        print("3. Run the 3-step compression pipeline")
        return 0
    else:
        print("⚠️  PHASE 2 NOT READY - Some issues found")
        print("=" * 60)
        print("\n🔧 Fix the issues above before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())
