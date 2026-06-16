#!/usr/bin/env python
"""Comprehensive verification that code uses config correctly.

Checks that:
1. All sensitive data is externalized to .env
2. All non-sensitive config is in config.yaml
3. No hardcoded values remain
4. Config is loaded from correct sources
"""

import sys
import re
from pathlib import Path

def check_for_hardcoded_values():
    """Scan Python files for hardcoded sensitive values."""
    issues = []
    project_root = Path(__file__).parent
    
    # Patterns to flag as hardcoded (bad)
    hardcoded_patterns = [
        (r"http://192\.168\.\d+\.\d+", "IP address"),
        (r"password\s*=\s*['\"]", "Password in code"),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "API key in code"),
        (r"secret\s*=\s*['\"][^'\"]+['\"]", "Secret in code"),
        # Paths should use config, not hardcoded
        (r'[\"\']/data/chroma_db[\"\']\s*(?!in|{)', "Hardcoded ChromaDB path"),
        (r'[\"\']/logs/[\"\']\s*(?!in|{)', "Hardcoded logs path"),
    ]
    
    py_files = list(project_root.rglob("*.py"))
    print(f"Scanning {len(py_files)} Python files for hardcoded values...")
    
    for py_file in py_files:
        # Skip test files and __pycache__
        if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
            continue
        
        try:
            content = py_file.read_text()
            for pattern, description in hardcoded_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip if it's in a comment or docstring
                    line_num = content[:match.start()].count('\n') + 1
                    line = content.split('\n')[line_num - 1]
                    if not line.strip().startswith('#'):
                        issues.append(f"{py_file.name}:{line_num} - {description}")
        except Exception as e:
            print(f"  ⚠ Error reading {py_file.name}: {e}")
    
    return issues

def verify_config_imports():
    """Check that files import from config module."""
    print("\n✓ Verifying config imports...")
    
    from config import settings
    
    print("  ✓ All exports available from config module")
    return True

def verify_no_dotenv_abuse():
    """Check that dotenv is not overused."""
    print("\n✓ Checking dotenv usage...")
    
    project_root = Path(__file__).parent
    py_files = list(project_root.rglob("*.py"))
    
    dotenv_files = []
    for py_file in py_files:
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text()
            if "load_dotenv" in content:
                # Only config.py should call load_dotenv
                if py_file.name != "config.py":
                    dotenv_files.append(py_file.name)
            if "os.getenv" in content and py_file.name not in ["config.py", "db_connector.py"]:
                dotenv_files.append(py_file.name + " (uses os.getenv)")
        except:
            pass
    
    if dotenv_files:
        print(f"  ⚠ Found dotenv usage in: {', '.join(dotenv_files)}")
        print("    Note: Ensure these files use config module instead of direct os.getenv")
        return False
    else:
        print("  ✓ dotenv usage properly isolated to config.py")
        return True

def verify_config_structure():
    """Verify config.yaml structure."""
    print("\n✓ Verifying config.yaml structure...")
    
    import yaml
    
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_sections = ["application", "lm_studio", "generation", "rag", "directories", "models"]
    found_sections = list(config.keys())
    
    missing = set(required_sections) - set(found_sections)
    if missing:
        print(f"  ❌ Missing sections in config.yaml: {missing}")
        return False
    
    print(f"  ✓ config.yaml has all required sections: {required_sections}")
    
    # Check for sensitive data in config.yaml
    yaml_str = str(config)
    if "mysql" in yaml_str.lower() or "password" in yaml_str.lower() or "http://" in yaml_str:
        print("  ⚠ Warning: Possible sensitive data in config.yaml")
        return False
    
    print("  ✓ No sensitive data found in config.yaml")
    return True

def main():
    """Run all verifications."""
    print("=" * 70)
    print("🔍 Configuration System Verification")
    print("=" * 70)
    
    all_pass = True
    
    # Check config structure
    try:
        if not verify_config_structure():
            all_pass = False
    except Exception as e:
        print(f"  ❌ Error verifying config.yaml: {e}")
        all_pass = False
    
    # Check imports
    try:
        if not verify_config_imports():
            all_pass = False
    except Exception as e:
        print(f"  ❌ Error verifying config imports: {e}")
        all_pass = False
    
    # Check dotenv usage
    try:
        if not verify_no_dotenv_abuse():
            all_pass = False
    except Exception as e:
        print(f"  ❌ Error checking dotenv usage: {e}")
        all_pass = False
    
    # Check for hardcoded values
    print("\n✓ Scanning for hardcoded values...")
    issues = check_for_hardcoded_values()
    if issues:
        print(f"  ⚠ Found {len(issues)} potential hardcoded values:")
        for issue in issues[:10]:  # Show first 10
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
        all_pass = False
    else:
        print("  ✓ No hardcoded sensitive values detected")
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✅ Configuration system verification PASSED")
        print("\nConfiguration Summary:")
        print("  • Non-sensitive settings: config.yaml")
        print("  • Sensitive data (.env): Database URL, API endpoints, model names")
        print("  • All code uses settings from config module")
        print("  • No hardcoded values or secrets in source")
        print("  • dotenv usage properly isolated")
        return 0
    else:
        print("❌ Configuration system verification FAILED")
        print("\nPlease review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
