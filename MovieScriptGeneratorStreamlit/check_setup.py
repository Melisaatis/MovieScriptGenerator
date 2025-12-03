#!/usr/bin/env python3
"""
🎬 MOVIE SCRIPT GENERATOR - SETUP & CONFIGURATION CHECKER
Run this to verify everything is set up correctly
"""

import sys
from pathlib import Path
import subprocess

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version OK:", f"{version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print("❌ Python version too old. Need 3.8+")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama installed")
            
            # Check if running
            import requests
            try:
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    print("✅ Ollama is running")
                    
                    # Check for mistral model
                    tags = resp.json()
                    models = [m['name'] for m in tags.get('models', [])]
                    if any('mistral' in m for m in models):
                        print("✅ Mistral model found")
                    else:
                        print("⚠️  Mistral model not found. Run: ollama pull mistral")
                    return True
                else:
                    print("⚠️  Ollama not running. Run: ollama serve")
                    return False
            except:
                print("⚠️  Ollama not running. Run: ollama serve")
                return False
        else:
            print("❌ Ollama not installed. Visit: https://ollama.ai")
            return False
    except FileNotFoundError:
        print("❌ Ollama not installed. Visit: https://ollama.ai")
        return False

def check_dependencies():
    """Check Python dependencies"""
    required = ["streamlit", "chromadb", "sentence_transformers", "requests", "fastapi"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"✅ {pkg} installed")
        except ImportError:
            print(f"❌ {pkg} missing")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_files():
    """Check required files exist"""
    required_files = [
        "streamlit_app.py",
        "mcp_server.py",
        "requirements.txt",
        "README.md"
    ]
    
    current_dir = Path(__file__).parent
    
    for file in required_files:
        filepath = current_dir / file
        if filepath.exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            return False
    return True

def check_rag_db():
    """Check if RAG database exists"""
    db_path = Path(__file__).parent.parent / "chroma_db"
    
    if db_path.exists():
        print(f"✅ RAG database found at {db_path}")
        return True
    else:
        print(f"⚠️  RAG database not found at {db_path}")
        print("   Run: python langchain+mistral.py to build it")
        return False

def main():
    print("=" * 70)
    print("🎬 MOVIE SCRIPT GENERATOR - SYSTEM CHECK")
    print("=" * 70)
    print()
    
    checks = {
        "Python": check_python(),
        "Ollama & Mistral": check_ollama(),
        "Dependencies": check_dependencies(),
        "Files": check_files(),
        "RAG Database": check_rag_db()
    }
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_good = all(checks.values())
    
    for name, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    
    print()
    
    if all_good:
        print("🎉 All checks passed! You're ready to go!")
        print()
        print("To start the app:")
        print("  ./run.sh")
        print("  OR")
        print("  streamlit run streamlit_app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("Quick fixes:")
        print("  • Install Ollama: brew install ollama")
        print("  • Start Ollama: ollama serve")
        print("  • Pull Mistral: ollama pull mistral")
        print("  • Install deps: pip install -r requirements.txt")
        print("  • Build RAG DB: python ../langchain+mistral.py")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
