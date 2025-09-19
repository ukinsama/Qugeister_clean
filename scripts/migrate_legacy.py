#!/usr/bin/env python3
"""
Migration script to move from legacy structure to new organized structure.
"""

import shutil
import os
from pathlib import Path


def main():
    """Migrate legacy files to new structure"""
    
    root = Path(__file__).parent.parent
    
    print("🔄 Migrating Qugeister project to new structure...")
    
    # Backup critical files that might be overwritten
    backup_files = [
        "README.md",
        "fast_quantum_model.pth"
    ]
    
    for file in backup_files:
        src = root / file
        if src.exists():
            backup = root / f"{file}.backup"
            shutil.copy2(src, backup)
            print(f"📋 Backed up {file}")
    
    # Move model files to models directory
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_files = ["fast_quantum_model.pth"]
    for model in model_files:
        src = root / model
        if src.exists():
            dst = models_dir / model
            shutil.move(str(src), str(dst))
            print(f"📦 Moved {model} to models/")
    
    # Move results
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Move tournament results
    tournament_src = root / "tournament_results"
    if tournament_src.exists():
        tournament_dst = results_dir / "tournaments"
        shutil.move(str(tournament_src), str(tournament_dst))
        print("🏆 Moved tournament_results to results/tournaments/")
    
    # Move qvalue analysis results  
    qvalue_src = root / "qvalue_analysis_results"
    if qvalue_src.exists():
        qvalue_dst = results_dir / "analysis"
        shutil.move(str(qvalue_src), str(qvalue_dst))
        print("📊 Moved qvalue_analysis_results to results/analysis/")
    
    # Clean up obsolete test files
    obsolete_tests = [
        "test5.py", "test7.py", "test7_fixed.py", "test09.py", "test10.py", 
        "test11.py", "test12.py", "test_3step_generation.py", 
        "test_code_generation_simulation.py", "test_parameter_impact.py"
    ]
    
    tests_dir = root / "tests" / "legacy"
    tests_dir.mkdir(parents=True, exist_ok=True)
    
    for test_file in obsolete_tests:
        src = root / test_file
        if src.exists():
            dst = tests_dir / test_file
            shutil.move(str(src), str(dst))
            print(f"🧪 Moved {test_file} to tests/legacy/")
    
    # Clean up obsolete analysis files
    obsolete_analysis = [
        "rl_loss_analysis.py", "rl_module_flow_analysis.py",
        "design_rationality_analysis.py", "parameter_impact_analysis.json",
        "api_design_benefits.py", "modular_quantum_api_design.py"
    ]
    
    legacy_dir = root / "legacy"
    legacy_dir.mkdir(exist_ok=True)
    
    for file in obsolete_analysis:
        src = root / file
        if src.exists():
            dst = legacy_dir / file
            shutil.move(str(src), str(dst))
            print(f"📚 Moved {file} to legacy/")
    
    # Replace README
    old_readme = root / "README.md"
    new_readme = root / "README_NEW.md"
    
    if new_readme.exists():
        if old_readme.exists():
            shutil.move(str(old_readme), root / "README_OLD.md")
        shutil.move(str(new_readme), str(old_readme))
        print("📄 Updated README.md")
    
    # Clean up empty directories and cache
    cleanup_dirs = ["__pycache__", ".pytest_cache", ".mypy_cache"]
    for cleanup_dir in cleanup_dirs:
        for path in root.rglob(cleanup_dir):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
    
    print("\n✅ Migration completed successfully!")
    print("\n🚀 Next steps:")
    print("1. Run: pip install -e .")  
    print("2. Test: qugeister --help")
    print("3. Train: make train-quick")
    print("4. Analyze: make analyze-quick")
    print("5. Web UI: make web-designer")


if __name__ == "__main__":
    main()