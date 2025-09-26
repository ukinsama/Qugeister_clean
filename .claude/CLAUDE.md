# Claude Code Instructions

## Coding Guidelines for this Project

### Character Encoding Rules
- **NEVER use Unicode emoji characters** in Python code (🔄, ✅, ⚡, 📊, 🎯, 📈, 🔥, 🚀, etc.)
- **Use ASCII characters only** for all print statements and comments
- **Replace arrows** with ASCII: use `->` instead of `→`
- **Use English comments** to avoid cp932 encoding issues on Windows

### Reason
This project runs on Windows with cp932 codec, which cannot encode Unicode emoji characters, causing `UnicodeEncodeError: 'cp932' codec can't encode character` errors.

### Safe Alternatives
- Instead of 🔄: use "PHASE" or "TRANSITION"
- Instead of ✅: use "OK" or "LOADED"
- Instead of 📊: use "STATS" or "RESULTS"
- Instead of →: use "->"

### Example
```python
# BAD - causes encoding error
print(f"🔄 Phase Transition: {old} → {new}")

# GOOD - ASCII only
print(f"PHASE TRANSITION: {old} -> {new}")
```

## Training Commands
- To run quantum reinforcement learning: `python copy61_improved_experiment.py`
- Models will be saved as: `copy61_p1_model_[timestamp].pth` and `copy61_p2_model_[timestamp].pth`
- Battle viewer: `python copy61_real_battle_viewer.py`

## Future Development Plan: index.html Enhancement

### Current Status
- The current `index.html` is very well-organized and structured
- Should be expanded as the main interface for the project

### Enhancement Strategy
- **Expand the organized index.html** as the primary project interface
- Add sections for:
  - Quantum AI experiment results and comparisons
  - Interactive model battle viewer integration
  - Training progress monitoring dashboard
  - Configuration management interface
  - Real-time learning analytics visualization
- Maintain the current clean, organized structure
- Consider adding web-based controls for experiment management