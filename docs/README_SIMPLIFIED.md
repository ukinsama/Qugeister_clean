# Simplified Quantum AI Project

## 🎯 Simple Workflow: JSON → Training → Battle

### **1. Generate Configuration**
Open `web/templates/quantum_designer.html` in browser:
- Configure quantum parameters (qubits, layers, etc.)
- Select modules (placement, reward, action strategy)
- Download structured JSON configuration

### **2. Train Models**
```bash
python unified_trainer.py configs/cqcnn_config_2025-09-26.json
```
**Output:**
- `models/[experiment_name]_p1_[timestamp].pth`
- `models/[experiment_name]_p2_[timestamp].pth`

### **3. Battle Models**
```bash
python model_battle.py models/model1_p1.pth models/model2_p1.pth
```
**Output:** Tournament results with win rates and balance scores

## 📁 Project Structure
```
📁 Qugeister_clean/
├── 🎨 web/templates/quantum_designer.html   # JSON config generator
├── 🤖 unified_trainer.py                    # JSON → PTH training
├── ⚔️  model_battle.py                      # PTH vs PTH battles
├── 📋 configs/                             # JSON configurations
│   └── cqcnn_config_*.json
├── 💾 models/                              # Trained PTH models
│   ├── *_p1_*.pth
│   └── *_p2_*.pth
└── 🗃️  legacy_experiments/                  # Old Copy 6.x files
```

## 🔧 Key Features

### **Unified Trainer**
- Reads structured JSON configurations
- 6-module execution order (placement → quantum → reward → cqcnn → qmap → action)
- Automatic model saving with timestamps
- ASCII-only output (Windows cp932 compatible)

### **Model Battle System**
- Load any PTH models for battles
- Automatic architecture matching
- Tournament statistics
- Balance score analysis

### **Configuration Generator**
- Web-based interface
- Structured module ordering
- Automatic experiment naming
- Download ready-to-use JSON configs

## 🚀 Quick Start

1. **Generate Config:** Open `quantum_designer.html` → Configure → Download JSON
2. **Train Models:** `python unified_trainer.py configs/your_config.json`
3. **Battle Results:** `python model_battle.py models/model1.pth models/model2.pth`

## 📊 Example Usage
```bash
# Step 1: Generate config (via web interface)
# Downloads: configs/cqcnn_config_2025-09-26.json

# Step 2: Train models
python unified_trainer.py configs/cqcnn_config_2025-09-26.json
# Creates: models/CQCNN_dqn_standard_4Q1L_p1_20250926_143022.pth
#          models/CQCNN_dqn_standard_4Q1L_p2_20250926_143022.pth

# Step 3: Battle models
python model_battle.py models/CQCNN_dqn_standard_4Q1L_p1_20250926_143022.pth models/CQCNN_dqn_standard_4Q1L_p2_20250926_143022.pth
```

## 🧹 Cleanup
- Old Copy 6.x experiments moved to `legacy_experiments/`
- All configs organized in `configs/`
- All trained models in `models/`
- Single training script replaces multiple Copy versions

---
**Simplified from complex Copy 6.x experimental system to clean JSON→PTH workflow**