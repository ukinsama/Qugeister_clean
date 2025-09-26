# 🌌 Qugeister - Simplified Quantum AI System

A streamlined quantum AI system for Geister game, featuring a clean JSON→Training→Battle workflow.

## 🚀 Live Demo

**🌐 GitHub Pages: [https://ukinsama.github.io/Qugeister_clean/](https://ukinsama.github.io/Qugeister_clean/)**

### Available Tools:
- **🎮 Quantum AI Designer**: Generate structured JSON configurations
- **🔬 Interactive Playground**: Visualize quantum neural networks

## 🎯 Simple Workflow

### **1. Design Configuration**
Open `web/templates/quantum_designer.html` in browser:
- Configure quantum parameters (qubits, layers, embedding)
- Select AI modules (placement, reward, action strategies)
- Download structured JSON configuration

### **2. Train Models**
```bash
python unified_trainer.py configs/your_config.json
```
**Output:**
- `models/[experiment_name]_p1_[timestamp].pth`
- `models/[experiment_name]_p2_[timestamp].pth`

### **3. Battle Models**
```bash
python model_battle.py models/model1.pth models/model2.pth
```
**Output:** Tournament results with win rates and statistics

## 📁 Clean Project Structure
```
📁 Qugeister_clean/
├── 🎨 web/templates/quantum_designer.html   # Configuration generator
├── 🤖 unified_trainer.py                    # JSON → PTH training
├── ⚔️  model_battle.py                      # PTH vs PTH battles
├── 📋 configs/                             # JSON configurations
│   └── *.json
├── 💾 models/                              # Trained models
│   └── *.pth
└── 🗃️  legacy_experiments/                  # Archived files
```

## 🚀 Quick Start

1. **Generate Config:** Open `quantum_designer.html` → Configure → Download
2. **Train Models:** `python unified_trainer.py configs/your_config.json`
3. **Battle Results:** `python model_battle.py models/model1.pth models/model2.pth`

## 🔧 Key Features

- **Structured JSON Configs**: 6-module execution order (placement → quantum → reward → cqcnn → qmap → action)
- **Unified Training**: Single script handles all quantum AI architectures
- **Model Battles**: Direct PTH file comparison system
- **Windows Compatible**: ASCII-only output, no Unicode issues
- **Auto File Management**: Timestamp-based naming and organization

## 🧠 Architecture

### Quantum Processing
- 4-8 qubits with 1-3 layers
- Amplitude or angle embedding
- Configurable entanglement patterns

### Classical-Quantum Hybrid (CQCNN)
- Classical frontend: 252D → Quantum dimension
- Quantum layer: Parameterized circuits
- Classical backend: Quantum output → 36D actions

## 📊 Training System

- **DQN-based**: Deep Q-Network with experience replay
- **Progressive Learning**: Epsilon-greedy exploration
- **Target Networks**: Stable Q-value updates
- **Automatic Saving**: Models saved with timestamps

## 🎮 Example Usage
```bash
# Generate config via web interface
# Downloads: configs/cqcnn_config_2025-09-26.json

# Train models
python unified_trainer.py configs/cqcnn_config_2025-09-26.json
# Creates: models/CQCNN_dqn_standard_4Q1L_p1_20250926_143022.pth

# Battle models
python model_battle.py models/model1_p1.pth models/model2_p1.pth
```

## 🧹 Simplified from Legacy
- Consolidated Copy 6.x experimental systems
- Removed duplicate and experimental files
- Unified complex training pipelines
- Streamlined JSON→PTH workflow

## 🔬 Technical Details

### Prerequisites
```bash
pip install pennylane torch numpy matplotlib
```

### Quantum Architecture
- **Quantum Device**: default.qubit (PennyLane)
- **Circuit Depth**: 1-3 layers
- **Entanglement**: Linear, Circular, or Full connectivity
- **Measurements**: Pauli-Z expectation values

### Training Configuration
- **Episodes**: 1000-5000 (configurable)
- **Batch Size**: 128
- **Learning Rate**: 0.002
- **Replay Buffer**: 10,000 experiences
- **Target Update**: Every 100 steps

---

**A clean, production-ready quantum AI system with minimal complexity.**