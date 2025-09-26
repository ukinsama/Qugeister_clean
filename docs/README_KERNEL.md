# 🚀 Qugeister Jupyter Kernel セットアップガイド

このプロジェクト専用のJupyter Notebookカーネルを作成するためのガイドです。

## 📦 必要なもの

- Python 3.8以上
- pip (Python package installer)
- Jupyter Notebook または JupyterLab

## 🔧 インストール方法

### 方法1: 自動セットアップ（推奨）

Windows環境での簡単セットアップ:

```cmd
# 1. バッチファイルを実行
setup_kernel.bat
```

### 方法2: 手動セットアップ

```cmd
# 1. 必要なパッケージをインストール
pip install jupyter ipykernel pennylane torch numpy matplotlib

# 2. カーネル作成スクリプトを実行
python create_jupyter_kernel.py
```

### 方法3: 既存のインストーラを使用

```cmd
# 既存の高機能インストーラを実行
python install_quantum_kernel.py
```

## 📚 使用方法

### Jupyter Notebook起動

```cmd
jupyter notebook
```

### JupyterLab起動

```cmd
jupyter lab
```

### カーネル選択

1. 新しいノートブックを作成する際に、カーネル選択で **「Qugeister Quantum AI」** を選択
2. 既存のノートブックでカーネルを変更: `Kernel` → `Change Kernel` → `Qugeister Quantum AI`

## 🔍 確認方法

カーネルが正常にインストールされているか確認:

```cmd
jupyter kernelspec list
```

出力に `qugeister_quantum` が含まれていれば成功です。

## 🎯 特徴

- **自動環境設定**: プロジェクトのPythonパスが自動設定
- **ライブラリ事前読み込み**: NumPy、PyTorch、PennyLaneなどが自動インポート
- **キャッシュ最適化**: 各ライブラリのキャッシュディレクトリを最適化
- **デバッグサポート**: IPythonデバッガーが利用可能
- **WebUI連携**: Quantum Designer設定ファイルとの連携

## 📁 作成されるファイル

```
Qugeister_clean/
├── create_jupyter_kernel.py    # カーネル作成スクリプト
├── setup_kernel.bat           # Windows用自動セットアップ
├── qugeister_startup.py       # 環境初期化スクリプト
├── kernel_spec.json          # カーネル仕様ファイル
└── README_KERNEL.md          # このファイル
```

## 🎨 Jupyter環境での利用例

```python
# 自動で実行される初期化（qugeister_startup.py）
🚀 Qugeister Quantum AI Environment Setup
==================================================
📁 Project Root: C:\Users\KS\Qugeister_clean
✅ Added to PYTHONPATH: C:\Users\KS\Qugeister_clean
✅ Added to PYTHONPATH: C:\Users\KS\Qugeister_clean\src

# ライブラリを使用
import torch
import pennylane as qml
from qugeister.quantum import QuantumTrainer
from qugeister.core import GameEngine

# 量子回路の作成例
dev = qml.device("lightning.qubit", wires=4)

@qml.qnode(dev)
def quantum_circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
```

## 🛠️ トラブルシューティング

### カーネルが表示されない場合

```cmd
# カーネル一覧を確認
jupyter kernelspec list

# カーネルを削除して再インストール
jupyter kernelspec remove qugeister_quantum
python create_jupyter_kernel.py
```

### Python実行可能ファイルが見つからない場合

`create_jupyter_kernel.py` 内の `find_python_executable()` 関数で、
お使いの環境のPythonパスを追加してください。

### パッケージが見つからない場合

```cmd
# 必要なパッケージを個別インストール
pip install torch pennylane numpy matplotlib jupyter ipykernel
```

## 🔄 アンインストール

```cmd
# カーネルを削除
jupyter kernelspec remove qugeister_quantum

# 作成されたファイルを削除
del create_jupyter_kernel.py
del setup_kernel.bat
del qugeister_startup.py
del kernel_spec.json
```

## 💡 高度な設定

### カスタム環境変数の追加

`kernel_spec.json` の `env` セクションに追加:

```json
{
  "env": {
    "CUSTOM_VAR": "your_value",
    "QUGEISTER_DEBUG": "true"
  }
}
```

### 追加ライブラリの自動インポート

`qugeister_startup.py` に追加:

```python
try:
    import your_library
    print(f"✅ Your Library {your_library.__version__}")
except ImportError as e:
    print(f"⚠️ Your Library: {e}")
```

---

🎉 **Qugeister Quantum AI Kernel をお楽しみください！** 🌌