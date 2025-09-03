# 🏆 Qugeister Competition

**3stepデザイナーで設計したAIが学習して対戦する量子インスパイアードAI競技システム**

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ 特徴

- **🎨 3stepデザイナー**: ビジュアルインターフェイスでAIを設計
- **🧠 量子インスパイアード**: 量子回路シミュレーションによる学習
- **⚔️ AI対戦システム**: 学習済みAI同士の自動トーナメント
- **📊 詳細分析**: 学習効果・対戦結果の可視化
- **🚀 簡単セットアップ**: ワンコマンドで環境構築

## 🎮 デモ

```bash
# クイックスタート
git clone https://github.com/[your-username]/Qugeister_clean.git
cd Qugeister_clean
python -m venv qugeister-env
source qugeister-env/bin/activate  # macOS/Linux
pip install -r requirements_minimal.txt
python simple_tournament.py
```

## 📋 システム要件

- Python 3.8以上
- 4GB以上のRAM  
- 1GB以上の空きディスク容量

## 🚀 インストール

### 1. リポジトリクローン
```bash
git clone https://github.com/[your-username]/Qugeister_clean.git
cd Qugeister_clean
```

### 2. 仮想環境セットアップ
```bash
# 仮想環境作成
python -m venv qugeister-env

# 有効化 (macOS/Linux)
source qugeister-env/bin/activate

# 有効化 (Windows)
qugeister-env\Scripts\activate
```

### 3. 依存関係インストール
```bash
pip install -r requirements_minimal.txt
```

### 4. 環境確認
```bash
python environment_check.py
```

## 🎯 使用方法

### 基本的な流れ

1. **AIデザイン** → 2. **学習** → 3. **トーナメント** → 4. **結果分析**

### 1. AIデザイン (3stepデザイナー)

```bash
# ブラウザで3stepデザイナーを開く
open quantum_battle_3step_system.html
```

ビジュアルインターフェイスで：
- 戦略 (攻撃的/防御的/バランス型)
- 量子ビット数・層数
- 学習パラメータ

を選択してAIコードを生成

### 2. 学習実行

```bash
# 生成されたAIを学習 (例: my_ai.py)
cd learning
python recipe_trainer.py --recipe my_ai 100

# バッチ学習 (全AIを自動学習)
python recipe_trainer.py --batch
```

### 3. トーナメント開催

```bash
# 自動トーナメント実行
python run_minimal_tournament.py

# または簡単トーナメント
python simple_tournament.py
```

### 4. 結果確認

```bash
# 最新結果の表示
python tournament/battle_viewer/battle_viewer.py --quick

# 詳細分析
ls tournament_results/
```

## 📁 プロジェクト構造

```
Qugeister_clean/
├── 📋 環境設定
│   ├── SETUP_GUIDE.md              # 詳細セットアップガイド
│   ├── requirements_minimal.txt     # 最小依存関係
│   ├── environment_check.py        # 環境確認スクリプト
│   └── run_minimal_tournament.py   # ワンクリック大会実行
├── 🎨 AIデザイン
│   └── quantum_battle_3step_system.html  # 3stepビジュアルデザイナー
├── 🤖 AIシステム
│   └── qugeister_ai_system/         # モジュラーAI作成システム
│       ├── ai_maker_system/         # AI工場システム
│       ├── tournament_system/       # トーナメント管理
│       ├── 3step_designer/          # デザイナー統合
│       ├── examples/                # 使用例
│       └── integrated_ais/          # サンプルAI
├── 🧠 学習システム
│   └── learning/                    # 学習・訓練システム
│       ├── recipe_trainer.py        # レシピ学習システム
│       └── trained_models/          # 学習済みモデル保存
├── 🏆 トーナメント
│   ├── simple_tournament.py         # 簡単トーナメント実行
│   └── tournament/                  # 高度なトーナメントシステム
└── 📊 結果
    └── tournament_results/          # 大会結果・統計
```

## 🎪 サンプルAI

プリインストールされた3種類のサンプルAI：

- **AggressiveAI** 🗡️: 攻撃的戦略
- **DefensiveAI** 🛡️: 防御的戦略  
- **EscapeAI** 🏃: 逃走重視戦略

## 📊 対戦結果例

```
🏆 トーナメント結果
==================================================
1位: AggressiveAI (勝率: 75.00%, 3/4勝)
2位: DefensiveAI (勝率: 50.00%, 2/4勝)
3位: EscapeAI (勝率: 25.00%, 1/4勝)
```

## 🧪 カスタマイズ

### 新しいAI戦略の追加

```python
# my_custom_ai.py
def get_ai_config():
    return {
        'name': 'my_custom_ai',
        'type': 'quantum_grad',
        'learning_rate': 0.001,
        'epochs': 100,
        'strategy': 'custom'
    }
```

### 学習パラメータの調整

```bash
# エポック数指定
python learning/recipe_trainer.py --recipe my_ai 200

# 学習率調整 (コード内で)
'learning_rate': 0.0001  # より慎重な学習
```

## 🔧 トラブルシューティング

### よくある問題

**NumPy互換性警告**:
```bash
pip install "numpy<2" --force-reinstall
```

**環境チェックでエラー**:
```bash
# 正しいディレクトリにいるか確認
python environment_check.py
```

**モデルが見つからない**:
```bash
# まず学習を実行
python learning/recipe_trainer.py --batch
```

### より詳しいヘルプ

```bash
python learning/recipe_trainer.py --help
python tournament/battle_viewer/battle_viewer.py --help
```

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 🙏 謝辞

- 量子コンピューティングコミュニティ
- PyTorch・PennyLaneの開発チーム
- オープンソース機械学習コミュニティ

## 📞 サポート

- 🐛 バグ報告: [Issues](https://github.com/[your-username]/Qugeister_clean/issues)
- 💡 機能要求: [Issues](https://github.com/[your-username]/Qugeister_clean/issues)
- 📖 ドキュメント: [Wiki](https://github.com/[your-username]/Qugeister_clean/wiki)

---

⚡ **Powered by Quantum-Inspired AI Technology**

Made with ❤️ by [Your Name]