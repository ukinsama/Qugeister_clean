# 廃止ファイル一覧

以下のファイルは**ランダム報酬シミュレーション**を含むため廃止されました。
実際のゲーム学習には `real_geister_experiment.py` を使用してください。

## 廃止されたファイル:

### 1. メイン量子トレーナー
- `src/qugeister/quantum/quantum_trainer.py`
  - `train_fast_quantum()` 関数: ランダム報酬使用
  - `train_convergence_test()` 関数: ランダム報酬使用
  - ⚠️ WARNING付きで残存（参考のため）

### 2. レガシーファイル
- `legacy/qugeister_ai_system/ai_maker_system/learning/reinforcement.py`
- `legacy/experiments/fast_quantum_trainer.py`

### 3. Jupyter ノートブック
- `quantum_ai_tutorial.ipynb`
- `quantum_ai_training_clean.ipynb`

### 4. ウェブテンプレート
- `web/templates/quantum_designer_backup.html`
- `web/templates/quantum_designer_refactored.html`

## 代替手段

**実際のGeister学習**: `real_geister_experiment.py`
- 真のゲームロジック
- 戦略的報酬システム
- 実際の勝敗判定
- 量子AI戦略学習

## 今後の学習について

「学習」と言ったら、必ず実際のGeisterゲームプレイを通した学習を指すものとします。
ランダム報酬シミュレーションは一切使用しません。