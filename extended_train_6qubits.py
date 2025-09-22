#!/usr/bin/env python3
"""
6量子ビットAI CNN風設計での長期学習（3000エピソード）
"""

import sys
from pathlib import Path
import torch
import time

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from test_qubits_6 import QuantumBattleAI_6Qubits, train_model

print('🧠 6量子ビットAI CNN風設計での長期学習')
print('=' * 60)
print('📊 設定: 3000エピソード学習')

# 長期学習設定
hyperparameters = {
    'epochs': 3000,         # 長期学習
    'batch_size': 8,        
    'learning_rate': 0.0005,  # 少し低めに設定
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'replay_buffer_size': 2000,  # バッファーサイズ増加
    'epsilon': 0.5,         # 探索を多めに
    'epsilon_decay': 0.9995,  # ゆっくりと減衰
    'l2_regularization': 1e-4,
    'gamma': 0.99,          # 長期報酬重視
    'target_update': 20     # ターゲットネットワーク更新頻度
}

start_time = time.time()

try:
    # AI初期化
    ai = QuantumBattleAI_6Qubits(name='6qubits_extended', player_id='A')
    print('✅ AI初期化完了')
    
    # モデル設定を取得
    config = ai.config
    
    print(f'🎯 長期学習開始（{hyperparameters["epochs"]}エピソード）...')
    print('進行状況は1000エピソードごとに表示されます')
    
    # 学習実行
    trained_model, win_rate = train_model(ai, config, hyperparameters)
    
    elapsed_time = time.time() - start_time
    print(f'✅ 学習完了！')
    print(f'📊 最終勝率: {win_rate:.2f}%')
    print(f'⏱️ 学習時間: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分)')
    
    # 学習後テスト
    from qugeister import GeisterEngine
    game = GeisterEngine()
    game.reset_game()
    legal_moves = game.get_legal_moves('A')
    move = trained_model.get_move(game, legal_moves)
    
    print(f'🎮 学習後の手選択: {move}')
    print(f'✓ 有効性: {move in legal_moves}')
    
    # モデル保存
    model_filename = 'trained_6qubits_extended_3000ep.pth'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'hyperparameters': hyperparameters,
        'final_win_rate': win_rate,
        'training_episodes': hyperparameters['epochs'],
        'training_time': elapsed_time
    }, model_filename)
    print(f'💾 学習済みモデル保存完了: {model_filename}')
    
    # 性能比較テスト
    print('\n🏆 性能比較テスト開始...')
    from test_trained_models import test_trained_model
    
    # 従来モデルとの比較
    print('短期学習モデル（30エピソード）との比較:')
    short_win_rate = test_trained_model(QuantumBattleAI_6Qubits, 'trained_6qubits_cnn.pth', '6_short')
    
    print(f'長期学習モデル（3000エピソード）のテスト:')
    long_win_rate = test_trained_model(QuantumBattleAI_6Qubits, model_filename, '6_extended')
    
    print(f'\n📈 改善結果:')
    print(f'短期学習: {short_win_rate:.1f}% → 長期学習: {long_win_rate:.1f}%')
    improvement = long_win_rate - short_win_rate
    print(f'改善度: {improvement:+.1f}%')
    
except Exception as e:
    print(f'❌ エラー: {e}')
    import traceback
    traceback.print_exc()

print('\n✅ 長期学習プロセス完了')