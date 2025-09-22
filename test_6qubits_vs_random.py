#!/usr/bin/env python3
"""
6量子ビットAI vs ランダムAI 対戦テスト
"""

import sys
from pathlib import Path
import torch
import random

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from test_qubits_6 import QuantumBattleAI_6Qubits
from qugeister import GeisterEngine

print('🎮 6量子ビットAI vs ランダムAI 対戦テスト')
print('=' * 60)

def test_vs_random(model_path, n_games=100, load_model=True):
    """
    6量子ビットAI vs ランダムAIの対戦テスト
    """
    # AI初期化
    ai = QuantumBattleAI_6Qubits(name='6qubits_vs_random', player_id='A')
    
    if load_model and Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            ai.load_state_dict(checkpoint['model_state_dict'])
            print(f'✅ 学習済みモデル読み込み: {model_path}')
            print(f'📊 学習時勝率: {checkpoint.get("final_win_rate", "N/A")}')
        except Exception as e:
            print(f'⚠️ モデル読み込み失敗: {e}')
            print('未学習状態でテスト実行')
    else:
        print('📝 未学習状態でテスト実行')
    
    ai.eval()  # 評価モード
    
    # 対戦統計
    ai_wins = 0
    random_wins = 0
    draws = 0
    total_moves = []
    
    print(f'🏁 {n_games}ゲーム対戦開始...')
    
    for game_idx in range(n_games):
        game = GeisterEngine()
        game.reset_game()
        
        move_count = 0
        max_moves = 100  # ターン制限
        
        with torch.no_grad():
            while not game.game_over and move_count < max_moves:
                current_player = game.current_player
                legal_moves = game.get_legal_moves(current_player)
                
                if not legal_moves:
                    break
                
                if current_player == 'A':  # 6量子ビットAI
                    try:
                        move = ai.get_move(game, legal_moves)
                        if move not in legal_moves:
                            move = random.choice(legal_moves)
                    except Exception as e:
                        print(f'AI手選択エラー (ゲーム{game_idx+1}): {e}')
                        move = random.choice(legal_moves)
                else:  # ランダムAI
                    move = random.choice(legal_moves)
                
                from_pos, to_pos = move
                game.make_move(from_pos, to_pos)
                move_count += 1
        
        # 結果集計
        if game.winner == 'A':
            ai_wins += 1
        elif game.winner == 'B':
            random_wins += 1
        else:
            draws += 1
        
        total_moves.append(move_count)
        
        # 進捗表示
        if (game_idx + 1) % 20 == 0:
            current_ai_rate = ai_wins / (game_idx + 1) * 100
            print(f'進捗 {game_idx+1}/{n_games}: AI勝率 {current_ai_rate:.1f}%')
    
    # 最終結果
    ai_win_rate = ai_wins / n_games * 100
    random_win_rate = random_wins / n_games * 100
    draw_rate = draws / n_games * 100
    avg_moves = sum(total_moves) / len(total_moves)
    
    print(f'\n🏆 最終結果 ({n_games}ゲーム):')
    print('=' * 40)
    print(f'6量子ビットAI: {ai_wins}勝 ({ai_win_rate:.1f}%)')
    print(f'ランダムAI:     {random_wins}勝 ({random_win_rate:.1f}%)')
    print(f'引き分け:       {draws}回 ({draw_rate:.1f}%)')
    print(f'平均手数:       {avg_moves:.1f}手')
    
    # 詳細統計
    print(f'\n📊 詳細統計:')
    print(f'最短ゲーム: {min(total_moves)}手')
    print(f'最長ゲーム: {max(total_moves)}手')
    
    # 性能評価
    if ai_win_rate > 55:
        print('✅ AI性能: 良好 (ランダムより有意に強い)')
    elif ai_win_rate > 45:
        print('📊 AI性能: 普通 (ランダムと同程度)')
    else:
        print('❌ AI性能: 要改善 (ランダムより弱い)')
    
    return {
        'ai_wins': ai_wins,
        'random_wins': random_wins,
        'draws': draws,
        'ai_win_rate': ai_win_rate,
        'avg_moves': avg_moves
    }

# テスト実行
print('🧪 テスト1: 学習済みモデル (3000エピソード)')
results_trained = test_vs_random('trained_6qubits_extended_3000ep.pth', n_games=50)

print('\n' + '='*60)
print('🧪 テスト2: 未学習モデル (比較用)')
results_untrained = test_vs_random('dummy.pth', n_games=50, load_model=False)

print('\n' + '='*60)
print('📈 比較結果:')
print(f'学習済み: {results_trained["ai_win_rate"]:.1f}% vs 未学習: {results_untrained["ai_win_rate"]:.1f}%')
improvement = results_trained['ai_win_rate'] - results_untrained['ai_win_rate']
print(f'改善度: {improvement:+.1f}%')

if improvement > 5:
    print('✅ 学習効果あり')
elif improvement > 0:
    print('📊 学習効果は限定的')
else:
    print('❌ 学習効果なし')

print('\n✅ 対戦テスト完了')