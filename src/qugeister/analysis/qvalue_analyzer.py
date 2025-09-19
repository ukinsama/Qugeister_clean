#!/usr/bin/env python3
"""
Q-Value Complete Output System

Complete visualization and analysis of Q-values for all board states
in the Geister game, providing comprehensive strategic insights.
"""

from collections import defaultdict
import itertools
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle

class GeisterStateEncoder:
    """Encoder for Geister board states"""
    
    def __init__(self):
        self.board_size = 6
        self.channels = 7  # 7チャンネルシステム
        
    def encode_full_state(self, board_state, player_pieces, opponent_pieces, 
                         legal_moves, escape_positions):
        """完全な盤面状態を252次元ベクトルにエンコード"""
        
        # 各チャンネルの初期化
        channels = torch.zeros(self.channels, self.board_size, self.board_size)
        
        # チャンネル0: 自分の善玉駒
        for pos, piece_type in player_pieces.items():
            if piece_type == 'good':
                y, x = pos
                channels[0, y, x] = 1.0
        
        # チャンネル1: 自分の悪玉駒
        for pos, piece_type in player_pieces.items():
            if piece_type == 'bad':
                y, x = pos
                channels[1, y, x] = 1.0
        
        # チャンネル2: 相手の駒（種類不明）
        for pos in opponent_pieces:
            y, x = pos
            channels[2, y, x] = 1.0
        
        # チャンネル3: 既知の相手善玉駒
        # （実際のゲームでは推定情報）
        
        # チャンネル4: 既知の相手悪玉駒
        # （実際のゲームでは推定情報）
        
        # チャンネル5: 合法手
        for move in legal_moves:
            if len(move) >= 2:
                y, x = move[1]  # 移動先
                channels[5, y, x] = 1.0
        
        # チャンネル6: 脱出位置
        for pos in escape_positions:
            y, x = pos
            channels[6, y, x] = 1.0
        
        # 252次元ベクトルに変換
        return channels.flatten()

class QValueFullOutputModule:
    """Q値マップ完全出力モジュール"""
    
    def __init__(self, model_path='fast_quantum_model.pth'):
        self.model_path = model_path
        self.encoder = GeisterStateEncoder()
        self.load_model()
        
        # 分析結果保存用
        self.results_dir = Path('qvalue_analysis_results')
        self.results_dir.mkdir(exist_ok=True)
    
    def load_model(self):
        """学習済み量子モデルを読み込み"""
        try:
            from fast_quantum_trainer import FastQuantumNeuralNetwork
            
            checkpoint = torch.load(self.model_path)
            self.model = FastQuantumNeuralNetwork(n_qubits=4, output_dim=36)  # 36次元Q値マップ出力
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ モデル読み込み成功: {self.model_path} (36次元Q値マップモード)")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            self.model = None
    
    def generate_all_possible_states(self, max_states=1000):
        """可能な盤面状態を生成（サンプリング版）"""
        states = []
        state_descriptions = []
        
        print(f"🎲 {max_states}個の代表的盤面状態を生成中...")
        
        for i in range(max_states):
            # ランダムな盤面状態を生成
            player_pieces = {}
            opponent_pieces = []
            
            # プレイヤーの駒をランダム配置（4個ずつ）
            positions = [(x, y) for x in range(6) for y in range(4, 6)]  # 下側2行
            selected_positions = np.random.choice(len(positions), size=4, replace=False)
            
            for idx, pos_idx in enumerate(selected_positions):
                pos = positions[pos_idx]
                piece_type = 'good' if idx < 2 else 'bad'  # 善玉2個、悪玉2個
                player_pieces[f"{pos[0]}_{pos[1]}"] = piece_type  # タプルを文字列に変換
            
            # 相手の駒をランダム配置
            opponent_positions = [(x, y) for x in range(6) for y in range(0, 2)]  # 上側2行
            selected_opp = np.random.choice(len(opponent_positions), size=4, replace=False)
            for pos_idx in selected_opp:
                pos = opponent_positions[pos_idx]
                opponent_pieces.append(f"{pos[0]}_{pos[1]}")  # タプルを文字列に変換
            
            # 合法手の生成（簡略版）
            legal_moves = []
            for pos_str in player_pieces.keys():
                x, y = map(int, pos_str.split('_'))
                # 基本的な移動（上下左右）
                for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                    new_y, new_x = y + dy, x + dx
                    if 0 <= new_y < 6 and 0 <= new_x < 6:
                        legal_moves.append(f"{x}_{y}_to_{new_x}_{new_y}")
            
            # 脱出位置
            escape_positions = ["0_0", "0_5", "5_0", "5_5"]
            
            # 状態をエンコード（簡略版）
            state_vector = torch.randn(252)  # 実際の実装では詳細なエンコードが必要
            
            states.append(state_vector)
            state_descriptions.append({
                'id': i,
                'player_pieces': player_pieces,
                'opponent_pieces': opponent_pieces,
                'legal_moves_count': len(legal_moves)
            })
        
        print(f"✅ {len(states)}個の盤面状態を生成完了")
        return states, state_descriptions
    
    def compute_full_qvalue_map(self, states, state_descriptions):
        """全状態に対するQ値マップを計算（36次元出力対応）"""
        if self.model is None:
            print("❌ モデルが読み込まれていません")
            return None
        
        print("🧠 36次元Q値マップを計算中...")
        qvalue_map = {}
        all_qvalue_maps = []  # 36次元Q値マップ
        all_action_qvalues = []  # 5行動Q値
        
        with torch.no_grad():
            for i, (state, desc) in enumerate(zip(states, state_descriptions)):
                # Q値マップを計算（36次元）
                state_tensor = state.unsqueeze(0)
                qvalue_map_36d = self.model(state_tensor).squeeze().numpy()  # 36次元出力
                
                # 6x6マップに変換
                qvalue_spatial = qvalue_map_36d.reshape(6, 6)
                
                # 5行動Q値を計算
                action_qvalues = self.model.get_action_from_qmap(state_tensor).squeeze().numpy()
                
                qvalue_map[i] = {
                    'description': desc,
                    'qvalue_map_36d': qvalue_map_36d,  # 36次元Q値マップ
                    'qvalue_spatial': qvalue_spatial,  # 6x6空間マップ
                    'action_qvalues': action_qvalues,  # 5行動Q値
                    'best_action': int(np.argmax(action_qvalues)),
                    'qvalue_variance': float(np.var(qvalue_map_36d)),
                    'max_qvalue': float(np.max(qvalue_map_36d)),
                    'min_qvalue': float(np.min(qvalue_map_36d)),
                    'spatial_hotspots': self._find_spatial_hotspots(qvalue_spatial)
                }
                
                all_qvalue_maps.append(qvalue_map_36d)
                all_action_qvalues.append(action_qvalues)
                
                if (i + 1) % 100 == 0:
                    print(f"   進捗: {i+1}/{len(states)} ({100*(i+1)/len(states):.1f}%)")
        
        # 統計情報を計算
        all_qvalue_maps = np.array(all_qvalue_maps)  # (N, 36)
        all_action_qvalues = np.array(all_action_qvalues)  # (N, 5)
        
        statistics = {
            'total_states': len(states),
            'qvalue_map_stats': {
                'mean': float(np.mean(all_qvalue_maps)),
                'std': float(np.std(all_qvalue_maps)),
                'range': [float(np.min(all_qvalue_maps)), float(np.max(all_qvalue_maps))],
                'spatial_variance': float(np.mean([np.var(qmap.reshape(6, 6)) for qmap in all_qvalue_maps]))
            },
            'action_distribution': np.bincount(
                [qvalue_map[i]['best_action'] for i in range(len(states))], minlength=5
            ).tolist(),
            'action_qvalue_stats': {
                'mean': float(np.mean(all_action_qvalues)),
                'std': float(np.std(all_action_qvalues)),
                'range': [float(np.min(all_action_qvalues)), float(np.max(all_action_qvalues))]
            }
        }
        
        print("✅ 36次元Q値マップ計算完了")
        return qvalue_map, statistics, all_qvalue_maps, all_action_qvalues
    
    def _find_spatial_hotspots(self, spatial_map):
        """6x6空間マップから注目領域を特定"""
        threshold = np.mean(spatial_map) + np.std(spatial_map)
        hotspots = []
        
        for i in range(6):
            for j in range(6):
                if spatial_map[i, j] > threshold:
                    hotspots.append((i, j, float(spatial_map[i, j])))
        
        return sorted(hotspots, key=lambda x: x[2], reverse=True)[:5]  # トップ5を返す
    
    def visualize_qvalue_heatmap(self, all_qvalues, save_path=None):
        """Q値のヒートマップを生成"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Q値マップ完全出力 - 行動別ヒートマップ', fontsize=16)
        
        action_names = ['上移動', '右移動', '下移動', '左移動', '脱出']
        
        for i, action_name in enumerate(action_names):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            
            # 各行動のQ値分布
            action_qvalues = all_qvalues[:, i]
            
            # ヒストグラム
            ax.hist(action_qvalues, bins=50, alpha=0.7, color=f'C{i}')
            ax.set_title(f'{action_name}\n(平均: {np.mean(action_qvalues):.3f})')
            ax.set_xlabel('Q値')
            ax.set_ylabel('頻度')
            ax.grid(True, alpha=0.3)
        
        # 最後のサブプロットは全体統計
        axes[1, 2].remove()
        ax_stats = fig.add_subplot(2, 3, 6)
        
        # 行動選択分布
        best_actions = np.argmax(all_qvalues, axis=1)
        action_counts = np.bincount(best_actions, minlength=5)
        
        bars = ax_stats.bar(range(5), action_counts, color=['C0', 'C1', 'C2', 'C3', 'C4'])
        ax_stats.set_title('最適行動の分布')
        ax_stats.set_xlabel('行動')
        ax_stats.set_ylabel('選択回数')
        ax_stats.set_xticks(range(5))
        ax_stats.set_xticklabels(['上', '右', '下', '左', '脱出'])
        
        # 各バーの上に値を表示
        for bar, count in zip(bars, action_counts):
            ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ヒートマップ保存: {save_path}")
        
        plt.show()
    
    def visualize_spatial_qvalue_maps(self, qvalue_map, save_path=None, num_samples=9):
        """6x6空間Q値マップの可視化"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('6x6 空間Q値マップサンプル（36次元出力）', fontsize=16)
        
        # サンプル状態を選択
        sample_ids = list(qvalue_map.keys())[:num_samples]
        
        for idx, state_id in enumerate(sample_ids):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            
            # 6x6空間マップを取得
            spatial_map = qvalue_map[state_id]['qvalue_spatial']
            
            # ヒートマップ描画
            im = ax.imshow(spatial_map, cmap='viridis', aspect='equal')
            ax.set_title(f'状態 {state_id}\nMax Q-value: {qvalue_map[state_id]["max_qvalue"]:.3f}')
            
            # グリッド表示
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.grid(True, color='white', linewidth=0.5)
            
            # 値を表示
            for i in range(6):
                for j in range(6):
                    text = ax.text(j, i, f'{spatial_map[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)
            
            # カラーバー
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Q値', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            spatial_path = save_path.replace('.png', '_spatial.png')
            plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
            print(f"🗺️  空間マップ保存: {spatial_path}")
        
        plt.show()
        
    def visualize_action_vs_spatial_comparison(self, qvalue_map, all_action_qvalues, save_path=None):
        """5行動Q値 vs 36次元空間マップの比較可視化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('5行動Q値 vs 36次元空間Q値マップ比較', fontsize=16)
        
        # 1. 行動Q値分布
        action_names = ['上', '右', '下', '左', '脱出']
        for i, name in enumerate(action_names):
            ax1.hist(all_action_qvalues[:, i], alpha=0.6, label=name, bins=30)
        ax1.set_title('5行動Q値分布')
        ax1.set_xlabel('Q値')
        ax1.set_ylabel('頻度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 空間Q値の位置別平均
        all_spatial_maps = np.array([data['qvalue_spatial'] for data in qvalue_map.values()])
        avg_spatial = np.mean(all_spatial_maps, axis=0)
        
        im2 = ax2.imshow(avg_spatial, cmap='viridis', aspect='equal')
        ax2.set_title('6x6空間Q値の平均マップ')
        ax2.set_xticks(range(6))
        ax2.set_yticks(range(6))
        
        # 値を表示
        for i in range(6):
            for j in range(6):
                ax2.text(j, i, f'{avg_spatial[i, j]:.2f}',
                        ha="center", va="center", color="white", fontsize=10)
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 3. 最適行動選択の比較
        best_actions_5d = [np.argmax(qvals) for qvals in all_action_qvalues]
        best_positions_36d = []
        
        for data in qvalue_map.values():
            spatial = data['qvalue_spatial']
            max_pos = np.unravel_index(np.argmax(spatial), spatial.shape)
            # 6x6座標を0-35のインデックスに変換
            best_positions_36d.append(max_pos[0] * 6 + max_pos[1])
        
        # 5行動分布
        action_counts = np.bincount(best_actions_5d, minlength=5)
        ax3.bar(range(5), action_counts, color=['C0', 'C1', 'C2', 'C3', 'C4'])
        ax3.set_title('5行動システム: 最適行動分布')
        ax3.set_xlabel('行動')
        ax3.set_ylabel('選択回数')
        ax3.set_xticks(range(5))
        ax3.set_xticklabels(action_names)
        
        # 36位置分布（トップ10のみ表示）
        position_counts = np.bincount(best_positions_36d, minlength=36)
        top_positions = np.argsort(position_counts)[-10:]
        
        ax4.bar(range(len(top_positions)), position_counts[top_positions])
        ax4.set_title('36次元システム: 最適位置分布（トップ10）')
        ax4.set_xlabel('位置インデックス')
        ax4.set_ylabel('選択回数')
        ax4.set_xticks(range(len(top_positions)))
        ax4.set_xticklabels([f'({pos//6},{pos%6})' for pos in top_positions], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            comparison_path = save_path.replace('.png', '_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"📊 比較分析保存: {comparison_path}")
        
        plt.show()
    
    def analyze_strategic_patterns(self, qvalue_map):
        """戦略パターンを分析"""
        print("🔍 戦略パターン分析中...")
        
        patterns = {
            'aggressive': [],    # 攻撃的（前進重視）
            'defensive': [],     # 守備的（後退重視）
            'escape_focused': [], # 脱出重視
            'balanced': []       # バランス型
        }
        
        for state_id, data in qvalue_map.items():
            qvalues = data['qvalues']
            best_action = data['best_action']
            
            # パターン分類
            if best_action == 0:  # 上移動（攻撃的）
                patterns['aggressive'].append(state_id)
            elif best_action == 2:  # 下移動（守備的）
                patterns['defensive'].append(state_id)
            elif best_action == 4:  # 脱出
                patterns['escape_focused'].append(state_id)
            else:
                patterns['balanced'].append(state_id)
        
        # パターン統計
        pattern_stats = {}
        for pattern_name, state_ids in patterns.items():
            if state_ids:
                qvals = [qvalue_map[sid]['qvalues'] for sid in state_ids]
                pattern_stats[pattern_name] = {
                    'count': len(state_ids),
                    'percentage': len(state_ids) / len(qvalue_map) * 100,
                    'avg_max_qvalue': float(np.mean([np.max(q) for q in qvals])),
                    'avg_variance': float(np.mean([np.var(q) for q in qvals]))
                }
        
        return pattern_stats
    
    def export_results(self, qvalue_map, statistics, pattern_stats, all_qvalue_maps, all_action_qvalues):
        """結果をファイルに出力"""
        
        # 1. JSON形式で詳細データを保存
        json_path = self.results_dir / 'qvalue_full_output.json'
        export_data = {
            'statistics': statistics,
            'pattern_analysis': pattern_stats,
            'sample_states': {
                str(i): {
                    'description': data['description'],
                    'qvalue_map_36d': data['qvalue_map_36d'].tolist(),
                    'qvalue_spatial': data['qvalue_spatial'].tolist(),
                    'action_qvalues': data['action_qvalues'].tolist(),
                    'best_action': data['best_action'],
                    'max_qvalue': data['max_qvalue'],
                    'spatial_hotspots': data['spatial_hotspots']
                }
                for i, data in list(qvalue_map.items())[:100]  # 最初の100個をサンプル
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON結果保存: {json_path}")
        
        # 2. Pickle形式で完全データを保存
        pickle_path = self.results_dir / 'qvalue_complete_data.pkl'
        complete_data = {
            'qvalue_map': qvalue_map,
            'statistics': statistics,
            'pattern_stats': pattern_stats,
            'all_qvalue_maps': all_qvalue_maps,
            'all_action_qvalues': all_action_qvalues
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(complete_data, f)
        print(f"🗄️  完全データ保存: {pickle_path}")
        
        # 3. CSV形式でサマリーを保存
        csv_path = self.results_dir / 'qvalue_summary.csv'
        with open(csv_path, 'w') as f:
            f.write("state_id,best_action,max_qvalue,qvalue_variance,legal_moves_count\\n")
            for i, data in qvalue_map.items():
                f.write(f"{i},{data['best_action']},{data['max_qvalue']:.4f},"
                       f"{data['qvalue_variance']:.4f},{data['description']['legal_moves_count']}\\n")
        print(f"📊 CSVサマリー保存: {csv_path}")
        
        # 4. 可視化結果を保存
        heatmap_path = self.results_dir / 'qvalue_heatmap.png'
        self.visualize_qvalue_heatmap(all_action_qvalues, heatmap_path)
        
        # 5. 新しい36次元可視化も保存
        self.visualize_spatial_qvalue_maps(qvalue_map, heatmap_path)
        self.visualize_action_vs_spatial_comparison(qvalue_map, all_action_qvalues, heatmap_path)
    
    def run_full_analysis(self, num_states=1000):
        """完全なQ値マップ分析を実行"""
        print("=" * 60)
        print("🎯 モジュール4: Q値マップ完全出力システム")
        print("=" * 60)
        
        # ステップ1: 盤面状態生成
        states, state_descriptions = self.generate_all_possible_states(num_states)
        
        # ステップ2: Q値計算
        qvalue_map, statistics, all_qvalue_maps, all_action_qvalues = self.compute_full_qvalue_map(
            states, state_descriptions
        )
        
        # ステップ3: 戦略パターン分析
        pattern_stats = self.analyze_strategic_patterns(qvalue_map)
        
        # ステップ4: 結果表示
        print("\\n📈 36次元Q値マップ分析結果:")
        print(f"  総状態数: {statistics['total_states']}")
        print(f"  36次元Q値範囲: [{statistics['qvalue_map_stats']['range'][0]:.3f}, {statistics['qvalue_map_stats']['range'][1]:.3f}]")
        print(f"  36次元Q値平均: {statistics['qvalue_map_stats']['mean']:.3f} ± {statistics['qvalue_map_stats']['std']:.3f}")
        print(f"  空間分散平均: {statistics['qvalue_map_stats']['spatial_variance']:.4f}")
        
        print("\\n📊 5行動Q値統計:")
        print(f"  5行動Q値範囲: [{statistics['action_qvalue_stats']['range'][0]:.3f}, {statistics['action_qvalue_stats']['range'][1]:.3f}]")
        print(f"  5行動Q値平均: {statistics['action_qvalue_stats']['mean']:.3f} ± {statistics['action_qvalue_stats']['std']:.3f}")
        
        print("\\n🎯 戦略パターン分布:")
        for pattern, stats in pattern_stats.items():
            print(f"  {pattern}: {stats['count']}個 ({stats['percentage']:.1f}%)")
        
        print("\\n🎮 最適行動分布:")
        action_names = ['上移動', '右移動', '下移動', '左移動', '脱出']
        for i, (name, count) in enumerate(zip(action_names, statistics['action_distribution'])):
            percentage = count / statistics['total_states'] * 100
            print(f"  {name}: {count}回 ({percentage:.1f}%)")
        
        # ステップ5: 結果出力
        self.export_results(qvalue_map, statistics, pattern_stats, all_qvalue_maps, all_action_qvalues)
        
        print("\\n✅ Q値マップ完全出力完了!")
        print(f"📁 結果保存先: {self.results_dir}")
        
        return qvalue_map, statistics, pattern_stats

def main():
    """メイン実行関数"""
    
    # 設定選択
    print("🎯 モジュール4: Q値マップ完全出力システム")
    print("\\n分析規模を選択してください:")
    print("1. クイック分析 (100状態)")
    print("2. 標準分析 (1000状態)")
    print("3. 詳細分析 (5000状態)")
    print("4. フル分析 (10000状態)")
    
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\\n選択 (1-4): ").strip()
    
    # 状態数設定
    state_counts = {'1': 100, '2': 1000, '3': 5000, '4': 10000}
    num_states = state_counts.get(choice, 1000)
    
    # 分析実行
    analyzer = QValueFullOutputModule()
    
    if analyzer.model is not None:
        qvalue_map, statistics, pattern_stats = analyzer.run_full_analysis(num_states)
        
        print("\\n🚀 次のステップ:")
        print("1. quantum_ai_playground.html で可視化")
        print("2. 3step_system.html でパラメータ調整")
        print("3. IBM Quantum Composer で量子回路設計")
        
    else:
        print("❌ 学習済みモデルが見つかりません")
        print("💡 まず fast_quantum_trainer.py で学習を実行してください")

if __name__ == "__main__":
    main()