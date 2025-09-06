#!/usr/bin/env python3
"""
Quantum Battle System - Mock implementation for test7.py compatibility
"""

import numpy as np
import torch
import torch.nn as nn
from src.qugeister_competitive.game_engine import GeisterGame, GameState


class QuantumBattleSystem:
    """量子バトルシステムのモックバージョン"""
    
    def __init__(self):
        self.game_engine = GeisterGame()
        self.current_state = None
        self.episode_step = 0
        self.max_steps = 100
        
    def reset(self):
        """ゲーム状態をリセット"""
        self.game_engine.reset_game()
        self.current_state = self.game_engine.get_game_state('A')
        self.episode_step = 0
        return self._get_state_vector()
    
    def step(self, action):
        """1ステップ実行"""
        if self.current_state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        
        # アクション（0=上, 1=下, 2=左, 3=右）を移動に変換
        moves = self._get_valid_moves()
        if not moves:
            return self._get_state_vector(), -1, True  # ゲーム終了
        
        # ランダムまたは指定されたアクションで移動選択
        if action < len(moves):
            move = moves[action]
        else:
            move = moves[0]  # デフォルト
        
        # 移動実行
        try:
            self.current_state = self.game_engine.make_move(
                self.current_state, move[0], move[1]
            )
            self.episode_step += 1
            
            # 報酬計算
            reward = self._calculate_reward()
            
            # 終了条件チェック
            done = (self.episode_step >= self.max_steps or 
                   self.game_engine.is_game_over(self.current_state))
            
            return self._get_state_vector(), reward, done
            
        except Exception:
            return self._get_state_vector(), -1, True
    
    def _get_state_vector(self):
        """現在の状態をベクトルに変換"""
        if self.current_state is None:
            return torch.zeros(36)  # 6x6ボード
        
        # ボード状態をフラット化
        board_flat = self.current_state.board.flatten()
        return torch.tensor(board_flat, dtype=torch.float32)
    
    def _get_valid_moves(self):
        """有効な移動リストを取得"""
        if self.current_state is None:
            return []
        
        moves = []
        # 現在のプレイヤーの駒位置を取得
        player_pieces = (self.current_state.player_a_pieces 
                        if self.current_state.current_player == 'A' 
                        else self.current_state.player_b_pieces)
        
        # 各駒について有効な移動を探す
        for pos in player_pieces.keys():
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if self._is_valid_move(pos, new_pos):
                    moves.append((pos, new_pos))
        
        return moves
    
    def _is_valid_move(self, from_pos, to_pos):
        """移動が有効かチェック"""
        x, y = to_pos
        if x < 0 or x >= 6 or y < 0 or y >= 6:
            return False
        
        # 自分の駒がある場所には移動できない
        player_pieces = (self.current_state.player_a_pieces 
                        if self.current_state.current_player == 'A' 
                        else self.current_state.player_b_pieces)
        
        return to_pos not in player_pieces
    
    def _calculate_reward(self):
        """報酬を計算"""
        if self.current_state is None:
            return 0
        
        # 基本報酬
        reward = 0.01  # 生存報酬
        
        # ゲーム終了時の報酬
        if self.game_engine.is_game_over(self.current_state):
            winner = self.game_engine.get_winner(self.current_state)
            if winner == self.current_state.current_player:
                reward += 10  # 勝利報酬
            else:
                reward -= 5   # 敗北ペナルティ
        
        return reward


if __name__ == "__main__":
    # テスト実行
    system = QuantumBattleSystem()
    state = system.reset()
    print(f"Initial state shape: {state.shape}")
    
    for i in range(10):
        action = np.random.randint(0, 4)
        next_state, reward, done = system.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")
        if done:
            break