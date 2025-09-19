#!/usr/bin/env python3
"""
修正されたゲームエンジンで学習されたテストAI
"""

import sys
import os
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.qugeister_competitive.ai_base import BaseAI
import random

class CorrectedEngineAI(BaseAI):
    """修正エンジンで学習されたAI"""
    
    def __init__(self, player_id="B"):
        super().__init__("CorrectedEngineAI", player_id)
        
    def get_move(self, game_state, legal_moves):
        """手を選択（修正されたエンジンに対応）"""
        if not legal_moves:
            return None
            
        # 脱出可能な手を最優先
        escape_moves = [move for move in legal_moves if isinstance(move[1], str) and move[1] == "ESCAPE"]
        if escape_moves:
            return escape_moves[0]
        
        # 相手駒を取る手を優先
        attack_moves = []
        opponent_pieces = game_state.player_a_pieces if self.player_id == "B" else game_state.player_b_pieces
        
        for move in legal_moves:
            from_pos, to_pos = move
            if to_pos in opponent_pieces:
                attack_moves.append(move)
        
        if attack_moves:
            return random.choice(attack_moves)
        
        # 前進手を優先（修正された座標系に対応）
        advance_moves = []
        for move in legal_moves:
            from_pos, to_pos = move
            if self.player_id == "A":
                # プレイヤーAは上方向（y座標減少）へ前進
                if to_pos[1] < from_pos[1]:
                    advance_moves.append(move)
            else:
                # プレイヤーBは下方向（y座標増加）へ前進
                if to_pos[1] > from_pos[1]:
                    advance_moves.append(move)
        
        if advance_moves:
            return random.choice(advance_moves)
        
        # それ以外はランダム
        return random.choice(legal_moves)
    
    def choose_action(self, game_state):
        """アクション選択（human_vs_ai_battle.py用）"""
        legal_moves = game_state.get_legal_moves(self.player_id)
        return self.get_move(game_state, legal_moves)
