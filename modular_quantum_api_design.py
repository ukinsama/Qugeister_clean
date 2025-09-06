#!/usr/bin/env python3
"""
敵コマ推定器の共通API設計
モジュラーで組み合わせ可能なアーキテクチャ
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class OpponentEstimationResult:
    """敵コマ推定の標準出力フォーマット"""
    piece_probabilities: torch.Tensor  # [8] 各敵コマの善玉確率 (0-1)
    confidence_scores: torch.Tensor    # [8] 各推定の信頼度 (0-1)
    strategy_pattern: Dict[str, float] # 戦略パターン推定
    correlation_matrix: Optional[torch.Tensor] = None  # [8x8] コマ間相関
    temporal_features: Optional[torch.Tensor] = None   # 時系列特徴
    debug_info: Optional[Dict[str, Any]] = None        # デバッグ情報

class EstimatorType(Enum):
    """推定器のタイプ"""
    PROBABILISTIC = "probabilistic"
    CORRELATIVE = "correlative" 
    TEMPORAL = "temporal"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"

class OpponentEstimatorAPI(ABC):
    """敵コマ推定器の共通インターフェース"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        推定器の初期化
        
        Args:
            config: 推定器固有の設定
        """
        self.config = config
        self.estimator_type = self.get_estimator_type()
        self.is_trained = False
        
    @abstractmethod
    def get_estimator_type(self) -> EstimatorType:
        """推定器のタイプを返す"""
        pass
    
    @abstractmethod
    def estimate(self, game_state: Any, opponent_pieces: Dict) -> OpponentEstimationResult:
        """
        敵コマの善悪推定を実行
        
        Args:
            game_state: ゲーム状態
            opponent_pieces: 敵コマ情報 {(x,y): piece_info}
            
        Returns:
            OpponentEstimationResult: 標準化された推定結果
        """
        pass
    
    @abstractmethod 
    def update(self, game_state: Any, opponent_pieces: Dict, 
               revealed_info: Optional[Dict] = None):
        """
        新しい情報に基づいて内部状態を更新
        
        Args:
            game_state: 現在のゲーム状態
            opponent_pieces: 敵コマ情報
            revealed_info: 判明した敵コマの真の善悪情報
        """
        pass
    
    def get_feature_requirements(self) -> Dict[str, Any]:
        """推定に必要な特徴の要件を返す"""
        return {
            "position_features": True,
            "movement_history": False,
            "correlation_features": False,
            "temporal_features": False
        }
    
    def get_computational_cost(self) -> Dict[str, float]:
        """計算コストの見積もりを返す"""
        return {
            "memory_mb": 50.0,
            "inference_ms": 10.0,
            "training_time_factor": 1.0
        }

class ProbabilisticQuantumEstimator(OpponentEstimatorAPI):
    """確率的量子推定器（共通API実装）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_qubits = config.get('n_qubits', 4)
        self.n_layers = config.get('n_layers', 3)
        self.setup_quantum_circuits()
        
    def get_estimator_type(self) -> EstimatorType:
        return EstimatorType.PROBABILISTIC
    
    def setup_quantum_circuits(self):
        """量子回路のセットアップ"""
        # 実装は前回のコードと同様
        pass
    
    def estimate(self, game_state: Any, opponent_pieces: Dict) -> OpponentEstimationResult:
        """確率的推定の実行"""
        # 特徴抽出
        features = self.extract_position_features(opponent_pieces)
        
        # 各コマを独立して推定
        probabilities = torch.zeros(8)
        confidences = torch.ones(8) * 0.7  # 中程度の信頼度
        
        for i, (pos, piece_info) in enumerate(list(opponent_pieces.items())[:8]):
            # 量子回路で推定（簡略化）
            prob = self.quantum_estimate_single_piece(features[i])
            probabilities[i] = prob
            
        # 戦略パターン推定
        strategy_pattern = {
            "aggressive": float(torch.mean(probabilities[:4])),  # 前方の善玉率
            "defensive": float(torch.mean(probabilities[4:])),   # 後方の善玉率
            "balanced": 1.0 - abs(torch.mean(probabilities[:4]) - torch.mean(probabilities[4:]))
        }
        
        return OpponentEstimationResult(
            piece_probabilities=probabilities,
            confidence_scores=confidences,
            strategy_pattern=strategy_pattern,
            debug_info={"estimator": "probabilistic", "features_shape": features.shape}
        )
    
    def update(self, game_state: Any, opponent_pieces: Dict, revealed_info: Optional[Dict] = None):
        """学習データの更新"""
        if revealed_info:
            # 判明した情報で量子パラメータを更新
            self.update_quantum_parameters(revealed_info)
    
    def quantum_estimate_single_piece(self, features: torch.Tensor) -> float:
        """単一コマの量子推定（簡略化）"""
        return 0.5 + 0.3 * torch.randn(1).item()  # ダミー実装
    
    def extract_position_features(self, opponent_pieces: Dict) -> torch.Tensor:
        """位置特徴の抽出"""
        features = torch.zeros(8, 4)  # 8コマ × 4特徴
        for i, (pos, piece_info) in enumerate(list(opponent_pieces.items())[:8]):
            x, y = pos
            features[i] = torch.tensor([x/6.0, y/6.0, 0.5, 0.5])  # 正規化
        return features

class CorrelativeQuantumEstimator(OpponentEstimatorAPI):
    """相関型量子推定器"""
    
    def get_estimator_type(self) -> EstimatorType:
        return EstimatorType.CORRELATIVE
    
    def get_feature_requirements(self) -> Dict[str, Any]:
        return {
            "position_features": True,
            "movement_history": True,
            "correlation_features": True,
            "temporal_features": False
        }
    
    def estimate(self, game_state: Any, opponent_pieces: Dict) -> OpponentEstimationResult:
        """相関を考慮した推定"""
        # 簡略化された実装
        probabilities = torch.rand(8) * 0.6 + 0.2  # 0.2-0.8の範囲
        confidences = torch.ones(8) * 0.8
        
        # 相関行列の構築（ダミー）
        correlation_matrix = torch.randn(8, 8) * 0.3
        correlation_matrix = (correlation_matrix + correlation_matrix.t()) / 2  # 対称化
        
        # 戦略パターン推定
        strategy_pattern = {
            "clustered": float(torch.mean(torch.abs(correlation_matrix))),
            "aggressive": float(torch.mean(probabilities[:4])),
            "defensive": float(torch.mean(probabilities[4:]))
        }
        
        return OpponentEstimationResult(
            piece_probabilities=probabilities,
            confidence_scores=confidences,
            strategy_pattern=strategy_pattern,
            correlation_matrix=correlation_matrix,
            debug_info={"estimator": "correlative", "correlations_computed": True}
        )
    
    def update(self, game_state: Any, opponent_pieces: Dict, revealed_info: Optional[Dict] = None):
        """相関パラメータの更新"""
        if revealed_info:
            # 判明した情報で相関モデルを更新
            pass

class TemporalQuantumEstimator(OpponentEstimatorAPI):
    """時系列量子推定器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.memory_length = config.get('memory_length', 5)
        self.history_buffer = []
    
    def get_estimator_type(self) -> EstimatorType:
        return EstimatorType.TEMPORAL
    
    def get_feature_requirements(self) -> Dict[str, Any]:
        return {
            "position_features": True,
            "movement_history": True,
            "correlation_features": False,
            "temporal_features": True
        }
    
    def estimate(self, game_state: Any, opponent_pieces: Dict) -> OpponentEstimationResult:
        """時系列を考慮した推定"""
        # 現在の特徴
        current_features = self.extract_temporal_features(game_state, opponent_pieces)
        
        # 履歴との統合
        temporal_context = self.integrate_temporal_context(current_features)
        
        # 量子時系列推定
        probabilities, temporal_features = self.quantum_temporal_estimate(temporal_context)
        
        # 行動パターン分析
        behavior_patterns = self.analyze_behavior_patterns()
        
        # 信頼度（履歴の長さに比例）
        history_factor = min(len(self.history_buffer) / self.memory_length, 1.0)
        confidences = torch.ones(8) * (0.5 + 0.4 * history_factor)
        
        return OpponentEstimationResult(
            piece_probabilities=probabilities,
            confidence_scores=confidences,
            strategy_pattern=behavior_patterns,
            temporal_features=temporal_features,
            debug_info={"estimator": "temporal", "history_length": len(self.history_buffer)}
        )
    
    def update(self, game_state: Any, opponent_pieces: Dict, revealed_info: Optional[Dict] = None):
        """履歴の更新"""
        current_state = torch.randn(8)  # 簡略化
        self.history_buffer.append(current_state)
        
        # バッファサイズ制限
        if len(self.history_buffer) > self.memory_length:
            self.history_buffer.pop(0)
    
    def extract_temporal_features(self, game_state: Any, opponent_pieces: Dict) -> torch.Tensor:
        """時系列特徴の抽出（簡略化）"""
        return torch.randn(8)
    
    def integrate_temporal_context(self, current_features: torch.Tensor) -> torch.Tensor:
        """時系列コンテキストの統合（簡略化）"""
        return current_features
    
    def quantum_temporal_estimate(self, temporal_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """量子時系列推定（簡略化）"""
        probabilities = torch.rand(8) * 0.8 + 0.1
        temporal_features = torch.randn(16)
        return probabilities, temporal_features
    
    def analyze_behavior_patterns(self) -> Dict[str, float]:
        """行動パターン分析（簡略化）"""
        return {
            "escape_oriented": 0.6,
            "aggressive": 0.3,
            "defensive": 0.4,
            "deceptive": 0.2
        }

class EstimatorFactory:
    """推定器のファクトリークラス"""
    
    _estimator_registry = {
        EstimatorType.PROBABILISTIC: ProbabilisticQuantumEstimator,
        EstimatorType.CORRELATIVE: CorrelativeQuantumEstimator, 
        EstimatorType.TEMPORAL: TemporalQuantumEstimator,
    }
    
    @classmethod
    def create_estimator(cls, estimator_type: EstimatorType, 
                        config: Dict[str, Any]) -> OpponentEstimatorAPI:
        """推定器を作成"""
        if estimator_type not in cls._estimator_registry:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        estimator_class = cls._estimator_registry[estimator_type]
        return estimator_class(config)
    
    @classmethod
    def register_estimator(cls, estimator_type: EstimatorType, 
                          estimator_class: type):
        """新しい推定器タイプを登録"""
        cls._estimator_registry[estimator_type] = estimator_class

class ModularEstimatorManager:
    """複数推定器の管理クラス"""
    
    def __init__(self):
        self.estimators: Dict[str, OpponentEstimatorAPI] = {}
        self.active_estimators: List[str] = []
        self.fusion_strategy = "weighted_average"
        
    def add_estimator(self, name: str, estimator: OpponentEstimatorAPI):
        """推定器を追加"""
        self.estimators[name] = estimator
        
    def set_active_estimators(self, estimator_names: List[str]):
        """アクティブな推定器を設定"""
        invalid_names = [name for name in estimator_names if name not in self.estimators]
        if invalid_names:
            raise ValueError(f"Unknown estimators: {invalid_names}")
        self.active_estimators = estimator_names
        
    def estimate_with_fusion(self, game_state: Any, 
                           opponent_pieces: Dict) -> OpponentEstimationResult:
        """複数推定器の結果を統合"""
        if not self.active_estimators:
            raise ValueError("No active estimators set")
        
        # 各推定器で推定
        results = {}
        for name in self.active_estimators:
            results[name] = self.estimators[name].estimate(game_state, opponent_pieces)
        
        # 統合戦略に応じて結果をマージ
        if self.fusion_strategy == "weighted_average":
            return self._weighted_average_fusion(results)
        elif self.fusion_strategy == "confidence_based":
            return self._confidence_based_fusion(results)
        elif self.fusion_strategy == "ensemble_voting":
            return self._ensemble_voting_fusion(results)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _weighted_average_fusion(self, results: Dict[str, OpponentEstimationResult]) -> OpponentEstimationResult:
        """重み付き平均による統合"""
        total_weight = 0
        fused_probabilities = torch.zeros(8)
        fused_confidences = torch.zeros(8)
        
        # 各推定器の重み（計算コストの逆数）
        weights = {}
        for name in results.keys():
            cost = self.estimators[name].get_computational_cost()
            weights[name] = 1.0 / (cost['inference_ms'] + 1.0)
            total_weight += weights[name]
        
        # 正規化重みで統合
        for name, result in results.items():
            weight = weights[name] / total_weight
            fused_probabilities += result.piece_probabilities * weight
            fused_confidences += result.confidence_scores * weight
        
        # 戦略パターンの統合
        fused_strategy = {}
        for pattern in results[list(results.keys())[0]].strategy_pattern.keys():
            fused_strategy[pattern] = sum(
                result.strategy_pattern.get(pattern, 0) * weights[name] / total_weight
                for name, result in results.items()
            )
        
        return OpponentEstimationResult(
            piece_probabilities=fused_probabilities,
            confidence_scores=fused_confidences,
            strategy_pattern=fused_strategy,
            debug_info={
                "fusion_strategy": "weighted_average",
                "estimators_used": list(results.keys()),
                "weights": weights
            }
        )
    
    def _confidence_based_fusion(self, results: Dict[str, OpponentEstimationResult]) -> OpponentEstimationResult:
        """信頼度ベースの統合"""
        # 各コマごとに最も信頼度の高い推定を採用
        fused_probabilities = torch.zeros(8)
        fused_confidences = torch.zeros(8)
        
        for i in range(8):
            best_confidence = 0
            best_prob = 0.5  # デフォルト
            
            for result in results.values():
                if result.confidence_scores[i] > best_confidence:
                    best_confidence = result.confidence_scores[i]
                    best_prob = result.piece_probabilities[i]
            
            fused_probabilities[i] = best_prob
            fused_confidences[i] = best_confidence
        
        return OpponentEstimationResult(
            piece_probabilities=fused_probabilities,
            confidence_scores=fused_confidences,
            strategy_pattern={},
            debug_info={"fusion_strategy": "confidence_based"}
        )

def demonstrate_modular_usage():
    """モジュラー設計の使用例"""
    print("🔧 モジュラー推定器システムのデモ")
    print("=" * 50)
    
    # 推定器の設定
    configs = {
        "probabilistic": {"n_qubits": 4, "n_layers": 3},
        "correlative": {"n_qubits": 8, "n_layers": 4},
        "temporal": {"n_qubits": 8, "memory_length": 5}
    }
    
    # マネージャーの初期化
    manager = ModularEstimatorManager()
    
    # 推定器を作成・追加
    for name, config in configs.items():
        estimator_type = getattr(EstimatorType, name.upper())
        estimator = EstimatorFactory.create_estimator(estimator_type, config)
        manager.add_estimator(name, estimator)
        print(f"✅ {name} estimator added")
    
    # 異なる組み合わせの例
    combinations = [
        (["probabilistic"], "軽量版AI"),
        (["probabilistic", "correlative"], "バランス版AI"), 
        (["correlative", "temporal"], "高精度版AI"),
        (["probabilistic", "correlative", "temporal"], "最強版AI")
    ]
    
    print("\n🎯 様々な組み合わせでAI構築:")
    
    for estimators, description in combinations:
        manager.set_active_estimators(estimators)
        
        # コスト計算
        total_cost = sum(
            manager.estimators[name].get_computational_cost()['inference_ms']
            for name in estimators
        )
        
        print(f"\n{description}:")
        print(f"  推定器: {', '.join(estimators)}")
        print(f"  推論時間: {total_cost:.1f}ms")
        print(f"  特徴要件: {[manager.estimators[name].get_feature_requirements() for name in estimators]}")
    
    print(f"\n🔄 実行時切り替えが可能:")
    print(f"  • 対戦相手の強さに応じて推定器を選択")
    print(f"  • リアルタイムに推定精度と計算コストを調整")
    print(f"  • 新しい推定器の追加が容易")

if __name__ == "__main__":
    demonstrate_modular_usage()
    
    print("\n" + "=" * 50)
    print("🏗️ 共通APIのメリット:")
    print("  • 推定器の差し替えが自由")
    print("  • 複数の組み合わせでA/Bテスト可能")
    print("  • 新しい推定器の開発が独立して可能")
    print("  • 用途に応じた最適構成を選択")
    print("=" * 50)