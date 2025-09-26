#!/usr/bin/env python3
"""
強化された収束検出システム
Loss改善を含む包括的な停止条件を実装
"""

import numpy as np
from collections import deque
import torch

class EnhancedConvergenceDetector:
    """Loss改善を含む包括的収束検出システム"""

    def __init__(self,
                 balance_threshold=0.95,
                 patience=50,
                 min_games=1000,
                 loss_patience=20,
                 loss_improvement_threshold=0.001,
                 min_loss_samples=100):

        # Balance収束条件
        self.balance_threshold = balance_threshold
        self.balance_patience = patience
        self.min_games = min_games
        self.consecutive_balance_good = 0
        self.best_balance = 0.0

        # Loss改善条件
        self.loss_patience = loss_patience
        self.loss_improvement_threshold = loss_improvement_threshold
        self.min_loss_samples = min_loss_samples
        self.loss_history_p1 = deque(maxlen=200)
        self.loss_history_p2 = deque(maxlen=200)
        self.best_loss_p1 = float('inf')
        self.best_loss_p2 = float('inf')
        self.loss_plateau_count = 0

        # 総合判定
        self.convergence_history = []
        self.early_stopping_triggered = False
        self.stopping_reason = None

        print(f"Enhanced収束検出器:")
        print(f"  Balance: 閾値={balance_threshold}, patience={patience}")
        print(f"  Loss: patience={loss_patience}, improvement_threshold={loss_improvement_threshold}")
        print(f"  最小ゲーム数: {min_games}")

    def update_losses(self, losses_p1, losses_p2):
        """Loss履歴を更新"""
        if losses_p1:
            recent_loss_p1 = np.mean(losses_p1[-10:]) if len(losses_p1) >= 10 else losses_p1[-1]
            self.loss_history_p1.append(recent_loss_p1)
            self.best_loss_p1 = min(self.best_loss_p1, recent_loss_p1)

        if losses_p2:
            recent_loss_p2 = np.mean(losses_p2[-10:]) if len(losses_p2) >= 10 else losses_p2[-1]
            self.loss_history_p2.append(recent_loss_p2)
            self.best_loss_p2 = min(self.best_loss_p2, recent_loss_p2)

    def check_loss_improvement(self):
        """Loss改善状況を判定"""
        if len(self.loss_history_p1) < self.min_loss_samples:
            return True, {'status': 'insufficient_loss_data'}

        # 直近のLoss平均
        recent_window = 50
        current_loss_p1 = np.mean(list(self.loss_history_p1)[-recent_window:])
        current_loss_p2 = np.mean(list(self.loss_history_p2)[-recent_window:])

        # ベストからの差
        loss_diff_p1 = current_loss_p1 - self.best_loss_p1
        loss_diff_p2 = current_loss_p2 - self.best_loss_p2

        # 改善判定
        p1_improving = loss_diff_p1 <= self.loss_improvement_threshold
        p2_improving = loss_diff_p2 <= self.loss_improvement_threshold

        # Loss停滞カウント
        if not p1_improving or not p2_improving:
            self.loss_plateau_count += 1
        else:
            self.loss_plateau_count = 0

        # Loss発散チェック
        loss_diverging = (loss_diff_p1 > 0.1 or loss_diff_p2 > 0.1)

        metrics = {
            'current_loss_p1': current_loss_p1,
            'current_loss_p2': current_loss_p2,
            'best_loss_p1': self.best_loss_p1,
            'best_loss_p2': self.best_loss_p2,
            'loss_diff_p1': loss_diff_p1,
            'loss_diff_p2': loss_diff_p2,
            'p1_improving': p1_improving,
            'p2_improving': p2_improving,
            'plateau_count': self.loss_plateau_count,
            'loss_diverging': loss_diverging
        }

        # Loss停止条件
        loss_early_stop = self.loss_plateau_count >= self.loss_patience

        if loss_early_stop:
            metrics['status'] = 'loss_plateau_reached'
        elif loss_diverging:
            metrics['status'] = 'loss_diverging'
        elif p1_improving and p2_improving:
            metrics['status'] = 'loss_improving'
        else:
            metrics['status'] = 'loss_mixed'

        return not loss_early_stop and not loss_diverging, metrics

    def check_balance_convergence(self, game_results):
        """Balance収束を判定"""
        if len(game_results) < self.min_games:
            return False, 0.0, {'reason': 'insufficient_games', 'games': len(game_results)}

        # 最近の結果を分析
        recent_games = game_results[-500:]

        wins_1 = sum(1 for r in recent_games if r.get('winner') == 1)
        wins_2 = sum(1 for r in recent_games if r.get('winner') == 2)
        draws = sum(1 for r in recent_games if r.get('winner') is None)

        total_games = len(recent_games)
        decisive_games = wins_1 + wins_2

        # バランス計算
        if decisive_games > 0:
            balance = min(wins_1, wins_2) / max(wins_1, wins_2)
            win_rate_1 = wins_1 / total_games
            win_rate_2 = wins_2 / total_games
            draw_rate = draws / total_games
        else:
            balance = 1.0
            win_rate_1 = win_rate_2 = 0.0
            draw_rate = 1.0

        # 収束判定
        is_balanced = balance >= self.balance_threshold
        has_active_games = decisive_games >= 50

        metrics = {
            'balance': balance,
            'win_rate_1': win_rate_1,
            'win_rate_2': win_rate_2,
            'draw_rate': draw_rate,
            'decisive_games': decisive_games,
            'total_games': total_games,
            'is_balanced': is_balanced,
            'has_active_games': has_active_games
        }

        # 連続カウント
        if is_balanced and has_active_games:
            self.consecutive_balance_good += 1
            metrics['status'] = 'balance_converging'
        else:
            self.consecutive_balance_good = 0
            if not is_balanced:
                metrics['status'] = 'balance_unbalanced'
            elif not has_active_games:
                metrics['status'] = 'too_many_draws'

        metrics['consecutive_good'] = self.consecutive_balance_good
        self.best_balance = max(self.best_balance, balance)
        metrics['best_balance'] = self.best_balance

        # Balance収束判定
        balance_converged = self.consecutive_balance_good >= self.balance_patience

        return balance_converged, balance, metrics

    def comprehensive_convergence_check(self, game_results, losses_p1, losses_p2, episode):
        """包括的収束判定（Balance + Loss）"""

        # Loss履歴更新
        self.update_losses(losses_p1, losses_p2)

        # Balance収束チェック
        balance_converged, balance, balance_metrics = self.check_balance_convergence(game_results)

        # Loss改善チェック
        loss_ok, loss_metrics = self.check_loss_improvement()

        # 総合判定
        comprehensive_metrics = {
            'episode': episode,
            'balance_converged': balance_converged,
            'loss_improving': loss_ok,
            'balance_metrics': balance_metrics,
            'loss_metrics': loss_metrics
        }

        # 停止条件判定
        if balance_converged and loss_ok:
            # 理想的な収束: Balance達成 + Loss改善
            converged = True
            self.stopping_reason = 'comprehensive_convergence'
            comprehensive_metrics['convergence_type'] = 'optimal'

        elif balance_converged and not loss_ok:
            # Balance達成だがLoss停滞
            if loss_metrics.get('loss_diverging', False):
                # Loss発散: 早期停止
                converged = True
                self.stopping_reason = 'loss_divergence_with_balance'
                comprehensive_metrics['convergence_type'] = 'early_stop_divergence'
            else:
                # Loss停滞: 継続判断
                converged = loss_metrics.get('plateau_count', 0) >= self.loss_patience
                self.stopping_reason = 'loss_plateau_with_balance' if converged else None
                comprehensive_metrics['convergence_type'] = 'plateau_stop' if converged else 'monitoring'

        elif not balance_converged and not loss_ok:
            # Balance未達成 + Loss問題
            if loss_metrics.get('loss_diverging', False):
                # Loss発散: 早期停止
                converged = True
                self.stopping_reason = 'loss_divergence_no_balance'
                comprehensive_metrics['convergence_type'] = 'early_stop_divergence'
            else:
                # 両方とも改善余地あり: 継続
                converged = False
                comprehensive_metrics['convergence_type'] = 'training_ongoing'

        else:
            # Balance未達成だがLoss改善: 継続
            converged = False
            comprehensive_metrics['convergence_type'] = 'loss_improving_continue'

        # 履歴記録
        self.convergence_history.append(comprehensive_metrics.copy())

        return converged, comprehensive_metrics

    def get_convergence_report(self):
        """収束レポート生成"""
        if not self.convergence_history:
            return "収束データなし"

        latest = self.convergence_history[-1]

        report = f"""
=== Enhanced Convergence Report ===
停止理由: {self.stopping_reason or '継続中'}
収束タイプ: {latest['convergence_type']}

Balance Status:
  - 現在Balance: {latest['balance_metrics']['balance']:.4f}
  - 目標達成: {latest['balance_converged']}
  - 連続達成: {latest['balance_metrics']['consecutive_good']}/{self.balance_patience}
  - Draw率: {latest['balance_metrics']['draw_rate']:.3f}

Loss Status:
  - P1 Loss: {latest['loss_metrics']['current_loss_p1']:.4f} (Best: {latest['loss_metrics']['best_loss_p1']:.4f})
  - P2 Loss: {latest['loss_metrics']['current_loss_p2']:.4f} (Best: {latest['loss_metrics']['best_loss_p2']:.4f})
  - 改善状況: P1={latest['loss_metrics']['p1_improving']}, P2={latest['loss_metrics']['p2_improving']}
  - 停滞カウント: {latest['loss_metrics']['plateau_count']}/{self.loss_patience}

推奨アクション:
"""

        if latest['convergence_type'] == 'optimal':
            report += "✅ 実験成功! モデル保存を推奨"
        elif latest['convergence_type'] == 'early_stop_divergence':
            report += "⚠️ Loss発散により早期停止。ハイパーパラメータ調整を推奨"
        elif latest['convergence_type'] == 'plateau_stop':
            report += "⏹️ Loss停滞により停止。現在のモデルを保存"
        elif latest['convergence_type'] == 'training_ongoing':
            report += "⏳ 学習継続中。進捗を監視"
        else:
            report += "📊 状況を継続監視中"

        return report

# 使用例とテスト
if __name__ == "__main__":
    # テスト用のダミーデータ
    detector = EnhancedConvergenceDetector(
        balance_threshold=0.95,
        patience=10,
        loss_patience=5
    )

    # テスト実行
    print("Enhanced Convergence Detector テスト完了")
    print("このモジュールをインポートして使用してください:")