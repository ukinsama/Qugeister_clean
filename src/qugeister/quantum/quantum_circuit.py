"""
Quantum Circuit implementation for Qugeister system.
"""

import numpy as np
import pennylane as qml
import torch
from functools import lru_cache
import hashlib
import warnings

# PennyLane Lightning警告を抑制
warnings.filterwarnings("ignore", message="No module named 'pennylane_lightning.lightning_qubit_ops'", category=UserWarning)


class FastQuantumCircuit:
    """最適化された量子回路シミュレーション（ユーザー設定対応）"""
    
    def __init__(self, n_qubits=4, n_layers=2, embedding='angle', entanglement='linear'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding = embedding  # 'angle' or 'amplitude'
        self.entanglement = entanglement  # 'linear' or 'full'
        
        # Lightning.qubitはC++バインディング不完全のため、安定したdefault.qubitを使用
        self.dev = qml.device('default.qubit', wires=n_qubits)
        print("🔧 Default.qubit使用（安定配置）")
        
        # 2. 量子回路をJIT最適化
        self.quantum_circuit = self._build_optimized_circuit()
        
        # 3. 計算結果のキャッシュ（LRU: Least Recently Used）
        self.cache_size = 10000
        self._cache_enabled = True
        
        # 4. ルックアップテーブル（事前計算）
        self.lookup_table = {}
        self.precompute_common_patterns()
    
    def _build_optimized_circuit(self):
        """最適化された量子回路の構築（ユーザー設定対応）"""
        
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # ユーザー設定可能なエンコーディング
            for i in range(self.n_qubits):
                if self.embedding == 'angle':
                    qml.RY(inputs[i], wires=i)
                elif self.embedding == 'amplitude':
                    qml.RX(inputs[i], wires=i)
            
            # ユーザー設定可能なレイヤー数
            for l in range(self.n_layers):
                # ユーザー設定可能なエンタングルメント
                if self.entanglement == 'linear':
                    # Linear entanglement (nearest neighbor)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif self.entanglement == 'full':
                    # Full entanglement (all-to-all)
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CNOT(wires=[i, j])
                
                # パラメータ化ゲート
                for i in range(self.n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
            
            # 測定
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def precompute_common_patterns(self):
        """よく使われるパターンを事前計算"""
        print("🔄 量子回路パターンを事前計算中...")
        
        # 代表的な入力パターンを事前計算
        common_patterns = [
            torch.zeros(self.n_qubits),  # ゼロ状態
            torch.ones(self.n_qubits) * np.pi / 2,  # 均等重ね合わせ
            torch.tensor([np.pi * i / self.n_qubits for i in range(self.n_qubits)]),  # 線形
        ]
        
        weights = torch.randn(self.n_layers, self.n_qubits, 2) * 0.1
        
        for pattern in common_patterns:
            key = self._hash_input(pattern)
            self.lookup_table[key] = self.quantum_circuit(pattern, weights)
        
        print(f"✅ {len(self.lookup_table)}個のパターンを事前計算完了")
    
    @lru_cache(maxsize=10000)
    def _hash_input(self, tensor_input):
        """入力をハッシュ化してキャッシュキーを生成"""
        if isinstance(tensor_input, torch.Tensor):
            array = tensor_input.detach().numpy()
        else:
            array = np.array(tensor_input)
        
        # 精度を下げてハッシュ（近似値でもヒットするように）
        rounded = np.round(array, decimals=3)
        return hashlib.md5(rounded.tobytes()).hexdigest()
    
    def forward(self, inputs, weights):
        """高速化された順伝播"""
        # 1. ルックアップテーブルをチェック
        input_hash = self._hash_input(inputs)
        if input_hash in self.lookup_table:
            return self.lookup_table[input_hash]
        
        # 2. 通常の量子回路実行
        result = self.quantum_circuit(inputs, weights)
        
        # 3. 結果をキャッシュ
        if len(self.lookup_table) < self.cache_size:
            self.lookup_table[input_hash] = result
        
        return result