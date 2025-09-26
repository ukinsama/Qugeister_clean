#!/usr/bin/env python3
"""
Quick Test - Simple Quantum AI Training Demo
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("    Quick Quantum AI Test - 100 Episodes")
print("=" * 60)

class SimpleQuantumLayer(nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits

        try:
            dev = qml.device('lightning.qubit', wires=n_qubits)
        except:
            dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(inputs, weights):
            for i in range(min(len(inputs), n_qubits)):
                qml.RY(inputs[i] * np.pi / 2, wires=i)

            for i in range(n_qubits):
                qml.RY(weights[i, 0], wires=i)
                qml.RZ(weights[i, 1], wires=i)

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_qubits, 2) * 0.1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        outputs = []
        for i in range(x.shape[0]):
            result = self.circuit(x[i].float()[:self.n_qubits], self.weights)
            outputs.append(torch.tensor(result, dtype=torch.float32))

        return torch.stack(outputs)

class QuickQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Linear(36, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        self.quantum = SimpleQuantumLayer(4)
        self.backend = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 36)
        )

    def forward(self, x):
        x = self.frontend(x.float())
        x = self.quantum(x)
        x = self.backend(x)
        return x

def simple_train():
    print("Initializing model...")
    model = QuickQCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()  # Huber Loss

    print("Starting training...")
    for episode in range(100):
        # Dummy training data
        state = torch.randn(36)
        target = torch.randn(36)

        # Forward pass
        output = model(state)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 20 == 0:
            print(f"Episode {episode:3d} | Loss: {loss.item():.4f}")

    print("\nTraining completed!")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"experiments/models/quick_test_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    return model_path

if __name__ == "__main__":
    model_path = simple_train()
    print(f"\nTest completed successfully!")
    print(f"Quantum AI model trained with Huber Loss")
    print(f"Saved to: {model_path}")