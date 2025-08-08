# circuits/hadamard_walk_numpy.py
# Classical simulation of a 1D Hadamard quantum walk.

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict

def hadamard_operator():
    return (1/np.sqrt(2)) * np.array([[1, 1],
                                      [1, -1]], dtype=complex)

def shift_operator(num_positions: int):
    """
    Shift operator for quantum walk (right if coin=1, left if coin=0).
    """
    S = np.zeros((2 * num_positions, 2 * num_positions), dtype=complex)
    # Fill block structure for shift
    for x in range(num_positions):
        # coin = 0 -> left
        if x > 0:
            S[x * 2, x * 2] = 1
            S[x * 2, (x - 1) * 2] = 1
        else:
            S[x * 2, x * 2] = 1  # Reflect at boundary
        # coin = 1 -> right
        if x < num_positions - 1:
            S[x * 2 + 1, x * 2 + 1] = 1
            S[x * 2 + 1, (x + 1) * 2 + 1] = 1
        else:
            S[x * 2 + 1, x * 2 + 1] = 1  # Reflect at boundary
    return S

def run_quantum_walk(steps: int) -> Dict[int, float]:
    num_positions = 2 * steps + 1  # positions from -steps to +steps
    # Initial state: at center, coin=0
    state = np.zeros((2 * num_positions,), dtype=complex)
    center = steps
    state[2 * center] = 1.0  # coin=0, position=center

    H = hadamard_operator()
    S = shift_operator(num_positions)

    # Apply steps
    for _ in range(steps):
        # Apply coin operator to each position
        new_state = np.zeros_like(state)
        for pos in range(num_positions):
            coin_state = state[2 * pos: 2 * pos + 2]
            new_coin_state = H @ coin_state
            new_state[2 * pos: 2 * pos + 2] = new_coin_state
        state = S @ new_state

    # Measure probability distribution over positions
    probs = {}
    for pos in range(num_positions):
        p = abs(state[2 * pos])**2 + abs(state[2 * pos + 1])**2
        if p > 1e-12:
            probs[pos - steps] = p  # shift index to -steps...+steps
    return probs

if __name__ == "__main__":
    steps = 3
    distribution = run_quantum_walk(steps)
    print("Hadamard Walk Output Distribution:")
    for pos in sorted(distribution):
        print(f"Position {pos}: {distribution[pos]:.4f}")

    os.makedirs("results", exist_ok=True)
    plt.bar(distribution.keys(), distribution.values(), color='purple', alpha=0.7)
    plt.title(f"Hadamard Quantum Walk - {steps} Steps")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/hadamard_walk.png")
    plt.show()
