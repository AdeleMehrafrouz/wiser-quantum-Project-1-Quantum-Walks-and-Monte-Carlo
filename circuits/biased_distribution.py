# circuits/biased_distribution.py
# Simulates a biased Galton board using Rx(θ) gates for controlled distribution shaping

from qiskit import QuantumCircuit, transpile, execute
from qiskit.providers.aer import Aer
import numpy as np
from typing import Dict
import os
import matplotlib.pyplot as plt


def generate_biased_galton_box(layers: int, theta: float = 2 * np.pi / 3) -> QuantumCircuit:
    """
    Create a biased quantum Galton board circuit using Rx(θ) gates.
    """
    qubits = layers + 1
    qc = QuantumCircuit(qubits, qubits)

    for i in range(layers):
        qc.rx(theta, i)
        qc.cx(i, i + 1)

    qc.measure(range(qubits), range(qubits))
    return qc


def run_simulation(circuit: QuantumCircuit, shots: int = 2048) -> Dict[str, int]:
    simulator = Aer.get_backend('qasm_simulator')
    compiled = transpile(circuit, simulator)
    job = execute(compiled, backend=simulator, shots=shots)
    result = job.result()
    return result.get_counts()


def postprocess_counts(counts: Dict[str, int]) -> Dict[int, float]:
    processed = {}
    total = sum(counts.values())

    for bitstring, count in counts.items():
        pos = bitstring[::-1].find('1')
        if pos not in processed:
            processed[pos] = 0
        processed[pos] += count

    return {k: v / total for k, v in processed.items()}


# Example usage
if __name__ == "__main__":
    layers = 4
    theta = 2 * np.pi / 3  # Biased toward left

    qc = generate_biased_galton_box(layers, theta)
    raw_counts = run_simulation(qc)
    distribution = postprocess_counts(raw_counts)

    # Print the normalized distribution
    for bin_index in sorted(distribution):
        print(f"Bin {bin_index}: {distribution[bin_index]:.4f}")

    # Save plot to results folder
    os.makedirs("results", exist_ok=True)
    plt.bar(distribution.keys(), distribution.values(), color='red', alpha=0.7)
    plt.title("Exponential Distribution (Biased Galton Board)")
    plt.xlabel("Slot Index")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/exponential_histogram.png")
    plt.show()
