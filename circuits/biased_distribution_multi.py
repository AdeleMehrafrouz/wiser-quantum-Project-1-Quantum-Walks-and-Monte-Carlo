# circuits/biased_distribution_multi.py
# Compare Gaussian (unbiased) and biased QGB distributions

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, transpile, execute
from typing import Dict
from circuits.quantum_peg_multi import generate_qgb, postprocess_qgb


def run_qgb_distribution(layers: int, theta: float, shots: int = 2048) -> Dict[int, float]:
    """
    Run a QGB simulation with a given theta and return a normalized distribution.
    """
    qc = generate_qgb(layers, theta)
    backend = Aer.get_backend("qasm_simulator")
    compiled = transpile(qc, backend)
    job = execute(compiled, backend=backend, shots=shots)
    counts = job.result().get_counts()
    return postprocess_qgb(counts)


if __name__ == "__main__":
    layers = 4
    shots = 2048

    # Gaussian (unbiased)
    dist_gauss = run_qgb_distribution(layers, theta=np.pi / 2, shots=shots)

    # Biased (exponential-like)
    dist_biased = run_qgb_distribution(layers, theta=2 * np.pi / 3, shots=shots)

    # Print results
    print("Gaussian QGB Distribution:")
    for k in sorted(dist_gauss):
        print(f"Bin {k}: {dist_gauss[k]:.4f}")

    print("\nBiased QGB Distribution:")
    for k in sorted(dist_biased):
        print(f"Bin {k}: {dist_biased[k]:.4f}")

    # Plot both distributions
    os.makedirs("results", exist_ok=True)
    plt.bar(dist_gauss.keys(), dist_gauss.values(), alpha=0.6, label="Gaussian (θ=π/2)")
    plt.bar(dist_biased.keys(), dist_biased.values(), alpha=0.6, label="Biased (θ=2π/3)")
    plt.title(f"Quantum Galton Board - {layers} Layers (Comparison)")
    plt.xlabel("Slot Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/qgb_gaussian_vs_biased.png")
    plt.show()
