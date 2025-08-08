# circuits/galton_generator_multi.py
# Unified Quantum Galton Box Generator (biased + unbiased) using controlled-SWAP logic

import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, transpile, execute
import os

# Import from your new quantum QGB logic
from circuits.quantum_peg_multi import generate_qgb, postprocess_qgb


def run_qgb_simulation(layers: int, theta: float = np.pi / 2, shots: int = 2048):
    """
    Run a QGB simulation for given number of layers and θ bias.
    Returns normalized slot probability distribution.
    """
    qc = generate_qgb(layers, theta)
    backend = Aer.get_backend("qasm_simulator")
    compiled = transpile(qc, backend)
    job = execute(compiled, backend=backend, shots=shots)
    counts = job.result().get_counts()
    return postprocess_qgb(counts)


def simulate_distributions(layers: int, shots: int = 2048):
    """
    Simulate both Gaussian (unbiased) and biased QGB distributions.
    """
    # Gaussian (Hadamard = π/2)
    dist_gauss = run_qgb_simulation(layers, theta=np.pi / 2, shots=shots)

    # Biased QGB (e.g., 2π/3 skew left)
    dist_biased = run_qgb_simulation(layers, theta=2 * np.pi / 3, shots=shots)

    return dist_gauss, dist_biased


def plot_distributions(gauss: dict, biased: dict, layers: int):
    """
    Plot the two distributions for comparison.
    """
    os.makedirs("results", exist_ok=True)

    plt.bar(gauss.keys(), gauss.values(), alpha=0.6, label="Gaussian (π/2)")
    plt.bar(biased.keys(), biased.values(), alpha=0.6, label="Biased (2π/3)")
    plt.title(f"Quantum Galton Box Output - {layers} Layers")
    plt.xlabel("Slot Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/gaussian_vs_exponential.png")
    plt.show()


# Optional: distance metrics
def total_variation_distance(p, q):
    keys = sorted(set(p) | set(q))
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def kl_divergence(p, q):
    eps = 1e-12
    keys = sorted(set(p) | set(q))
    p_vals = np.array([p.get(k, eps) for k in keys])
    q_vals = np.array([q.get(k, eps) for k in keys])
    return np.sum(p_vals * np.log(p_vals / q_vals))


if __name__ == "__main__":
    layers = 4
    shots = 2048

    gauss, biased = simulate_distributions(layers, shots=shots)

    # Print distributions
    print("\nGaussian QGB Output:")
    for k in sorted(gauss): print(f"Slot {k}: {gauss[k]:.4f}")

    print("\nBiased QGB Output:")
    for k in sorted(biased): print(f"Slot {k}: {biased[k]:.4f}")

    # Plot and compare
    plot_distributions(gauss, biased, layers)

    # Distance metrics
    tvd = total_variation_distance(gauss, biased)
    kld = kl_divergence(gauss, biased)
    print(f"\nTotal Variation Distance: {tvd:.4f}")
    print(f"Kullback-Leibler Divergence: {kld:.4f}")
