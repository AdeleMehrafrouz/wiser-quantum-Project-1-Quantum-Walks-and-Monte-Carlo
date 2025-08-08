# circuits/noise_real_backend.py
# Run Quantum Galton Board & Hadamard Walk under a real hardware noise model
# Compare noisy vs target distributions with TVD, KL, and bootstrap confidence intervals.

import os
import numpy as np
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeToronto  # Example real backend

from circuits.quantum_peg_multi import run_qgb, postprocess_qgb
from circuits.hadamard_walk_qiskit import run_walk_simulation
from metrics.distance_metrics_multi import total_variation_distance, kl_divergence
from metrics.bootstrap_ci import bootstrap_tvd_ci
from scipy.stats import binom

# --------------------------
# Helper Functions
#------------

def get_real_noise_model():
    """
    Load a realistic noise model from a Qiskit FakeBackend (e.g., FakeToronto).
    Returns a simulator configured with that noise model.
    """
    backend = FakeToronto()
    sim_noisy = AerSimulator.from_backend(backend)
    return sim_noisy, backend.configuration().coupling_map, backend.name()


def evaluate_distribution(dist, target_dist, shots=2048, bootstrap=True):
    """
    Compute TVD and KL divergence vs a target distribution.
    Optionally compute a bootstrap confidence interval for TVD.
    """
    tvd = total_variation_distance(dist, target_dist)
    kld = kl_divergence(dist, target_dist)

    if bootstrap:
        tvd_mean, tvd_ci = bootstrap_tvd_ci(dist, target_dist, shots=shots, iters=300)
        return tvd_mean, tvd_ci, kld
    else:
        return tvd, (tvd, tvd), kld


def build_gaussian_target(layers):
    """Return a binomial target distribution for unbiased QGB."""
    keys = range(layers + 1)
    return {k: binom.pmf(k, layers, 0.5) for k in keys}


# --------------------------
# Main Experiment
# --------------------------

def run_noisy_qgb(layers=4, shots=2048, theta=np.pi/2):
    """
    Run QGB under a real backend noise model.
    Compare with the target Gaussian distribution (if theta=π/2).
    """
    print(f"\n--- Running Noisy QGB (layers={layers}, theta={theta:.2f}) ---")
    # Create QGB circuit (ideal)
    qc, ideal_counts = run_qgb(layers, theta, shots=shots)
    ideal_dist = postprocess_qgb(ideal_counts)

    # Load noise model
    sim_noisy, coupling_map, backend_name = get_real_noise_model()

    # Transpile for this backend
    compiled = transpile(qc, sim_noisy, optimization_level=3, coupling_map=coupling_map)
    print(f"Backend: {backend_name}")
    print(f"Circuit depth (optimized): {compiled.depth()}")

    # Run noisy simulation
    noisy_counts = sim_noisy.run(compiled, shots=shots).result().get_counts()
    noisy_dist = postprocess_qgb(noisy_counts)

    # Determine target distribution
    if np.isclose(theta, np.pi/2):
        target_dist = build_gaussian_target(layers)
    else:
        target_dist = ideal_dist  # For biased cases, compare to ideal circuit output

    # Metrics
    tvd_mean, tvd_ci, kld = evaluate_distribution(noisy_dist, target_dist, shots)
    print(f"TVD vs target: {tvd_mean:.4f} ± [{tvd_ci[0]:.4f}, {tvd_ci[1]:.4f}]")
    print(f"KL Divergence vs target: {kld:.4f}")

    # Plot
    os.makedirs("results", exist_ok=True)
    plt.bar(noisy_dist.keys(), noisy_dist.values(), alpha=0.6, label="Noisy")
    plt.bar(target_dist.keys(), target_dist.values(), alpha=0.4, label="Target")
    plt.title(f"Noisy QGB vs Target (L={layers}, θ={theta:.2f})")
    plt.xlabel("Slot Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/noisy_qgb_L{layers}.png")
    plt.show()

    return noisy_dist, target_dist


def run_noisy_hadamard_walk(steps=3, shots=2048):
    """
    Run the Hadamard quantum walk circuit under real backend noise.
    Compare vs ideal noiseless walk.
    """
    print(f"\n--- Running Noisy Hadamard Walk (steps={steps}) ---")
    from circuits.hadamard_walk_qiskit import hadamard_walk_circuit

    qc = hadamard_walk_circuit(steps)
    ideal_dist = run_walk_simulation(steps, shots=shots)

    sim_noisy, coupling_map, backend_name = get_real_noise_model()
    compiled = transpile(qc, sim_noisy, optimization_level=3, coupling_map=coupling_map)
    print(f"Backend: {backend_name}")
    print(f"Circuit depth (optimized): {compiled.depth()}")

    noisy_counts = sim_noisy.run(compiled, shots=shots).result().get_counts()
    total = sum(noisy_counts.values())
    noisy_dist = {int(k, 2) - steps: v / total for k, v in noisy_counts.items()}

    tvd_mean, tvd_ci, kld = evaluate_distribution(noisy_dist, ideal_dist, shots)
    print(f"TVD vs ideal: {tvd_mean:.4f} ± [{tvd_ci[0]:.4f}, {tvd_ci[1]:.4f}]")
    print(f"KL Divergence vs ideal: {kld:.4f}")

    os.makedirs("results", exist_ok=True)
    plt.bar(noisy_dist.keys(), noisy_dist.values(), alpha=0.6, label="Noisy")
    plt.bar(ideal_dist.keys(), ideal_dist.values(), alpha=0.4, label="Ideal")
    plt.title(f"Noisy vs Ideal Hadamard Walk (steps={steps})")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/noisy_hadamard_walk_{steps}.png")
    plt.show()

    return noisy_dist, ideal_dist


if __name__ == "__main__":
    shots = 2048

    # Run Gaussian QGB under noise
    run_noisy_qgb(layers=3, shots=shots, theta=np.pi/2)

    # Run biased QGB under noise (Exponential-like)
    run_noisy_qgb(layers=3, shots=shots, theta=2 * np.pi / 3)

    # Run Hadamard Walk under noise
    run_noisy_hadamard_walk(steps=3, shots=shots)
