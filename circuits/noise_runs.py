# circuits/noise_runs.py
# Runs ideal vs noisy simulations for Quantum Galton Board distributions.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from scipy.stats import binom

from circuits.quantum_peg_multi import run_qgb, postprocess_qgb
from metrics.distance_metrics_multi import total_variation_distance, kl_divergence

# -------------------- Noise Model -------------------- #
def get_noise_model():
    """
    Create a synthetic noise model with 1-, 2-, and 3-qubit depolarizing errors,
    plus readout errors.
    """
    noise_model = NoiseModel()
    p1, p2, p3 = 0.001, 0.01, 0.02  # error rates

    # 1-qubit errors
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rx', 'h', 'x'])

    # 2-qubit errors
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])

    # 3-qubit errors (for CCX gates after CSWAP decomposition)
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p3, 3), ['ccx'])

    # Readout error
    readout_err = 0.02
    readout = ReadoutError([[1 - readout_err, readout_err],
                            [readout_err, 1 - readout_err]])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model

# ------------ Bootstrap CI ----------
def bootstrap_metric_ci(metric_func, p, q, shots=2048, iters=500, alpha=0.05):
    """
    Compute confidence interval for a distance metric (TVD or KL) via bootstrap resampling.
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    pvals = np.array([p.get(k, 0.0) for k in keys])
    qvals = np.array([q.get(k, 0.0) for k in keys])

    values = []
    for _ in range(iters):
        p_counts = np.random.multinomial(shots, pvals)
        q_counts = np.random.multinomial(shots, qvals)
        p_hat = {k: p_counts[i] / shots for i, k in enumerate(keys)}
        q_hat = {k: q_counts[i] / shots for i, k in enumerate(keys)}
        values.append(metric_func(p_hat, q_hat))

    mean_val = np.mean(values)
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean_val, (lo, hi)

# -------------------- Experiment ------------- #
def run_noise_experiment(layers=4, theta=np.pi/2, shots=4096):
    """
    Compare ideal vs noisy QGB for a given layer count and theta.
    Also computes distances vs the target (binomial) distribution.
    """
    # Ideal run
    ideal_qc, ideal_counts = run_qgb(layers, theta, shots=shots)
    ideal_dist = postprocess_qgb(ideal_counts)
    print(f"Ideal circuit depth: {ideal_qc.depth()}")

    # Noisy run (decomposed circuit for CCX noise)
    noise_model = get_noise_model()
    sim_noisy = AerSimulator(noise_model=noise_model)
    noisy_job = sim_noisy.run(ideal_qc.decompose(), shots=shots)
    noisy_counts = noisy_job.result().get_counts()
    noisy_dist = postprocess_qgb(noisy_counts)

    # Distance metrics with bootstrap CI (ideal vs noisy)
    tvd, tvd_ci = bootstrap_metric_ci(total_variation_distance, ideal_dist, noisy_dist, shots)
    kld, kld_ci = bootstrap_metric_ci(kl_divergence, ideal_dist, noisy_dist, shots)

    # Target distribution (Gaussian if theta = pi/2)
    if np.isclose(theta, np.pi / 2):
        target_dist = {k: binom.pmf(k, layers, 0.5) for k in range(layers + 1)}
    else:
        target_dist = ideal_dist  # For biased QGB, target = ideal distribution

    tvd_target, tvd_target_ci = bootstrap_metric_ci(total_variation_distance, noisy_dist, target_dist, shots)
    kld_target, kld_target_ci = bootstrap_metric_ci(kl_divergence, noisy_dist, target_dist, shots)

    return ideal_dist, noisy_dist, target_dist, tvd, tvd_ci, kld, kld_ci, tvd_target, tvd_target_ci, kld_target, kld_target_ci

# -------------------- Main ----------------
if __name__ == "__main__":
    layers = 4
    theta = np.pi / 2
    shots = 4096

    (ideal_dist, noisy_dist, target_dist,
     tvd, tvd_ci, kld, kld_ci,
     tvd_target, tvd_target_ci,
     kld_target, kld_target_ci) = run_noise_experiment(layers, theta, shots)

    print(f"\n--- QGB Noise Analysis (layers={layers}) ---")
    print(f"TVD (ideal vs noisy): {tvd:.4f} ± [{tvd_ci[0]:.4f}, {tvd_ci[1]:.4f}]")
    print(f"KL Divergence (ideal vs noisy): {kld:.4f} ± [{kld_ci[0]:.4f}, {kld_ci[1]:.4f}]")
    print(f"TVD (noisy vs target): {tvd_target:.4f} ± [{tvd_target_ci[0]:.4f}, {tvd_target_ci[1]:.4f}]")
    print(f"KL Divergence (noisy vs target): {kld_target:.4f} ± [{kld_target_ci[0]:.4f}, {kld_target_ci[1]:.4f}]")

    # Save results to a text file
    os.makedirs("results", exist_ok=True)
    with open("results/noise_analysis.txt", "w") as f:
        f.write(f"--- QGB Noise Analysis (layers={layers}) ---\n")
        f.write(f"TVD (ideal vs noisy): {tvd:.4f} ± [{tvd_ci[0]:.4f}, {tvd_ci[1]:.4f}]\n")
        f.write(f"KL Divergence (ideal vs noisy): {kld:.4f} ± [{kld_ci[0]:.4f}, {kld_ci[1]:.4f}]\n")
        f.write(f"TVD (noisy vs target): {tvd_target:.4f} ± [{tvd_target_ci[0]:.4f}, {tvd_target_ci[1]:.4f}]\n")
        f.write(f"KL Divergence (noisy vs target): {kld_target:.4f} ± [{kld_target_ci[0]:.4f}, {kld_target_ci[1]:.4f}]\n")

    # Plot
    plt.bar(ideal_dist.keys(), ideal_dist.values(), alpha=0.6, label="Ideal")
    plt.bar(noisy_dist.keys(), noisy_dist.values(), alpha=0.6, label="Noisy")
    plt.bar(target_dist.keys(), target_dist.values(), alpha=0.4, label="Target", color="green")
    plt.title(f"QGB Ideal vs Noisy vs Target (layers={layers}, theta={theta:.2f})")
    plt.xlabel("Slot Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/qgb_noisy_vs_ideal_target.png")
    plt.show()
