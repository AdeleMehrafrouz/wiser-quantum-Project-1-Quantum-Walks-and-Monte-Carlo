# circuits/verify_gaussian.py
# Verify that the Quantum Galton Board outputs a Gaussian-like distribution for θ = π/2

import numpy as np
from scipy.stats import binom
from circuits.quantum_peg_multi import run_qgb, postprocess_qgb
from metrics.distance_metrics_multi import total_variation_distance, kl_divergence

def verify_gaussian(max_layers=6, shots=4096):
    """
    Runs QGB for 1..max_layers layers and compares the distribution to the binomial target.
    Prints TVD and KL divergence for each layer.
    """
    for L in range(1, max_layers + 1):
        _, counts = run_qgb(L, theta=np.pi/2, shots=shots)
        dist = postprocess_qgb(counts)

        # The ideal binomial distribution with p=0.5
        keys = sorted(dist.keys())
        target = {k: binom.pmf(k, L, 0.5) for k in keys}

        tvd = total_variation_distance(dist, target)
        kld = kl_divergence(dist, target)
        print(f"L={L}: TVD={tvd:.4f}, KL={kld:.4f}")

if __name__ == "__main__":
    verify_gaussian(max_layers=5, shots=4096)
