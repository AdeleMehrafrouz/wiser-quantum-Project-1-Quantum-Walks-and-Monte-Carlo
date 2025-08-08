# metrics/distance_metrics_multi.py
# Computes distance metrics (TVD, KL divergence) and provides a comparison plot
# for quantum output distributions of 1-layer and multi-layer QGB.

import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

def total_variation_distance(p: dict, q: dict) -> float:
    """
    Compute the Total Variation Distance (TVD) between two distributions p and q.
    Both p and q are dictionaries {key: probability}.
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_vals = np.array([p.get(k, 0.0) for k in keys])
    q_vals = np.array([q.get(k, 0.0) for k in keys])
    return 0.5 * np.sum(np.abs(p_vals - q_vals))


def kl_divergence(p: dict, q: dict) -> float:
    """
    Compute the Kullback-Leibler divergence D_KL(p || q).
    Both p and q are dictionaries {key: probability}.
    A small epsilon (1e-12) is used to avoid log(0).
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_vals = np.array([p.get(k, 1e-12) for k in keys])
    q_vals = np.array([q.get(k, 1e-12) for k in keys])
    return entropy(p_vals, q_vals)


def plot_distribution_comparison(p: dict, q: dict, filename="results/distribution_comparison.png", labels=("P", "Q")):
    """
    Plot and save a bar chart comparing two distributions p and q.
    Parameters:
        p (dict): Distribution P (e.g., QGB output).
        q (dict): Distribution Q (e.g., target distribution).
        filename (str): Path to save the plot.
        labels (tuple): Legend labels for (p, q).
    """
    os.makedirs("results", exist_ok=True)
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_vals = [p.get(k, 0) for k in keys]
    q_vals = [q.get(k, 0) for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, p_vals, width, label=f"Distribution {labels[0]}")
    plt.bar(x + width / 2, q_vals, width, label=f"Distribution {labels[1]}")
    plt.xlabel("Bin Index")
    plt.ylabel("Probability")
    plt.title(f"Distribution Comparison ({labels[0]} vs {labels[1]})")
    plt.xticks(x, keys)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Example usage (for testing):
if __name__ == "__main__":
    # Example distributions
    p = {0: 0.4, 1: 0.6}
    q = {0: 0.5, 1: 0.5}

    print("Total Variation Distance:", total_variation_distance(p, q))
    print("KL Divergence:", kl_divergence(p, q))
    plot_distribution_comparison(p, q, filename="results/test_distribution_comparison.png")
