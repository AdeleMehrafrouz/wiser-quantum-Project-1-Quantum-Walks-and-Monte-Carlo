# optimization/visualize_circuits.py
# Compare and visualize original vs optimized QGB circuits:
# - Draw circuit diagrams side-by-side
# - Compare gate counts and depth
# - Verify distribution similarity

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import transpile, Aer
import matplotlib.pyplot as plt
from circuits.quantum_peg_multi import generate_qgb, run_qgb, postprocess_qgb
from optimization.quantum_peg_optimized import generate_qgb_optimized, run_qgb_optimized, postprocess_qgb_optimized

# -----------------------------
# Helper Functions
# -----------------------------

def draw_both_circuits(qc_orig, qc_opt, layers):
    """
    Draw original and optimized circuits side-by-side in a single figure.
    Forces original QGB to display as a single panel (fold=-1).
    """
    os.makedirs("results/circuits", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Draw original circuit (no folding)
    qc_orig.draw(output='mpl', ax=axes[0], fold=-1)
    axes[0].set_title(f"Original QGB ({layers} layers)")

    # Draw optimized circuit
    qc_opt.draw(output='mpl', ax=axes[1])
    axes[1].set_title(f"Optimized QGB ({layers} layers)")

    plt.tight_layout()
    save_path = f"results/circuits/qgb_comparison_{layers}layers.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison image saved at {save_path}")
    plt.show()


def compare_circuits(layers=3):
    """
    Compare original vs optimized QGB circuits for a given layer count.
    - Saves side-by-side diagram
    - Prints depth and gate counts
    - Checks output distribution similarity
    """
    backend = Aer.get_backend('qasm_simulator')

    # Generate circuits
    qc_orig = generate_qgb(layers)
    qc_opt = generate_qgb_optimized(layers)

    # Transpile for fair comparison
    qc_orig_t = transpile(qc_orig, backend, optimization_level=3)
    qc_opt_t = transpile(qc_opt, backend, optimization_level=3)

    # Save side-by-side diagrams
    draw_both_circuits(qc_orig_t, qc_opt_t, layers)

    # Print depth and gate counts
    print(f"--- Circuit Comparison (Layers = {layers}) ---")
    print("Original QGB:")
    print(f"  Depth: {qc_orig_t.depth()}")
    print(f"  Gate counts: {qc_orig_t.count_ops()}")
    print("\nOptimized QGB:")
    print(f"  Depth: {qc_opt_t.depth()}")
    print(f"  Gate counts: {qc_opt_t.count_ops()}")

    # Verify distribution similarity
    _, counts_orig = run_qgb(layers)
    _, counts_opt = run_qgb_optimized(layers)
    dist_orig = postprocess_qgb(counts_orig)
    dist_opt = postprocess_qgb_optimized(counts_opt, layers)

    print("\n--- Distribution Comparison ---")
    print("Original QGB Distribution:", dist_orig)
    print("Optimized QGB Distribution:", dist_opt)


if __name__ == "__main__":
    # Test for layers = 4
    compare_circuits(layers=4)
