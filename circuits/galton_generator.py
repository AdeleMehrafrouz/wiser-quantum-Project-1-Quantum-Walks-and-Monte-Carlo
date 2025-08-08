# circuits/galton_generator.py
# Description: Quantum Galton Box Generator

from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

##################################
# Deliverable 2: General Galton Box Generator
#########################

def generate_galton_box_circuit(layers: int, biased=False, theta=np.pi/2) -> QuantumCircuit:
    """
    Generate a quantum Galton box circuit with a given number of layers.
    Can simulate fair (Hadamard) or biased (Rx) pegs.
    """
    qubits = layers + 1
    qc = QuantumCircuit(qubits, qubits)

    for i in range(layers):
        if biased:
            qc.rx(theta, i)
        else:
            qc.h(i)
        qc.cx(i, i + 1)

    qc.measure(range(qubits), range(qubits))
    return qc

def run_simulation(circuit: QuantumCircuit, shots=2048):
    simulator = Aer.get_backend('qasm_simulator')
    compiled = transpile(circuit, simulator)
    job = execute(compiled, backend=simulator, shots=shots)
    result = job.result()
    return result.get_counts()

def postprocess_counts(counts):
    """Convert one-hot outputs to slot indices."""
    new_counts = {}
    for bitstring, count in counts.items():
        position = bitstring[::-1].find('1')
        if position not in new_counts:
            new_counts[position] = 0
        new_counts[position] += count
    return new_counts

def normalize_distribution(dist):
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}

###################################
# Deliverable 3: Biased & Hadamard Variants
###########################

def simulate_distribution(layers, biased=False, theta=np.pi/2, shots=2048):
    qc = generate_galton_box_circuit(layers, biased=biased, theta=theta)
    raw_counts = run_simulation(qc, shots=shots)
    processed = postprocess_counts(raw_counts)
    return normalize_distribution(processed)

###################################
# Deliverable 4–5: Distance Metrics
############################

def total_variation_distance(p, q):
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_vals = np.array([p.get(k, 0.0) for k in keys])
    q_vals = np.array([q.get(k, 0.0) for k in keys])
    return 0.5 * np.sum(np.abs(p_vals - q_vals))

def kl_divergence(p, q):
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_vals = np.array([p.get(k, 1e-12) for k in keys])
    q_vals = np.array([q.get(k, 1e-12) for k in keys])
    return entropy(p_vals, q_vals)

# Example Execution

if __name__ == "__main__":
    layers = 4

    # Gaussian (Hadamard)
    dist_gauss = simulate_distribution(layers)

    # Exponential (Biased)
    dist_exp = simulate_distribution(layers, biased=True, theta=2 * np.pi / 3)

    # Ensure 'results' folder exists
    os.makedirs("results", exist_ok=True)

    # Plot both distributions
    plt.bar(dist_gauss.keys(), dist_gauss.values(), alpha=0.6, label="Gaussian")
    plt.bar(dist_exp.keys(), dist_exp.values(), alpha=0.6, label="Exponential")
    plt.title("Simulated Distributions")
    plt.xlabel("Output bin")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/gaussian_vs_exponential.png")
    plt.show()

    # Distance metrics
    print("TVD:", total_variation_distance(dist_gauss, dist_exp))
    print("KL Divergence:", kl_divergence(dist_gauss, dist_exp))
