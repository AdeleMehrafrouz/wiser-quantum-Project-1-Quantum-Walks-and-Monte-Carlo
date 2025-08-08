# circuits/quantum_peg_multi.py
# Multi-layer Quantum Galton Board using controlled swaps and optional bias

from qiskit import QuantumCircuit, Aer, transpile, execute
import numpy as np
import matplotlib.pyplot as plt


def generate_qgb(layers: int, theta: float = np.pi / 2) -> QuantumCircuit:
    """
    Generate a multi-layer Quantum Galton Board circuit.
    Each layer uses controlled swaps to split probability across slots.
    """
    control = 0
    output_slots = 2 ** layers
    total_qubits = 1 + output_slots  # 1 control + output slots
    qc = QuantumCircuit(total_qubits, output_slots)

    # Initialize the ball in the leftmost slot
    qc.x(1)

    # For each layer, apply Rx and controlled swaps to split paths
    for layer in range(layers):
        step = 2 ** layer
        for i in range(0, output_slots, step * 2):
            left = 1 + i
            right = 1 + i + step
            qc.rx(theta, control)
            qc.cswap(control, left, right)
        qc.reset(control)  # Reset control for the next layer

    # Measure all final slots
    final_slots = list(range(1, output_slots + 1))
    qc.measure(final_slots, range(output_slots))

    return qc


def run_qgb(layers: int, theta: float = np.pi / 2, shots: int = 2048):
    """
    Run the QGB simulation on the Aer qasm_simulator.
    """
    qc = generate_qgb(layers, theta)
    backend = Aer.get_backend("qasm_simulator")
    compiled = transpile(qc, backend)
    job = execute(compiled, backend=backend, shots=shots)
    return qc, job.result().get_counts()


def postprocess_qgb(counts):
    """
    Convert raw bitstring counts into slot probabilities (slot index = position of '1').
    """
    total = sum(counts.values())
    dist = {}
    for bitstring, count in counts.items():
        idx = bitstring[::-1].find('1')  # find which slot is '1'
        if idx != -1:
            dist[idx] = dist.get(idx, 0) + count
    return {k: v / total for k, v in sorted(dist.items())}


if __name__ == "__main__":
    layers = 4
    theta = 2 * np.pi / 3  # biased

    qc, counts = run_qgb(layers, theta)
    dist = postprocess_qgb(counts)

    print("QGB Output Distribution:")
    for k in sorted(dist):
        print(f"Bin {k}: {dist[k]:.4f}")

    # Plot
    plt.bar(dist.keys(), dist.values(), color="teal", alpha=0.7)
    plt.title(f"Quantum Galton Board - {layers} layers (theta={theta:.2f})")
    plt.xlabel("Slot Index")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
