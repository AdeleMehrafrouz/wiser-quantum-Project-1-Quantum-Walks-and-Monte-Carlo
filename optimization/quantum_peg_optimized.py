# optimization/quantum_peg_optimized.py
# Optimized Quantum Galton Board using binary position encoding for reduced depth.

from qiskit import QuantumCircuit, Aer, transpile, execute
import numpy as np
import matplotlib.pyplot as plt


def generate_qgb_optimized(layers: int, theta: float = np.pi / 2) -> QuantumCircuit:
    """
    Generate an optimized multi-layer Quantum Galton Board circuit.
    - Uses a binary position register instead of CSWAP gates.
    - A single coin qubit controls the direction at each layer.
    """
    # Number of position qubits needed: ceil(log2(layers + 1))
    pos_qubits = int(np.ceil(np.log2(layers + 1))) if layers > 1 else 1
    total_qubits = 1 + pos_qubits  # 1 coin + position register
    qc = QuantumCircuit(total_qubits, pos_qubits)

    coin = 0
    pos = list(range(1, total_qubits))

    # Initialize walker at position 0 (all position qubits = 0)
    # No initialization needed; by default all qubits start in |0>

    # Walk through each layer
    for _ in range(layers):
        qc.rx(theta, coin)  # Bias or Hadamard-like split

        # If coin=1, increment position; if coin=0, leave it as is.
        # Achieved by controlled increment on the position register.
        increment_position(qc, coin, pos)

    # Measure final position
    qc.measure(pos, range(pos_qubits))
    return qc


def increment_position(qc, ctrl, pos_qubits):
    """
    Controlled increment of binary position register by +1 (mod 2^n).
    Equivalent to adding 1 when coin qubit is |1>.
    """
    qc.cx(ctrl, pos_qubits[0])
    for i in range(1, len(pos_qubits)):
        controls = [ctrl] + pos_qubits[:i]
        qc.mcx(controls, pos_qubits[i])


def run_qgb_optimized(layers: int, theta: float = np.pi / 2, shots: int = 2048):
    """
    Run the optimized QGB simulation on the Aer qasm_simulator.
    """
    qc = generate_qgb_optimized(layers, theta)
    backend = Aer.get_backend("qasm_simulator")
    compiled = transpile(qc, backend, optimization_level=3)
    job = execute(compiled, backend=backend, shots=shots)
    return qc, job.result().get_counts()


def postprocess_qgb_optimized(counts, layers):
    """
    Convert binary measurement results into slot indices (0..layers).
    """
    total = sum(counts.values())
    dist = {}
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        dist[idx] = dist.get(idx, 0) + count
    return {k: v / total for k, v in sorted(dist.items())}


if __name__ == "__main__":
    layers = 4
    theta = np.pi / 2  # unbiased

    qc, counts = run_qgb_optimized(layers, theta)
    dist = postprocess_qgb_optimized(counts, layers)

    print("Optimized QGB Output Distribution:")
    for k in sorted(dist):
        print(f"Bin {k}: {dist[k]:.4f}")

    # Plot
    plt.bar(dist.keys(), dist.values(), color="green", alpha=0.7)
    plt.title(f"Optimized Quantum Galton Board - {layers} layers (theta={theta:.2f})")
    plt.xlabel("Slot Index")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
