# circuits/quantum_peg.py
from qiskit import QuantumCircuit, Aer, transpile, execute
import numpy as np
import matplotlib.pyplot as plt


def quantum_peg(theta=np.pi/2) -> QuantumCircuit:
    """
    Build a single quantum peg using CSWAP and Rx(θ).
    Peg has 1 control qubit, 1 input ("ball"), and 2 output paths.
    """
    qc = QuantumCircuit(4, 3)  # q0: control, q1: left, q2: ball, q3: right

    # Initialize ball on middle wire (q2)
    qc.x(2)

    # Control qubit in biased superposition
    qc.rx(theta, 0)

    # Controlled-SWAP: ball ↔ left
    qc.cswap(0, 1, 2)

    # Controlled-SWAP: ball ↔ right
    qc.cswap(0, 2, 3)

    # Measure all output channels (not the control)
    qc.measure([1, 2, 3], [0, 1, 2])

    return qc


if __name__ == "__main__":
    peg = quantum_peg(theta=np.pi / 2)  # unbiased peg
    backend = Aer.get_backend("qasm_simulator")
    compiled = transpile(peg, backend)
    job = execute(compiled, backend=backend, shots=1024)
    result = job.result().get_counts()

    # Plot
    plt.bar(result.keys(), result.values(), color="teal", alpha=0.7)
    plt.title("Quantum Peg Output (1 peg)")
    plt.xlabel("Measurement")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
