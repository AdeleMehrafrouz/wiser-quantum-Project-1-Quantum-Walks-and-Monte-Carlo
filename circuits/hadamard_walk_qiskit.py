# circuits/hadamard_walk_qiskit.py
# Proper Qiskit implementation of a small Hadamard quantum walk
# using controlled modular +/- 1 on a binary-encoded position register.

from qiskit import QuantumCircuit, Aer, transpile, execute
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List


# ---------- Small helper adders ----------

def c_increment_mod_2n(qc: QuantumCircuit, ctrl: int, reg: List[int]):
    """
    Controlled increment by +1 (mod 2^n) on 'reg', controlled on ctrl qubit being |1>.
    Ripple-carry style: bit i flips iff ctrl==1 and all lower bits are 1 (carry).
    """
    # flip LSB if ctrl==1
    qc.cx(ctrl, reg[0])
    # propagate carry
    for i in range(1, len(reg)):
        # controls are [ctrl] + all lower bits
        controls = [ctrl] + reg[:i]
        qc.mcx(controls, reg[i])  # multi-controlled X (Toffoli generalization)


def c_decrement_mod_2n(qc: QuantumCircuit, ctrl: int, reg: List[int]):
    """
    Controlled decrement by -1 (mod 2^n) using two's complement trick:
    x - 1 = ~ ( (~x) + 1 )
    So do: X all bits -> c_increment -> X all bits.
    """
    for q in reg:
        qc.x(q)
    c_increment_mod_2n(qc, ctrl, reg)
    for q in reg:
        qc.x(q)


# ---------- The walk itself ----------

def hadamard_walk_circuit(steps: int) -> QuantumCircuit:
    """
    Build a proper Qiskit circuit for a 1D Hadamard quantum walk with 'steps' steps.
    - 1 coin qubit
    - binary-encoded position register (ceil(log2(2*steps+1)) qubits)
    - controlled modular increment/decrement each step
    """
    pos_qubits = int(np.ceil(np.log2(2 * steps + 1)))
    total_qubits = 1 + pos_qubits  # 1 coin + position register
    qc = QuantumCircuit(total_qubits, pos_qubits)

    coin = 0
    pos = list(range(1, total_qubits))

    # Start walker at the center position (= steps)
    start = steps  # integer
    bin_start = format(start, f"0{pos_qubits}b")
    for i, bit in enumerate(reversed(bin_start)):
        if bit == "1":
            qc.x(pos[i])

    # Perform the walk
    for _ in range(steps):
        qc.h(coin)

        # coin = |1>  -> move right (+1)
        c_increment_mod_2n(qc, coin, pos)

        # coin = |0>  -> move left (-1)
        qc.x(coin)
        c_decrement_mod_2n(qc, coin, pos)
        qc.x(coin)

    # Measure position
    qc.measure(pos, range(pos_qubits))
    return qc


def run_walk_simulation(steps: int, shots: int = 2048) -> Dict[int, float]:
    qc = hadamard_walk_circuit(steps)
    backend = Aer.get_backend("qasm_simulator")
    compiled = transpile(qc, backend)
    job = execute(compiled, backend=backend, shots=shots)
    counts = job.result().get_counts()

    total = sum(counts.values())
    # Convert binary indices to integers (these are 0..2^n-1, centered at 'steps')
    dist = {}
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        # shift to [-steps, +steps] to be human-friendly
        pos_qubits = len(bitstring)
        span = 2 ** pos_qubits
        # We started at 'steps', so interpret idx relative to that
        rel = idx - steps
        dist[rel] = dist.get(rel, 0) + c / total
    return dist


# ---------- quick executable demo ----------

if __name__ == "__main__":
    steps = 3
    shots = 4096

    dist = run_walk_simulation(steps, shots)
    print("Qiskit Hadamard Walk Output Distribution:")
    for k in sorted(dist):
        print(f"Position {k:+d}: {dist[k]:.4f}")

    os.makedirs("results", exist_ok=True)
    plt.bar(dist.keys(), dist.values(), color="purple", alpha=0.7)
    plt.title(f"Qiskit Hadamard Quantum Walk - {steps} Steps")
    plt.xlabel("Position (centered)")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/hadamard_walk_qiskit.png")
    plt.show()
