# Wiser-quantum-Project1 - Quantum Walks and Monte Carlo

This repository implements a **quantum version of the Galton Box** using Qiskit and simulates various probabilistic distributions, including:
- **Gaussian distributions** via Hadamard gates
- **Exponential distributions** using Rx-biasing
- **Quantum walks** using discrete-time Hadamard quantum circuits

It is developed as part of the **Womanium + WISER 2025 Quantum Program** challenge on quantum-enhanced statistical simulation.

---

## Project Structure

```
project1-Montecarlo/
├── circuits/
│   ├── galton_generator.py         # General Galton Box circuit builder
│   ├── biased_distribution.py      # Simulates exponential distribution with Rx bias
│   └── hadamard_walk.py            # Quantum walk circuit
├── results/
│   ├── gaussian_histogram.png
│   ├── exponential_histogram.png
│   ├── hadamard_walk.png
│   └── noisy_comparison.png
├── metrics/
│   └── distance_metrics.py         # TVD and KL divergence computations
├── docs/
│   ├── 2-page-summary.pdf          # Summary of Carney & Varcoe paper
│   └── README.md                   # Project documentation
└── run_all.ipynb                   # Interactive notebook to run simulations
```

---

## How to Run

1. **Create a virtual environment:**
```bash
python -m venv qugalton-env
source qugalton-env/bin/activate  # Or use `qugalton-env\Scripts\activate` on Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run individual modules:**
```bash
python circuits/galton_generator.py
python circuits/biased_distribution.py
python circuits/hadamard_walk.py
python metrics/distance_metrics.py
```

4. **Explore results:**
All `.png` visualizations are saved in the `results/` directory.

---

## Output Samples

- Gaussian and Exponential histograms
- Quantum walk distribution
- Distance metrics comparison
- Noisy vs ideal simulations

---

## Challenge Deliverables

- 2-page summary of the Carney & Varcoe paper
- General quantum Galton Box implementation
- Simulation of alternative distributions
- Noisy model optimization
- Distance metrics analysis

---

## Team

- Adele Mehrafrouz
- Womanium + WISER Quantum Program 2025 Participant

---

## References

- Carney, M. & Varcoe, B. (2022). *Universal Statistical Simulator*. [arXiv:2202.01735](https://arxiv.org/abs/2202.01735)
- Qiskit Documentation: https://qiskit.org/documentation/

