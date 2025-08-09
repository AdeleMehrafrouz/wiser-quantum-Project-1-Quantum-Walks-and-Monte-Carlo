# WISER Quantum Project-1 - Quantum Walks and Monte Carlo

This repository implements a **quantum version of the Galton Box** using Qiskit and simulates various probabilistic distributions, including:

* **Gaussian distributions** via Hadamard gates
* **Exponential distributions** using Rx-biasing
* **Quantum walks** using discrete-time Hadamard quantum circuits

It is developed as part of the **WISER Quantum Project** challenge on quantum-enhanced statistical simulation.

---

## Project Structure

```
project1-Montecarlo/
├── circuits/
│   ├── galton_generator.py         # General Galton Box circuit builder
│   ├── biased_distribution.py      # Simulates exponential distribution with Rx bias
│   ├── hadamard_walk.py            # Quantum walk circuit
│   ├── quantum_peg_multi.py        # Multi-qubit QGB implementation
│   ├── biased_distribution_multi.py
│   ├── hadamard_walk_qiskit.py
│   └── verify_gaussian.py
├── metrics/
│   ├── distance_metrics.py         # TVD and KL divergence computations
│   └── distance_metrics_multi.py
├── results/
│   ├── gaussian_histogram.png
│   ├── exponential_histogram.png
│   ├── hadamard_walk.png
│   ├── noisy_comparison.png
│   └── qgb_gaussian_vs_biased.png
├── docs/
│   ├── 2-page-summary.pdf          # Summary of Carney & Varcoe paper
│   
├── README.md                     # Project documentation
├── run_all.ipynb                 # Interactive notebook to run simulations
├── run_all_multi.ipynb           # Enhanced notebook with multiple distributions
└── noise_vs_ideal.ipynb          # Notebook comparing noisy vs ideal execution
```

---

## Generalized Galton Box Algorithm

This project includes two levels of implementation for the Quantum Galton Box (QGB):

* `.py` files such as `galton_generator.py` are **simple, standalone prototypes**, ideal for testing and foundational demonstration.
* `*_multi.py` files (like `quantum_peg_multi.py`) are **fully generalized implementations** capable of generating QGB circuits for any number of layers. These are modular and integrated into the full pipeline.

This project builds on 1- and 2-layer QGB logic to implement a fully **scalable quantum Galton Box** for any number of layers:

* `quantum_peg_multi.py` defines a configurable function `run_qgb(layers, theta, shots)` to create circuits of arbitrary depth.
* `verify_gaussian.py` automatically evaluates the Gaussian shape of the output.
* `run_all_multi.ipynb` runs both biased and unbiased QGB simulations and compares the distributions.

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

3. **Run modules individually or explore the notebooks:**

```bash
# Simple prototype scripts:
python circuits/galton_generator.py
python circuits/biased_distribution.py
python circuits/hadamard_walk.py
python metrics/distance_metrics_multi.py

# Generalized and modular simulations:
python circuits/galton_generator_multi.py
python circuits/biased_distribution_multi.py
python circuits/hadamard_walk_qiskit.py
python circuits/verify_gaussian.py
python metrics/distance_metrics_multi.py
```

You can also explore:

* `run_all_multi.ipynb` for full pipeline simulation & plots
* `noise_vs_ideal.ipynb` for noise model vs ideal circuit comparison

4. **Explore results:**
   All `.png` visualizations are saved in the `results/` directory.

---

## Output Samples

* Gaussian and Exponential histograms
* Quantum walk distribution
* TVD and KL divergence plots
* Noise vs ideal comparisons

---

## Challenge Deliverables

* 2-page summary of the Carney & Varcoe paper
* General quantum Galton Box implementation for any number of layers
* Simulation of alternative distributions
* Noise model implementation & analysis
* Distance metrics evaluation

---

## Team

* Adele Mehrafrouz
* WISER Quantum Project

---

## References

* Carney, M. & Varcoe, B. (2022). *Universal Statistical Simulator*. [arXiv:2202.01735](https://arxiv.org/abs/2202.01735)
* Qiskit Documentation: [https://qiskit.org/documentation/](https://qiskit.org/documentation/)
