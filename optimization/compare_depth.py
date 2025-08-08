# optimization/compare_depth.py
# Compare depth and gate count of original and optimized QGB circuits.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from circuits.quantum_peg_multi import generate_qgb
from optimization.quantum_peg_optimized import generate_qgb_optimized
from qiskit import transpile, Aer

def compare_depths(max_layers=5):
    backend = Aer.get_backend("qasm_simulator")
    print(f"{'Layers':<10}{'Original Depth':<20}{'Optimized Depth':<20}{'Gate Reduction (%)':<20}")
    print("-" * 70)
    
    for L in range(1, max_layers + 1):
        orig_qc = generate_qgb(L)
        opt_qc = generate_qgb_optimized(L)
        
        # Transpile both for fair comparison
        orig_compiled = transpile(orig_qc, backend, optimization_level=3)
        opt_compiled = transpile(opt_qc, backend, optimization_level=3)
        
        orig_depth = orig_compiled.depth()
        opt_depth = opt_compiled.depth()
        
        reduction = (1 - opt_depth / orig_depth) * 100 if orig_depth > 0 else 0
        print(f"{L:<10}{orig_depth:<20}{opt_depth:<20}{reduction:<20.2f}")

if __name__ == "__main__":
    compare_depths(max_layers=5)
