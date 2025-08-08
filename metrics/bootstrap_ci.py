# metrics/bootstrap_ci.py

import numpy as np

def bootstrap_tvd_ci(p, q, shots=2048, iters=1000, alpha=0.05):
    keys = sorted(set(p.keys()) | set(q.keys()))
    pvals = np.array([p.get(k, 0.0) for k in keys])
    qvals = np.array([q.get(k, 0.0) for k in keys])

    tvds = []
    for _ in range(iters):
        p_counts = np.random.multinomial(shots, pvals)
        q_counts = np.random.multinomial(shots, qvals)
        p_hat = p_counts / shots
        q_hat = q_counts / shots
        tvds.append(0.5 * np.sum(np.abs(p_hat - q_hat)))

    lo, hi = np.percentile(tvds, [100*alpha/2, 100*(1-alpha/2)])
    return np.mean(tvds), (lo, hi)
