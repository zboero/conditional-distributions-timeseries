import json
import numpy as np

def safe_jsonable(v):
    try:
        json.dumps(v, default=str)
        return v
    except Exception:
        return str(v)

def make_lagged_pairs(x: np.ndarray, k: int):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k < 1 or k >= n:
        raise ValueError("k must satisfy 1 <= k < len(x).")
    Y = np.zeros((n - k, k), dtype=float)
    y_next = x[k:].copy()
    for i in range(n - k):
        Y[i, :] = x[i + k - 1 : i - 1 : -1] if i > 0 else x[k - 1 :: -1]
    return Y, y_next

def robust_standardize_matrix(Y: np.ndarray, eps: float = 1e-12):
    med = np.median(Y, axis=0)
    mad = np.median(np.abs(Y - med), axis=0)
    scale = 1.4826 * mad
    scale = np.maximum(scale, eps)
    Z = (Y - med) / scale
    return Z, med, scale

def gaussian_kernel(u2: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u2)

def nw_weights(Yz: np.ndarray, qz: np.ndarray, h: float, min_wsum: float = 1e-12):
    d2 = np.sum((Yz - qz[None, :]) ** 2, axis=1)
    h2 = max(h, 1e-12) ** 2
    w = gaussian_kernel(d2 / h2)
    wsum = float(np.sum(w))
    if wsum < min_wsum:
        w = np.ones_like(w) / len(w)
    else:
        w = w / wsum
    return w

def weighted_cdf_at_grid(x_next: np.ndarray, w: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x_next)
    xs = x_next[order]
    ws = w[order]
    cws = np.cumsum(ws)
    idx = np.searchsorted(xs, x_grid, side="right") - 1
    F = np.where(idx >= 0, cws[np.clip(idx, 0, len(cws)-1)], 0.0)
    return F

def crps_from_cdf(x_grid: np.ndarray, F: np.ndarray, y: float) -> float:
    dx = float(x_grid[1] - x_grid[0])
    H = (x_grid >= y).astype(float)
    return float(np.sum((F - H) ** 2) * dx)

def choose_bandwidth_crps_snapshot(x: np.ndarray, k: int, x_grid: np.ndarray, h_grid: np.ndarray,
                                  cv_points: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n <= k + 60:
        return float(h_grid[len(h_grid)//2])

    Y, y_next = make_lagged_pairs(x, k)
    Yz, med, scale = robust_standardize_matrix(Y)

    candidates = np.arange(k + 10, n)
    if len(candidates) > cv_points:
        eval_t = np.sort(rng.choice(candidates, size=cv_points, replace=False))
    else:
        eval_t = candidates

    best_h, best = None, np.inf
    for h in h_grid:
        scores = []
        for t in eval_t:
            i_max = (t - 1) - k
            if i_max < 30:
                continue
            Yz_train = Yz[: i_max + 1]
            y_train = y_next[: i_max + 1]

            q = x[t-1:t-k-1:-1]
            qz = (q - med) / scale
            w = nw_weights(Yz_train, qz, h)
            F = weighted_cdf_at_grid(y_train, w, x_grid)
            scores.append(crps_from_cdf(x_grid, F, x[t]))
        if scores:
            m = float(np.mean(scores))
            if m < best:
                best = m
                best_h = float(h)
    return best_h if best_h is not None else float(h_grid[len(h_grid)//2])

def compute_snapshot_multik_nw(
    x: np.ndarray,
    k_values,
    grid_size: int = 600,
    grid_quantiles=(0.001, 0.999),
    cv_points: int = 200,
):
    """
    Computes one conditional CDF per k for forecasting X_{T+1} beyond snapshot.
    Returns:
      x_grid (G,), cdf_stack (K,G), h_used (K,)
    """
    x = np.asarray(x, dtype=float)
    lo, hi = np.quantile(x, grid_quantiles[0]), np.quantile(x, grid_quantiles[1])
    x_grid = np.linspace(lo, hi, grid_size)

    h_grid = np.array([0.5, 0.7, 0.9, 1.1, 1.4, 1.8], dtype=float)

    cdf_stack = []
    h_used = []

    for k in k_values:
        k = int(k)
        if len(x) <= k + 20:
            cdf_stack.append(np.full_like(x_grid, np.nan))
            h_used.append(np.nan)
            continue

        h = choose_bandwidth_crps_snapshot(x, k, x_grid, h_grid, cv_points=cv_points, seed=0)
        Y, y_next = make_lagged_pairs(x, k)
        Yz, med, scale = robust_standardize_matrix(Y)

        q = x[-1:-k-1:-1]          # condition on X_T..X_{T-k+1}
        qz = (q - med) / scale

        w = nw_weights(Yz, qz, h)
        F = weighted_cdf_at_grid(y_next, w, x_grid)

        cdf_stack.append(F)
        h_used.append(h)

    return x_grid, np.vstack(cdf_stack), np.asarray(h_used, dtype=float)
