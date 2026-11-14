"""Frozen ACE/MOS/COMET feature maps, with clamping outside training support."""
import numpy as np


def apply_map(x, z, newx):
    x = np.asarray(x, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float)
    newx = np.asarray(newx, dtype=float)
    assert z.ndim in (1, 2) and len(z) == len(x) and len(x) > 0
    assert np.isfinite(x).all() and np.isfinite(z).all() and not np.isnan(newx).any()
    u, first, inverse = np.unique(x, return_index=True, return_inverse=True)
    v = z[first]
    assert np.array_equal(z, v[inverse]), 'Equal inputs must have equal fitted scores.'
    query = np.clip(newx.reshape(-1), u[0], u[-1])
    if len(u) == 1:
        answer = np.broadcast_to(v[0], (query.size,) + z.shape[1:]).copy()
    else:
        index = np.clip(np.searchsorted(u, query, side='right') - 1, 0, len(u) - 2)
        t = (query - u[index]) / (u[index + 1] - u[index])
        if z.ndim == 2:
            t = t[:, None]
        left, right = v[index], v[index + 1]
        answer = left + t * (right - left)
        answer = np.clip(answer, np.minimum(left, right), np.maximum(left, right))
        answer[query == u[0]] = v[0]
        answer[query == u[-1]] = v[-1]
    return answer.reshape(newx.shape) if z.ndim == 1 else answer
