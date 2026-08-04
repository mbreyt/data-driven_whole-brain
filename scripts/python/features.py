"""BOLD summary-feature extraction.

Two feature definitions are used in the paper, one per analysis, and they are
intentionally different — do not merge them:

- :func:`get_features_sims` — the high-dimensional vector used for SBI on
  *simulations* (FC/FCD statistics, per-network integration/segregation, and
  per-node time-series statistics). Consumed by
  ``notebooks/sbi-on-sims-training.ipynb``.

- :func:`get_features_data` — a reduced 7-feature vector used for SBI on the
  *empirical* subject. The same extraction generated the ``sbi_data`` training
  set (stored feature arrays are ``(N, 7)``). The empirical inference uses
  features ``[5, 6] = [FCD skewness, BOLD zero-crossing]``.

Both share the sliding-window functional-connectivity stream computed by
:func:`get_metrics`.

Leading feature order is identical for the first six entries::

    0 sum_fc   1 var_fc   2 fl   3 fcd_min   4 fcd_max   5 skew(fcd)

and then differs: ``get_features_sims`` appends ``matrix_stat`` and per-network
features (index 6 onward), whereas ``get_features_data`` appends the BOLD
zero-crossing count (index 6).
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as st

# `features_utils.matrix_stat` is provided by the `vbi` package.
from vbi.feature_extraction import features_utils

#: Number of parcels (Schaefer 100).
N_NODES = 100
#: Sliding-window length and step (in TRs) for the FCD stream.
WIN_LEN = 15
WIN_SP = 2


# --- Sliding-window FC stream (shared) -------------------------------------

@jax.jit
def fcd_fun_bold(c, x):
    win_len = WIN_LEN
    get = jax.lax.dynamic_slice
    x = x[0].astype(int)
    ts = get(c, (x, 0), (win_len, N_NODES))
    fc = jnp.corrcoef(ts.squeeze().T)
    return c, (fc[jnp.triu_indices_from(fc, k=1)], fc)


@jax.jit
def get_metrics(ts):
    win_len = WIN_LEN
    win_sp = WIN_SP
    x_range = jnp.arange(0, ts.shape[0] - win_len, win_sp)[..., None]
    c, (fc, fc_stream) = jax.lax.scan(fcd_fun_bold, ts, x_range)
    return fc_stream


# --- Feature vectors -------------------------------------------------------

def get_features_sims(ts, fc_stream, names_idx, win_len, win_sp):
    """High-dimensional feature vector for SBI on simulations."""
    ts2 = ts.copy()
    ts = st.zscore(ts, axis=0)
    k = (float(win_len) / (float(win_len) - float(win_len - win_sp))) + 1
    temp = []
    fc = np.corrcoef(ts.T)
    sum_fc = np.sum(fc)
    var_fc = np.var(fc)
    triu_idx = np.triu_indices_from(fc, 1)
    triu_fc_stream = fc_stream[:, triu_idx[0], triu_idx[1]]
    fcd = np.corrcoef(triu_fc_stream)
    fl = np.var(fcd[np.triu_indices_from(fcd, k)])
    sk = st.skew(fcd[np.triu_indices_from(fcd, k)])
    fcd_min = np.min(fcd[np.triu_indices_from(fcd, k)])
    fcd_max = np.max(fcd[np.triu_indices_from(fcd, k)])
    temp += [sum_fc, var_fc, fl, fcd_min, fcd_max, sk]
    vbi_feats, vbi_labels = features_utils.matrix_stat(fc)
    temp += vbi_feats
    vbi_feats, vbi_labels = features_utils.matrix_stat(fcd, k=k)
    temp += vbi_feats
    for i in range(len(names_idx) - 1):
        k = (float(win_len) / (float(win_len) - float(win_len - win_sp))) + 1
        fc = np.corrcoef(ts.T)
        net_start = names_idx[i]
        net_stop = names_idx[i + 1]

        # static fc
        net_triu_idx = np.triu_indices_from(fc[net_start:net_stop, net_start:net_stop], 1)  # get triangle of network fc diagonal block
        in_fc = np.sum(fc[net_start:net_stop, net_start:net_stop][net_triu_idx])  # sum segregated fc

        fc[net_start:net_stop, net_start:net_stop] = 0
        out_fc = np.sum(fc[net_start:net_stop, :])  # sum integrated fc after zeroing block

        # fc stream
        sub_fcs_in = fc_stream[:, net_start:net_stop, net_start:net_stop]  # select diagonal block
        r, c = np.triu_indices_from(sub_fcs_in[0], 1)
        sub_fcd_in = sub_fcs_in[:, r, c]
        sub_fcd_in = np.corrcoef(sub_fcd_in)  # segregated fcd
        in_fl = np.var(sub_fcd_in[np.triu_indices_from(sub_fcd_in, k)])  # segregated fluidity
        in_sk = st.skew(sub_fcd_in[np.triu_indices_from(sub_fcd_in, k)])  # segregated skewness
        in_max = np.max(sub_fcd_in[np.triu_indices_from(sub_fcd_in, k)])  # segregated max
        in_min = np.min(sub_fcd_in[np.triu_indices_from(sub_fcd_in, k)])  # segregated min
        sub_fcs_out = np.concatenate([fc_stream[:, net_start:net_stop, :net_start], fc_stream[:, net_start:net_stop, net_stop:]], axis=2)  # row without diag block
        sub_fcd_out = sub_fcs_out.reshape((sub_fcs_out.shape[0], -1))
        sub_fcd_out = np.corrcoef(sub_fcd_out)  # integrated fcd
        out_fl = np.var(sub_fcd_out[np.triu_indices_from(sub_fcd_out, k)])
        out_sk = st.skew(sub_fcd_out[np.triu_indices_from(sub_fcd_out, k)])
        out_max = np.max(sub_fcd_out[np.triu_indices_from(sub_fcd_out, k)])
        out_min = np.min(sub_fcd_out[np.triu_indices_from(sub_fcd_out, k)])
        temp += [in_fc, in_fl, in_sk, in_min, in_max, out_fc, out_fl, out_sk, out_min, out_max]
    temp += ((ts[:-1] * ts[1:]) < 0).sum(axis=0).tolist()
    temp += np.min(ts, axis=0).tolist()
    temp += np.max(ts, axis=0).tolist()
    temp += np.mean(ts2, axis=0).tolist()
    temp += np.var(ts2, axis=0).tolist()
    return np.array(temp)


def get_features_data(ts, fc_stream, names_idx, win_len, win_sp):
    """Reduced 7-feature vector for SBI on the empirical subject.

    Returns ``[sum_fc, var_fc, fl, fcd_min, fcd_max, skew(fcd), zero_crossing]``.
    The downstream inference selects ``[[5, 6]] = [skew(fcd), zero_crossing]``.
    """
    ts = st.zscore(ts, axis=0)
    k = (float(win_len) / (float(win_len) - float(win_len - win_sp))) + 1
    temp = []
    fc = np.corrcoef(ts.T)
    sum_fc = np.mean(fc)
    var_fc = np.var(fc)
    triu_idx = np.triu_indices_from(fc, 1)
    triu_fc_stream = fc_stream[:, triu_idx[0], triu_idx[1]]
    fcd = np.corrcoef(triu_fc_stream)
    fl = np.var(fcd[np.triu_indices_from(fcd, k)])
    sk = st.skew(fcd[np.triu_indices_from(fcd, k)])
    fcd_min = np.min(fcd[np.triu_indices_from(fcd, k)])
    fcd_max = np.max(fcd[np.triu_indices_from(fcd, k)])
    temp += [sum_fc, var_fc, fl, fcd_min, fcd_max, sk]
    temp += [((ts[:-1] * ts[1:]) < 0).sum(axis=0).mean()]
    return np.array(temp)
