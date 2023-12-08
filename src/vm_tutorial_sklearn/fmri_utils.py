import numpy as np
from scipy.stats import zscore as scipy_zscore


def explainable_variance(Y, do_zscore=True, bias_correction=True):
    """ Computes the explainable variance across repetition of voxel responses.
    Explainable variance is the amount of variance in a voxel's response that can be explained
    by the mean response across several repetitions. Repetitions are recorded while the voxel
    is exposed to the same stimulus several times.

    Parameters
    ----------
    Y : np.ndarray (nrepeats, nTRs, nvoxels)
        Repeated time course for each voxel. Each voxel and repeat is nTRs long.
        Repeats should be zscored across time samples.

    do_zscore : bool
        z-score the data across time. Only set to False
        if Y across time is already z-scored. Default is True.
    bias_correction : bool
        Bias correction for the number of repetitions

    Returns
    -------
    ev : np.array (nvoxels, 1)
        Explainable variance per voxel

    References
    ----------
    Schoppe et al. 2016, Hsu et al. 2004

    Compare to https://github.com/gallantlab/tikreg/blob/master/tikreg/utils.py

    """

    if do_zscore:
        Y = scipy_zscore(Y, axis=1)

    res = Y - Y.mean(axis=0)  # mean across reps
    res_var = np.mean(res.var(axis=1), axis=0)
    ev = 1 - res_var

    if bias_correction:
        ev = ev - ((1 - ev) / np.float((Y.shape[0] - 1)))

    return ev


def shuffle_ts(ts, block_len=10):
    """Given a voxel time series (ts) returns a shuffled version of the ts and
    the indexes.
    Use block_len to keep nearby timepoints together. block_len should not
    be set to shorter than 5 for fMRI voxel time series.
    """
    blocks = np.array(np.array_split(range(np.array(ts).shape[0]),
                      int(np.array(ts).shape[0] / block_len)))
    idx = np.random.permutation(len(blocks))
    idx = np.hstack(blocks[idx])
    return ts[idx], idx


def shuffle_data(data, block_len=10):
    """Given a data matrix with each column vector representing a timeseries,
    block shuffle the data across time (e.g. data = (time, voxels)).
    """
    shuffled_data = []
    idx_data = []
    for c in np.arange(data.shape[1]):
        shuffled_ts, idx = shuffle_ts(data[:, c], block_len)
        shuffled_data.append(shuffled_ts)
        idx_data.append(idx)

    return np.array(shuffled_data).T, np.array(idx_data).T


def shuffle_voxel_ts(ts, block_len=10):
    """Given a voxel time series (ts) return a shuffled version of the ts.
    Use block_len to keep nearby timepoints together. block_len should not
    be set to shorter than 5 for fMRI voxel time series.
    """
    blocks = np.array_split(ts, int(ts.shape[0] / block_len))
    _ = np.random.shuffle(blocks)
    out = np.hstack(blocks)
    return out
