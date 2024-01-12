from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import periodogram

from tqdm import tqdm

# function computing the PSD of a time-series
def compute_psd(
    feature_set: Dict,
    timescale: str,
    agg_method: str = "mean",
    sampling_rate: float = 0.5,
    fft_size: Optional[int] = None,
):
    """
    Compute the PSD of a feature set for a given timescale.

    Parameters
    ----------
    feature_set : Dict
        Dictionary containing the feature set.
    timescale : str
        Timescale of the feature set.
    agg_method : str
        Aggregation method to use for the PSD. Options are 'mean', 'max' or 'min'.

    Returns
    -------
    psd : np.ndarray
        PSD of the feature set for the given timescale.
    """
    feature = feature_set[timescale]
    n_feature = feature.shape[1]
    # n_feature=10
    ps = []
    f = []
    for i in tqdm(range(n_feature)):
        f, p = periodogram(feature[:, i], fs=sampling_rate, nfft=fft_size)

        ps.append(p)

    psd = np.vstack(ps)

    assert psd.shape[0] == n_feature
    if agg_method == "max":
        psd = np.max(psd, axis=0)
    elif agg_method == "min":
        psd = np.min(psd, axis=0)
    else:  # agg_method == 'mean':
        psd = np.mean(psd, axis=0)

    return {"f": f, "psd": psd}


def plot_psd(psd, periods, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for p in periods:
        res = psd[p]

        sns.lineplot(x=res["f"][1:-1], y=res["psd"][1:-1], ax=ax, label=p)

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")

    ax.legend()
    ax.set_yscale("log")

    return ax


def compute_timescale_crosscorrelation(feature_set, periods, agg_method="mean"):
    """
    Compute the cross-correlation of the PSD of a feature set for a given timescale.

    Parameters
    ----------
    feature_set : Dict
        Dictionary containing the feature set.
    periods : list
        List of timescales of the feature set.
    agg_method : str
        Aggregation method to use. Options are 'mean', 'max' or 'min'.

    Returns
    -------
    crosscorr : np.ndarray
        Cross-correlation of the PSD of the feature set for the given timescale.
    """
    
    # aggregating over all features
    features = []
    for p in periods:
        if agg_method == "max":
            f = np.max(feature_set[p], axis=1)
        elif agg_method == "min":
            f = np.min(feature_set[p], axis=1)
        else:  # agg_method == 'mean':
            f = np.mean(feature_set[p], axis=1)
        features.append(f)

    # stack em up
    features = np.vstack(features)

    # compute cross-correlation across timescales
    crosscorr = np.corrcoef(features)

    return crosscorr