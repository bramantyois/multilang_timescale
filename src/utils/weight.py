from typing import Optional

import copy

import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Primal Coefs utils
def undelay_weights(signal, delays):
    """Gets undelayed weights corresponding to features fit using delay_signal.
    args:
        signal: [(num_signal_dims * num_delays), num_voxels]
    returns:
        undelayed_signal: [num_delays, num_signal_dims, num_voxels]
    """
    if signal.ndim == 1:
        signal = signal[..., None]
    num_delayed_dims, num_voxels = signal.shape
    num_signal_dims = num_delayed_dims // len(delays)
    undelayed_signal = np.ones(
        (len(delays), num_signal_dims, num_voxels), dtype=signal.dtype
    )

    for delay_index, delay in enumerate(delays):
        begin, end = delay_index * num_signal_dims, (delay_index + 1) * num_signal_dims
        undelayed_signal[delay_index, :, :] = signal[begin:end]
    return undelayed_signal


def scale_weights_by_score(
    primal_weights: np.ndarray, scores: np.ndarray, normalize: bool = True
):
    """Scale weights by sqrt of scores."""
    primal_weights = copy.deepcopy(primal_weights)
    if normalize:
        norm = np.linalg.norm(primal_weights, axis=0)
        primal_weights[:, norm != 0] /= norm[norm != 0]
    primal_weights *= np.maximum(0, scores)
    return primal_weights


def process_primal_weight(
    weights: np.ndarray, 
    prediction_score: np.ndarray, 
    delay: int = 4,
    normalize: bool = False,
):
    """
    Process primal weights
    """
    delays = np.arange(1, delay + 1)

    primal_weights = undelay_weights(weights, delays).mean(0)
    primal_weights = scale_weights_by_score(primal_weights, prediction_score)
    primal_weights = np.nan_to_num(primal_weights).T # n_voxel x n_feature

    # normalize to norm==1
    if normalize:
        primal_weights = primal_weights / np.linalg.norm(primal_weights, axis=0)
        
    return primal_weights


def project_weights_to_pcs(
    weights: np.ndarray, n_components: int = 10, pipeline: Optional[Pipeline] = None
):
    """
    Project weights to PCs
    """
    if pipeline is None:
        steps = [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
        ]

        pipeline = Pipeline(steps)
        pipeline.fit(weights)

    weights_pca = pipeline.transform(weights)

    # scaler = StandardScaler()
    # scaled = scaler.fit_transform(weights)

    # pca = PCA(n_components=n_components)
    # weights_pca = pca.fit_transform(scaled)

    return weights_pca, pipeline


def project_weights_to_rgb(weights: np.ndarray, pipeline: Optional[Pipeline] = None):
    """
    Project weights to RGB
    """
    weights_pca = project_weights_to_pcs(weights, n_components=3, pipeline=pipeline)[0]

    # normalize and scale to 0-255
    weights_rgb = (
        (weights_pca - weights_pca.min())
        / (weights_pca.max() - weights_pca.min())
        * 255
    )
    weights_rgb = weights_rgb.astype(np.uint8)

    return weights_rgb


def load_fasttext_aligned_vectors(fname, skip_first_line=True):
    """Return dictionary of word embeddings from file saved in fasttext format.

    Parameters:
    -----------
    fname : str
        Name of file containing word embeddings.
    skip_first_line : bool
        If True, skip first line of file. Should do this if first line of file
        contains metadata.

    Returns:
    --------
    data : dict
        Dictionary of word embeddings.
    """
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    if skip_first_line:
        n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = np.array([float(token) for token in tokens[1:]])
    return data

