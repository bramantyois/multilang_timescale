import time
import logging

import h5py

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.signal import periodogram

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from .config import timescales, timescale_ranges


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


# timescale selectivity
def compute_timescale_selectivity(timescale_scores: np.ndarray) -> np.ndarray:
    """
    Compute the timescale selectivity from scores.

    Parameters
    ----------
    timescale_scores : np.ndarray
        Scores of the timescale selectivity. Shape is (n_timescale, n_voxel).

    Returns
    -------
    selectivity : np.ndarray
        Timescale selectivity. Shape is (n_voxel,).
    """
    nz_scores = np.maximum(timescale_scores, 0)
    score_sum = np.sum(nz_scores, axis=0)

    normalized_scores = np.nan_to_num(nz_scores / score_sum)

    mid_ranges = np.array(
        [np.mean(timescale_ranges[key]) for key in timescales]
    )
    mid_ranges = np.log2(mid_ranges)

    weighted_scores = np.stack(
        [
            normalized_scores[i,] * mid_ranges[i]
            for i in range(normalized_scores.shape[0])
        ]
    )

    return weighted_scores.sum(axis=0)


# permutation test
def permutation_test(
    targets: np.ndarray,
    predictions: np.ndarray,
    score_func: callable,
    num_permutations: int = 1000,
    permutation_block_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the p-values of the given predictions using a permutation test.

    Parameters
    ----------
    targets : np.ndarray
        Ground truth.
    predictions : np.ndarray
        Predicted values.
    score_func : callable
        Callable function to compute the score.
    num_permutations : int, optional
        Number of permutations, by default 1000
    permutation_block_size : int, optional
        Block size, intended to keep correlation high, by default 10

    Returns
    -------
    pvalues : np.ndarray
        p-values.
    true_scores : np.ndarray
        True scores.
    """

    true_scores = score_func(targets, predictions)
    
    num_TRs = predictions.shape[0]
    blocks = np.array_split(np.arange(num_TRs), int(num_TRs / permutation_block_size))
    
    num_get_true_score = np.zeros(true_scores.shape)

    for permutation_num in tqdm(range(num_permutations)):
        _ = np.random.shuffle(blocks)
        permutation_order = np.concatenate(blocks)
        shuffled_pred = predictions[permutation_order]
        shuffled_scores = score_func(targets, shuffled_pred)
        num_get_true_score[shuffled_scores >= true_scores] += 1
    pvalues = num_get_true_score / num_permutations
    
    return pvalues, true_scores


# Below are codes taken from git_address


# Visualizing on cortical surfaces using mapper files
def load_sparse_array(fname, varname):
    """Load a numpy sparse array from an hdf file

    Parameters
    ----------
    fname: string
        file name containing array to be loaded
    varname: string
        name of variable to be loaded

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.

    By Mark Lescroart

    """
    with h5py.File(fname, mode="r") as hf:
        data = (
            hf["%s_data" % varname],
            hf["%s_indices" % varname],
            hf["%s_indptr" % varname],
        )
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf["%s_shape" % varname])
    return sparsemat


def map_to_flat(voxels, mapper_file):
    """Generate flatmap image for an individual subject from voxel array

    This function maps a list of voxels into a flattened representation
    of an individual subject's brain.

    Parameters
    ----------
    voxels: array
        n x 1 array of voxel values to be mapped
    mapper_file: string
        file containing mapping arrays

    Returns
    -------
    image : array
        flatmap image, (n x 1024)

    By Mark Lescroart

    """
    pixmap = load_sparse_array(mapper_file, "pixmap")
    with h5py.File(mapper_file, mode="r") as hf:
        pixmask = hf["pixmask"][()]
    badmask = np.array(pixmap.sum(1) > 0).ravel()
    img = (np.nan * np.ones(pixmask.shape)).astype(voxels.dtype)
    mimg = (np.nan * np.ones(badmask.shape)).astype(voxels.dtype)
    mimg[badmask] = (pixmap * voxels.ravel())[badmask].astype(mimg.dtype)
    img[pixmask] = mimg
    return img.T[::-1]


# Loading dictioneries
def recursively_save_dict_contents_to_group(h5file, path, filedict):
    """ """
    for key, item in list(filedict.items()):
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def recursively_load_dict_contents_from_group(h5file, path):
    """ """
    filedict = {}
    for key, item in list(h5file[path].items()):
        if isinstance(item, h5py._hl.dataset.Dataset):
            filedict[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            filedict[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return filedict


def save_dict(filename, filedict):
    """Saves the variables in [filedict]."""

    with h5py.File(filename, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, "/", filedict)


def load_dict(filename):
    with h5py.File(filename, "r") as h5file:
        filedict = recursively_load_dict_contents_from_group(h5file, "/")
    return filedict


def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  ## negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


def best_corr_vec(wvec, vocab, SU, n=10):
    """Returns the [n] words from [vocab] most similar to the given [wvec], where each word is represented
    as a row in [SU].  Similarity is computed using correlation."""
    wvec = wvec - np.mean(wvec)
    nwords = len(vocab)
    corrs = np.nan_to_num(
        [
            np.corrcoef(wvec, SU[wi, :] - np.mean(SU[wi, :]))[1, 0]
            for wi in range(nwords - 1)
        ]
    )
    scorrs = np.argsort(corrs)
    words = list(reversed([(corrs[i], vocab[i]) for i in scorrs[-n:]]))
    return words


def get_word_prob():
    """Returns the probabilities of all the words in the mechanical turk video labels."""
    import constants as c
    import cPickle

    data = cPickle.load(open(c.datafile))  # Read in the words from the labels
    wordcount = dict()
    totalcount = 0
    for label in data:
        for word in label:
            totalcount += 1
            if word in wordcount:
                wordcount[word] += 1
            else:
                wordcount[word] = 1

    wordprob = dict([(word, float(wc) / totalcount) for word, wc in wordcount.items()])
    return wordprob


def best_prob_vec(wvec, vocab, space, wordprobs):
    """Orders the words by correlation with the given [wvec], but also weights the correlations by the prior
    probability of the word appearing in the mechanical turk video labels.
    """
    words = best_corr_vec(
        wvec, vocab, space, n=len(vocab)
    )  ## get correlations for all words
    ## weight correlations by the prior probability of the word in the labels
    weightwords = []
    for wcorr, word in words:
        if word in wordprobs:
            weightwords.append((wordprobs[word] * wcorr, word))

    return sorted(weightwords, key=lambda ww: ww[0])


def find_best_words(vectors, vocab, wordspace, actual, display=True, num=15):
    cwords = []
    for si in range(len(vectors)):
        cw = best_corr_vec(vectors[si], vocab, wordspace, n=num)
        cwords.append(cw)
        if display:
            print("Closest words to scene {}:".format(si))
            print([b[1] for b in cw])
            print("Actual words:")
            print(actual[si])
            print("")
    return cwords


def find_best_stims_for_word(wordvector, decstims, n):
    """Returns a list of the indexes of the [n] stimuli in [decstims] (should be decoded stimuli)
    that lie closest to the vector [wordvector], which should be taken from the same space as the
    stimuli.
    """
    scorrs = np.array([np.corrcoef(wordvector, ds)[0, 1] for ds in decstims])
    scorrs[np.isnan(scorrs)] = -1
    return np.argsort(scorrs)[-n:][::-1]


def princomp(x, use_dgesvd=False):
    """Does principal components analysis on [x].
    Returns coefficients, scores and latent variable values.
    Translated from MATLAB princomp function.  Unlike the matlab princomp function, however, the
    rows of the returned value 'coeff' are the principal components, not the columns.
    """

    n, p = x.shape
    # cx = x-np.tile(x.mean(0), (n,1)) ## column-centered x
    cx = x - x.mean(0)
    r = np.min([n - 1, p])  ## maximum possible rank of cx

    if use_dgesvd:
        from svd_dgesvd import svd_dgesvd

        U, sigma, coeff = svd_dgesvd(cx, full_matrices=False)
    else:
        U, sigma, coeff = np.linalg.svd(cx, full_matrices=False)

    sigma = np.diag(sigma)
    score = np.dot(cx, coeff.T)
    sigma = sigma / np.sqrt(n - 1)

    latent = sigma**2

    return coeff, score, latent


def eigprincomp(x, npcs=None, norm=False, weights=None):
    """Does principal components analysis on [x].
    Returns coefficients (eigenvectors) and eigenvalues.
    If given, only the [npcs] greatest eigenvectors/values will be returned.
    If given, the covariance matrix will be computed using [weights] on the samples.
    """
    n, p = x.shape
    # cx = x-np.tile(x.mean(0), (n,1)) ## column-centered x
    cx = x - x.mean(0)
    r = np.min([n - 1, p])  ## maximum possible rank of cx

    xcov = np.cov(cx.T)
    if norm:
        xcov /= n

    if npcs is not None:
        latent, coeff = scipy.linalg.eigh(xcov, eigvals=(p - npcs, p - 1))
    else:
        latent, coeff = np.linalg.eigh(xcov)

    ## Transpose coeff, reverse its rows
    return coeff.T[::-1], latent[::-1]


def weighted_cov(x, weights=None):
    """If given [weights], the covariance will be computed using those weights on the samples.
    Otherwise the simple covariance will be returned.
    """
    if weights is None:
        return np.cov(x)
    else:
        w = weights / weights.sum()  ## Normalize the weights
        dmx = (x.T - (w * x).sum(1)).T  ## Subtract the WEIGHTED mean
        wfact = 1 / (1 - (w**2).sum())  ## Compute the weighting factor
        return wfact * np.dot(w * dmx, dmx.T.conj())  ## Take the weighted inner product


def test_weighted_cov():
    """Runs a test on the weighted_cov function, creating a dataset for which the covariance is known
    for two different populations, and weights are used to reproduce the individual covariances.
    """
    T = 1000  ## number of time points
    N = 100  ## A signals
    M = 100  ## B signals
    snr = 5  ## signal to noise ratio

    ## Create the two datasets
    siga = np.random.rand(T)
    noisea = np.random.rand(T, N)
    respa = (noisea.T + snr * siga).T

    sigb = np.random.rand(T)
    noiseb = np.random.rand(T, M)
    respb = (noiseb.T + snr * sigb).T

    ## Compute self-covariance matrixes
    cova = np.cov(respa)
    covb = np.cov(respb)

    ## Compute the full covariance matrix
    allresp = np.hstack([respa, respb])
    fullcov = np.cov(allresp)

    ## Make weights that will recover individual covariances
    wta = np.ones(
        [
            N + M,
        ]
    )
    wta[N:] = 0

    wtb = np.ones(
        [
            N + M,
        ]
    )
    wtb[:N] = 0

    recova = weighted_cov(allresp, wta)
    recovb = weighted_cov(allresp, wtb)

    return locals()


def fixPCs(orig, new):
    """Finds and fixes sign-flips in PCs by finding the coefficient with the greatest
    magnitude in the [orig] PCs, then negating the [new] PCs if that coefficient has
    a different sign.
    """
    flipped = []
    for o, n in zip(orig, new):
        maxind = np.abs(o).argmax()
        if o[maxind] * n[maxind] > 0:
            ## Same sign, no need to flip
            flipped.append(n)
        else:
            ## Different sign, flip
            flipped.append(-n)

    return np.vstack(flipped)


def plot_model_comparison(corrs1, corrs2, name1, name2, thresh=0.35):
    fig = figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    good1 = corrs1 > thresh
    good2 = corrs2 > thresh
    better1 = corrs1 > corrs2
    # both = np.logical_and(good1, good2)
    neither = np.logical_not(np.logical_or(good1, good2))
    only1 = np.logical_and(good1, better1)
    only2 = np.logical_and(good2, np.logical_not(better1))

    ptalpha = 0.3
    ax.plot(corrs1[neither], corrs2[neither], "ko", alpha=ptalpha)
    # ax.plot(corrs1[both], corrs2[both], 'go', alpha=ptalpha)
    ax.plot(corrs1[only1], corrs2[only1], "ro", alpha=ptalpha)
    ax.plot(corrs1[only2], corrs2[only2], "bo", alpha=ptalpha)

    lims = [-0.5, 1.0]

    ax.plot([thresh, thresh], [lims[0], thresh], "r-")
    ax.plot([lims[0], thresh], [thresh, thresh], "b-")

    ax.text(
        lims[0] + 0.05,
        thresh,
        "$n=%d$" % np.sum(good2),
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    ax.text(
        thresh,
        lims[0] + 0.05,
        "$n=%d$" % np.sum(good1),
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    ax.plot(lims, lims, "-", color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)

    show()
    return fig


import matplotlib.colors

bwr = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bwr", ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
)
bkr = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bkr", ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
)
bgr = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bgr", ((0.0, 0.0, 1.0), (0.5, 0.5, 0.5), (1.0, 0.0, 0.0))
)


def plot_model_comparison2(corrFile1, corrFile2, name1, name2, thresh=0.35):
    fig = figure(figsize=(9, 10))
    # ax = fig.add_subplot(3,1,[1,2], aspect="equal")
    ax = fig.add_axes([0.25, 0.4, 0.6, 0.5], aspect="equal")

    corrs1 = tables.openFile(corrFile1).root.semcorr.read()
    corrs2 = tables.openFile(corrFile2).root.semcorr.read()
    maxcorr = np.clip(np.vstack([corrs1, corrs2]).max(0), 0, thresh) / thresh
    corrdiff = (corrs1 - corrs2) + 0.5
    colors = (bgr(corrdiff).T * maxcorr).T
    colors[:, 3] = 1.0  ## Don't scale alpha

    ptalpha = 0.8
    ax.scatter(corrs1, corrs2, s=10, c=colors, alpha=ptalpha, edgecolors="none")
    lims = [-0.5, 1.0]

    ax.plot([thresh, thresh], [lims[0], thresh], color="gray")
    ax.plot([lims[0], thresh], [thresh, thresh], color="gray")

    good1 = corrs1 > thresh
    good2 = corrs2 > thresh
    ax.text(
        lims[0] + 0.05,
        thresh,
        "$n=%d$" % np.sum(good2),
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    ax.text(
        thresh,
        lims[0] + 0.05,
        "$n=%d$" % np.sum(good1),
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    ax.plot(lims, lims, "-", color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(name1 + " model")
    ax.set_ylabel(name2 + " model")

    fig.canvas.draw()
    show()
    ## Add over-under comparison
    # ax_left = ax.get_window_extent()._bbox.x0
    # ax_right = ax.get_window_extent()._bbox.x1
    # ax_width = ax_right-ax_left
    # print ax_left, ax_right
    # ax2 = fig.add_axes([ax_left, 0.1, ax_width, 0.2])
    ax2 = fig.add_axes([0.25, 0.1, 0.6, 0.25])  # , sharex=ax)
    # ax2 = fig.add_subplot(3, 1, 3)
    # plot_model_overunder_comparison(corrs1, corrs2, name1, name2, thresh=thresh, ax=ax2)
    plot_model_histogram_comparison(corrs1, corrs2, name1, name2, thresh=thresh, ax=ax2)

    fig.suptitle("Model comparison: %s vs. %s" % (name1, name2))
    show()
    return fig


def plot_model_overunder_comparison(corrs1, corrs2, name1, name2, thresh=0.35, ax=None):
    """Plots over-under difference between two models."""
    if ax is None:
        fig = figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

    maxcorr = max(corrs1.max(), corrs2.max())
    vals = np.linspace(0, maxcorr, 500)
    overunder = lambda c: np.array([np.sum(c > v) - np.sum(c < -v) for v in vals])

    ou1 = overunder(corrs1)
    ou2 = overunder(corrs2)

    oud = ou2 - ou1

    ax.fill_between(vals, 0, np.clip(oud, 0, 1e9), facecolor="blue")
    ax.fill_between(vals, 0, np.clip(oud, -1e9, 0), facecolor="red")

    yl = np.max(np.abs(np.array(ax.get_ylim())))
    ax.plot([thresh, thresh], [-yl, yl], "-", color="gray")
    ax.set_ylim(-yl, yl)
    ax.set_xlim(0, maxcorr)
    ax.set_xlabel("Voxel correlation")
    ax.set_ylabel("%s better           %s better" % (name1, name2))

    show()
    return ax


def plot_model_histogram_comparison(corrs1, corrs2, name1, name2, thresh=0.35, ax=None):
    """Plots over-under difference between two models."""
    if ax is None:
        fig = figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

    maxcorr = max(corrs1.max(), corrs2.max())
    nbins = 100
    hist1 = np.histogram(corrs1, nbins, range=(-1, 1))
    hist2 = np.histogram(corrs2, nbins, range=(-1, 1))

    ouhist1 = hist1[0][nbins / 2 :] - hist1[0][: nbins / 2][::-1]
    ouhist2 = hist2[0][nbins / 2 :] - hist2[0][: nbins / 2][::-1]

    oud = ouhist2 - ouhist1
    bwidth = 2.0 / nbins
    barlefts = hist1[1][nbins / 2 : -1]

    # ax.fill_between(vals, 0, np.clip(oud, 0, 1e9), facecolor="blue")
    # ax.fill_between(vals, 0, np.clip(oud, -1e9, 0), facecolor="red")

    ax.bar(barlefts, np.clip(oud, 0, 1e9), bwidth, facecolor="blue")
    ax.bar(barlefts, np.clip(oud, -1e9, 0), bwidth, facecolor="red")

    yl = np.max(np.abs(np.array(ax.get_ylim())))
    ax.plot([thresh, thresh], [-yl, yl], "-", color="gray")
    ax.set_ylim(-yl, yl)
    ax.set_xlim(0, maxcorr)
    ax.set_xlabel("Voxel correlation")
    ax.set_ylabel("%s better           %s better" % (name1, name2))

    show()
    return ax


def plot_model_comparison_rois(
    corrs1, corrs2, name1, name2, roivoxels, roinames, thresh=0.35
):
    """Plots model correlation comparisons per ROI."""
    fig = figure()
    ptalpha = 0.3

    for ri in range(len(roinames)):
        ax = fig.add_subplot(4, 4, ri + 1)
        ax.plot(corrs1[roivoxels[ri]], corrs2[roivoxels[ri]], "bo", alpha=ptalpha)
        lims = [-0.3, 1.0]
        ax.plot(lims, lims, "-", color="gray")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(roinames[ri])

    show()
    return fig


def center(mat, return_uncvals=False):
    """Centers the rows of [mat] by subtracting off the mean, but doesn't
    divide by the SD.
    Can be undone like zscore.
    """
    cmat = np.empty(mat.shape)
    uncvals = np.ones((mat.shape[0], 2))
    for ri in range(mat.shape[0]):
        uncvals[ri, 1] = np.mean(mat[ri, :])
        cmat[ri, :] = mat[ri, :] - uncvals[ri, 1]

    if return_uncvals:
        return cmat, uncvals

    return cmat


def ridge(A, b, alpha):
    """Performs ridge regression, estimating x in Ax=b with a regularization
    parameter of alpha.
    With $G=\alpha I(m_A)$, this function returns $W$ with:
    $W=(A^TA+G^TG)^{-1}A^Tb^T$
    Tantamount to minimizing $||Ax-b||+||\alpha I||$.
    """
    G = np.matrix(np.identity(A.shape[1]) * alpha)
    return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + np.dot(G.T, G)), A.T), b.T)


def model_voxels(Rstim, Pstim, Rresp, Presp, alpha):
    """Use ridge regression with regularization parameter [alpha] to model [Rresp]
    using [Rstim].  Correlation coefficients on the test set ([Presp] and [Pstim])
    will be returned for each voxel, as well as the linear weights.
    """
    Rresp[np.isnan(Rresp)] = 0.0
    Presp[np.isnan(Presp)] = 0.0

    print("Running ridge regression...")
    rwts = ridge(Rstim, Rresp.T, alpha)
    print("Finding correlations...")
    pred = np.dot(Pstim, rwts)
    prednorms = np.apply_along_axis(np.linalg.norm, 0, pred)
    respnorms = np.apply_along_axis(np.linalg.norm, 0, Presp)
    correlations = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze() / (
        prednorms * respnorms
    )

    print("Max correlation: %0.3f" % np.max(correlations))
    print("Skewness: %0.3f" % scipy.stats.skew(correlations))
    return np.array(correlations), rwts


def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  ## negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))

    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d * mtx.T).T
    else:
        return d * mtx


def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)

    for count, thing in enumerate(iterable):
        yield thing

        if not count % countevery:
            current_time = time.time()
            rate = float(count + 1) / (current_time - start_time)

            if rate > 1:  ## more than 1 item/second
                ratestr = "%0.2f items/second" % rate
            else:  ## less than 1 item/second
                ratestr = "%0.2f seconds/item" % (rate**-1)

            if total is not None:
                remitems = total - (count + 1)
                remtime = remitems / rate
                timestr = ", %s remaining" % time.strftime(
                    "%H:%M:%S", time.gmtime(remtime)
                )
                itemstr = "%d/%d" % (count + 1, total)
            else:
                timestr = ""
                itemstr = "%d" % (count + 1)

            formatted_str = "%s items complete (%s%s)" % (itemstr, ratestr, timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)
