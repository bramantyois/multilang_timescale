import os
import time
import logging
import multiprocessing
import json
import copy

import h5py

import copy
import io

from tqdm import tqdm
from itertools import product
from typing import Dict, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import scipy.linalg
import scipy.sparse
from scipy.signal import periodogram

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, ks_2samp

from ..configurations import timescales, timescale_ranges
from ..settings import ResultSetting


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

    mid_ranges = np.array([np.mean(timescale_ranges[key]) for key in timescales])
    mid_ranges = np.log2(mid_ranges)

    weighted_scores = np.stack(
        [
            normalized_scores[i,] * mid_ranges[i]
            for i in range(normalized_scores.shape[0])
        ]
    )

    return weighted_scores.sum(axis=0)


# permutation test
def single_test(
    blocks: List[np.ndarray],
    true_scores: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    score_func: callable,
    repeats: int,
    seed: int,
):
    np.random.seed(seed)
    num_get_true_score = np.zeros(true_scores.shape)
    for i in range(repeats):
        np.random.shuffle(blocks)
        permutation_order = np.concatenate(blocks)
        shuffled_pred = predictions[permutation_order]
        shuffled_scores = score_func(targets, shuffled_pred)
        num_get_true_score[shuffled_scores >= true_scores] += 1
    return num_get_true_score


def permutation_test_mp(
    targets: np.ndarray,
    predictions: np.ndarray,
    score_func: callable,
    num_permutations: int = 1000,
    permutation_block_size: int = 10,
    initial_seed: int = 0,
    num_processes: int = 10,
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
    initial_seed : int, optional
        Initial random seed, by default 0
    num_processes : int, optional
        Number of processes to use, by default 10

    Returns
    -------
    pvalues : np.ndarray
        p-values.
    true_scores : np.ndarray
        True scores.
    """
    true_scores = score_func(targets, predictions)

    num_TRs = targets.shape[0]
    blocks = np.array_split(np.arange(num_TRs), int(num_TRs / permutation_block_size))

    repeats = num_permutations // num_processes

    np.random.seed(initial_seed)
    seeds = np.random.randint(0, 1000000, num_processes)

    with multiprocessing.Pool(num_processes) as pool:
        params = list(
            product(
                [blocks] * num_processes,
                [true_scores],
                [predictions],
                [targets],
                [score_func],
                [repeats],
            )
        )
        for i, seed in enumerate(seeds):
            params[i] = params[i] + (seed,)
        num_get_true_scores = pool.starmap(single_test, params)

    num_get_true_score_sum = np.sum(num_get_true_scores, axis=0)
    p_values = num_get_true_score_sum / (repeats * num_processes)

    return p_values


def permutation_test(
    responses_test: np.ndarray,
    predictions: np.ndarray,
    score_func: callable,
    num_permutations: int = 1000,
    permutation_block_size: int = 10,
):
    true_scores = score_func(responses_test, predictions).detach().cpu().numpy()
    num_get_true_score = np.zeros(true_scores.shape)

    num_TRs = predictions.shape[0]
    blocks = np.array_split(np.arange(num_TRs), int(num_TRs / permutation_block_size))
    for permutation_num in tqdm(range(num_permutations)):
        _ = np.random.shuffle(blocks)
        permutation_order = np.concatenate(blocks)
        predictions = predictions[permutation_order]
        shuffled_scores = score_func(responses_test, predictions).detach().cpu().numpy()
        num_get_true_score[shuffled_scores >= true_scores] += 1
    pvalues = num_get_true_score / num_permutations

    return pvalues


def two_side_ks_test(
    data_1: np.ndarray, data_2: np.ndarray, alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Compute the KS test between two data.

    Parameters
    ----------
    data_1 : np.ndarray
        Data 1.
    data_2 : np.ndarray
        Data 2.

    Returns
    -------
    float
        KS test value.
    """
    _, ks_pval = ks_2samp(data_1, data_2)

    return ks_pval, ks_pval > alpha


def perm_func(data_1, data_2):
    return np.abs(np.mean(data_1) - np.mean(data_2))


def timescale_permutation_test(
    timescale_1: np.ndarray,
    timescale_2: np.ndarray,
    score_func: callable = perm_func,
    num_permutations: int = 1000,
    alpha: float = 0.05,
):
    """
    do permutation test on timescale data
    """
    true_scores = score_func(timescale_1, timescale_2)

    pooled_data = np.concatenate([timescale_1, timescale_2])

    num_get_true_score = 0

    for i in range(num_permutations):
        np.random.shuffle(pooled_data)
        shuffled_1 = pooled_data[: len(timescale_1)]
        shuffled_2 = pooled_data[len(timescale_1) :]

        shuffled_scores = score_func(shuffled_1, shuffled_2)

        num_get_true_score += shuffled_scores >= true_scores

    p_value = num_get_true_score / num_permutations

    return p_value, p_value > alpha


# P-Values correction
def get_bh_invalid_voxels(pvalues: np.ndarray, alpha: float):
    """
    Get invalid voxels using Benjamini-Hochberg procedure.

    Parameters
    ----------
    pvalues : np.ndarray
        p-values.
    alpha : float
        Alpha value.

    Returns
    -------
    np.ndarray
        Invalid voxels.
    """
    num_values = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    max_p = pvalues_sorted[
        np.argmax(
            np.where(
                pvalues_sorted <= ((np.arange(1, num_values + 1) / num_values) * alpha)
            )
        )
    ]
    return pvalues > max_p


def put_values_on_mask(
    value_to_be_stored: np.ndarray,
    p_values: np.ndarray,
    ev_mask: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    valid_range: Tuple[float, float] = (8, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Put voxels values of voxels given masks.

    Parameters
    ----------
    value_to_be_stored : np.ndarray
        Values to be stored.
    p_values : np.ndarray
        p-values.
    ev_mask : np.ndarray
        Mask.
    alpha : float, optional
        Alpha value, by default 0.05
    valid_range : Tuple[float, float], optional
        Valid range of value_to_be_stored, by default (8,256)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Whole voxel and valid voxels.

    """
    if ev_mask is None:
        ev_mask = np.ones(p_values.shape, dtype=bool)

    whole_voxel = np.full(p_values.shape, np.nan)

    invalid_p_values = get_bh_invalid_voxels(p_values, alpha)

    valid_values = (value_to_be_stored >= valid_range[0]) & (
        value_to_be_stored <= valid_range[1]
    )

    value_to_be_stored[~valid_values] = np.nan
    value_to_be_stored[invalid_p_values] = np.nan

    whole_voxel[ev_mask] = value_to_be_stored

    valid_voxels = np.where(~np.isnan(whole_voxel))

    return whole_voxel, valid_voxels


def get_valid_voxels(
    stats_dict: Dict,
    metric: Literal["r2", "r"] = "r2",
    alpha: Optional[float] = 0.05,
    score_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    get valid voxels from stats_dict.

    Parameters
    ----------
    stats_dict : Dict
        Dictionary containing stats.
    metric : Literal["r2", "r"]
        Metric to be used.
    alpha : float, optional
        Alpha value, by default 0.05
    score_threshold : float, optional
        Score threshold, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        binary mask and indices of valid voxels.
    """
    if score_threshold is None:
        p_val_keyword = f"test_p_values_{metric}_mask"
        p_values = stats_dict[p_val_keyword]

        whole_voxels = np.full(p_values.shape, True, dtype=bool)
        invalid_voxels = get_bh_invalid_voxels(p_values, alpha)

        whole_voxels[invalid_voxels] = False

        indices = np.where(whole_voxels)[0]

        return whole_voxels, indices
    else:
        score_keyword = f"test_joint_{metric}_score_mask"
        scores = stats_dict[score_keyword]

        if metric == "r2":
            scores = np.sqrt(np.maximum(scores, 0))

        valid_voxels = scores > score_threshold

        return valid_voxels, np.where(valid_voxels)[0]


def get_valid_result(
    stat_dict: Dict,
    result_key: str,
    metric: Literal["r", "r2"] = "r2",
    alpha: float = 0.05,
    replace_by_nan: bool = True,
):
    """
    Get valid results from stat_dict. invalid voxel determined by p-values will be filtered out/replace by np.nan.
    This function uses get_valid_voxel (permutation test) to determind the valid voxels.

    Parameters
    ----------
    stat_dict : Dict
        Dictionary containing stats.
    result_key : str
        Result key.
    metric : Literal["r", "r2"], optional
        Metric to be used, by default "r2"
    alpha : float, optional
        Alpha value for permutation test, by default 0.05
    replace_by_nan : bool, optional
        replace invalid voxels by np.nan. If False, return only valid voxels (smaller voxel number), by default True

    """

    values = stat_dict[result_key]

    mask, valid_indices = get_valid_voxels(stat_dict, metric=metric, alpha=alpha)

    if replace_by_nan:
        values[~mask] = np.nan
    else:
        values = values[valid_indices]

    return values


def get_joint_indices(
    p_values_1: np.ndarray,
    p_values_2: np.ndarray,
    alpha: float = 0.05,
):
    assert p_values_1.shape == p_values_2.shape

    indices = np.arange(len(p_values_1))

    invalid_p_values_1 = get_bh_invalid_voxels(p_values_1, alpha)
    invalid_p_values_2 = get_bh_invalid_voxels(p_values_2, alpha)

    valid_voxel_indices = indices[~(invalid_p_values_1 | invalid_p_values_2)]

    return valid_voxel_indices


# Response Cooking
def cook_responses(
    responses: Dict,
    test_runs: List[str],
    train_runs: List[str] = None,
    trim_start_length: int = 10,
    trim_end_length: int = 10,
    do_zscore: bool = True,
    do_mean_centering: bool = False,
    multiseries: str = "separate",
    resample_proportion: float = None,
    trim_exceptions_dict: Dict = None,
    trs_to_remove_dict: Dict = None,
):
    """
    args:
        responses: Dictionary of responses per run
        test_runs: run ids ([{textgrid_name}]) to use for test set.
        train_runs: run ids ([{textgrid_name}]) to use for train set.
        trim_start_length: Number of TRs to trim from start of features.
        trim_end_length: Number of TRs to trim from end of features.
        do_zscore: Whether to zscore results before returning them.
        do_mean_centering:
        multiseries :
        trim_exceptions_dict : {run_name: [start_trim, end_trim]}
        trs_to_remove_dict : {run_name: [trs_to_remove]} dictionary of trs to remove (e.g., because of large motion).
    returns:
        train_responses, test_responses: [num_trimmed_TRs x num_voxels] matrices of responses.
        resample_proportion: factor by which to resample data (original_tr / new_tr).
    """
    responses_by_run_name = copy.deepcopy(responses)

    assert trim_end_length >= 0
    # Does not work with Python negative indexing
    # if trim_end_length == 0:
    #     trim_end_length = -np.inf

    if train_runs is None:
        # logger.info('train_runs not specified, will be taken from responses.keys()')
        train_runs = [
            run for run in responses_by_run_name.keys() if run not in test_runs
        ]

    assert len(set(train_runs).intersection(test_runs)) == 0

    # logger.info(f'Responses will be returned using {multiseries}')
    for run_name in train_runs + test_runs:
        response = responses_by_run_name[run_name]
        # Handle repetitions of a run
        if multiseries == "mean":
            response = np.sum(response, axis=0) / len(response)
        elif multiseries == "concat":
            response = np.vstack(response)
        elif multiseries == "separate":
            response = np.squeeze(response)
        elif multiseries == "average_across":
            if len(np.shape(response)) > 2:
                response = np.mean(response, axis=0)

        # Remove TRs if applicable.
        if trs_to_remove_dict is not None:
            if run_name in trs_to_remove_dict:
                # logger.info(f'Removing {len(trs_to_remove_dict[run_name])} TRs from {run_name} for responses')
                response = np.delete(response, trs_to_remove_dict[run_name], axis=0)

        # Trim and zscore each run separately
        trim_start = trim_start_length
        trim_end = trim_end_length
        if trim_exceptions_dict is not None:
            if run_name in list(trim_exceptions_dict.keys()):
                trim_start, trim_end = trim_exceptions_dict[run_name]
        # logger.info(f'Trim {run_name} with [{trim_start}:-{trim_end}]')

        # If response has more than one repetition as in "multiseries == separate"
        if len(np.shape(response)) > 2:
            response = np.array([res[trim_start:-trim_end] for res in response])
        else:
            if trim_end == 0:
                response = np.array(response[trim_start:])
            else:
                response = np.array(response[trim_start:-trim_end])

        if do_zscore:
            # If response has more than one repetition as in "multiseries == separate"
            if len(np.shape(response)) > 2:
                response = np.array([zscore(res) for res in response])
            else:
                response = np.array(zscore(response))

        responses_by_run_name[run_name] = response

    if resample_proportion:
        responses_by_run_name = {
            key: resample(response, round(response.shape[0] * resample_proportion))
            for key, response in responses_by_run_name.items()
        }

    train_responses = np.concatenate(
        [responses_by_run_name[run] for run in train_runs], axis=0
    )
    # logger.info(f'Train runs: {train_runs}')
    # logger.info(f'Train responses: {np.shape(train_responses)}')

    # logger.info(f'Test runs: {test_runs}')
    test_responses = [responses_by_run_name[run] for run in test_runs]
    test_sizes = [np.shape(test_response) for test_response in test_responses]
    # logger.info(f'Test responses (Each test run is an entry in the list): {test_sizes}')

    return train_responses, test_responses


def get_surface_dict(
    subject_id: str,
    surfaces_json: str = ".temp/fmri/bling/surfaces.json",
    description: str = "default",
):
    """
    get dictinary of surface for a given subject_id and description
    """
    surfaces_dict = json.load(open(surfaces_json, "r"))

    subject_surface = surfaces_dict[subject_id]
    sel_subject_surface = None
    for s in subject_surface:
        if s["description"] == description:
            sel_subject_surface = s
            break
    if not sel_subject_surface:
        raise ValueError(f"Description {description} not found in surfaces_dict")

    return {
        "transform": sel_subject_surface["transform"],
        "surface": sel_subject_surface["surface"],
    }


"""

Below are codes taken from bling_repositories

"""


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
