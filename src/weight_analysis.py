import copy
import cottoncandy as cc
import numpy as np
import os
import pickle

from himalaya.kernel_ridge import primal_weights_weighted_kernel_ridge
from sklearn.decomposition import PCA

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    NOTE: Taken from AH regression code.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  # negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  # d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def load_data_matrices(data_parameters_dict):
    '''Load Y and X values, either pre-saved matrices of re-extracted.
    Parameters:
    -----------
    data_parameters_dict : dict
        Dictionary containing specs for how to load data.
        # TODO: Write docs for expected parameters and save formats.
    '''
    if data_parameters_dict.get('do_load_Y', True):
        if data_parameters_dict['Y_load_method'] == 'from_local':
            saved_Y_matrices = np.load(data_parameters_dict['Y_save_filepath'])
            Y_train, Y_test = saved_Y_matrices['Y_train'], saved_Y_matrices['Y_test']
        elif data_parameters_dict['Y_load_method'] == 'from_cc':
            cci = cc.get_interface(data_parameters_dict['cc_bucket_name_Y'])
            Y_train = np.nan_to_num(cci.download_raw_array(data_parameters_dict['Y_train_save_filepath']))
            Y_test = np.nan_to_num(cci.download_raw_array(data_parameters_dict['Y_test_save_filepath']))
        else:
            Y_train, Y_test = load_data_matrices_from_runs(data_parameters_dict, data_type='Y')
            if data_parameters_dict['do_save_Y']:
                np.savez(data_parameters_dict['Y_save_filepath'], Y_train=Y_train, Y_test=Y_test)
    else:
        Y_train = None
        Y_test = None

    if data_parameters_dict.get('do_load_Xs', True):
        delays = data_parameters_dict['delays']
        if data_parameters_dict['Xs_load_method'] == 'from_local':
            saved_Xs_matrices = np.load(data_parameters_dict['Xs_save_filepath'], allow_pickle=True)
            saved_Xs_matrices = {key: saved_Xs_matrices[key].item() for key in saved_Xs_matrices}
            Xs_train, Xs_test = saved_Xs_matrices['train_features'], saved_Xs_matrices['test_features']  # TODO current naming, change back to Xs_train/test
            feature_names = data_parameters_dict['features']['feature_names']
            Xs_train = [Xs_train[feature_name] for feature_name in feature_names]
            Xs_test = [Xs_test[feature_name][0] for feature_name in feature_names]  # NOTE: Maybe should not have [0].
            
            Xs_train = [make_delayed(X_train, delays) for X_train in Xs_train]
            if Xs_test is not None:
                Xs_test = [make_delayed(X_test, delays) for X_test in Xs_test]
        elif data_parameters_dict['Xs_load_method'] == 'from_cc':
            cci = cc.get_interface(data_parameters_dict['cc_bucket_name_X'])
            Xs_train = [np.nan_to_num(cci.download_raw_array(X_train_save_filepath)) for X_train_save_filepath in data_parameters_dict['Xs_train_save_filepaths']]
            if 'Xs_test_save_filepaths' in data_parameters_dict:
                Xs_test = [np.nan_to_num(cci.download_raw_array(X_test_save_filepath)) for X_test_save_filepath in data_parameters_dict['Xs_test_save_filepaths']]
            else:
                Xs_test = None
            Xs_train = [make_delayed(X_train, delays) for X_train in Xs_train]
            if Xs_test is not None:
                Xs_test = [make_delayed(X_test, delays) for X_test in Xs_test]
        else:
            Xs_train, Xs_test = load_data_matrices_from_runs(data_parameters_dict, data_type='Xs')
            if data_parameters_dict['do_save_Xs']:
                np.savez(data_parameters_dict['Xs_save_filepath'], Xs_train=Xs_train, Xs_test=Xs_test)

            Xs_train = [make_delayed(X_train, delays) for feature_name, X_train in Xs_train.items()]
            if Xs_test is not None:
                Xs_test = [make_delayed(X_test[0], delays) for feature_name, X_test in Xs_test.items()]
    else:
        Xs_train = None
        Xs_test = None

    return Y_train, Y_test, Xs_train, Xs_test

def scale_weights_by_score_sqrt(primal_weights, scores, normalize=True):
    '''Scale weights by sqrt of scores.
    Parameters:
    -----------
    TODO
    Returns:
    ---------
    TODO
    '''
    primal_weights = copy.deepcopy(primal_weights)
    if normalize:
        norm = np.linalg.norm(primal_weights, axis=0)
        primal_weights[:, norm != 0] /= norm[norm != 0]
    primal_weights *= np.sqrt(np.maximum(0, scores))
    return primal_weights

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
    undelayed_signal = np.ones((len(delays), num_signal_dims, num_voxels), dtype=signal.dtype)

    for delay_index, delay in enumerate(delays):
        begin, end = delay_index * num_signal_dims, (delay_index + 1) * num_signal_dims
        undelayed_signal[delay_index, :, :] = signal[begin:end]
    return undelayed_signal

def get_primal_weights(load_method, model_filepath, data_parameters_dict, cc_bucket_name=None, do_undelay_weights=True, do_mean_weights=True, primal_weights_save_filepath=None):
    '''Get primal weights from dual weights.
    Parameters:
    -----------
    load_method : str
        How to load primal weights. Options: [`primal_from_local`, `dual_from_local`, `dual_from_cc`].
    model_filepath : str
        Filepath where model is saved. Either on s3 or locally.
        If `load_method` is `dual_from_cc`, assumes that dual_weights and deltas are saved on s3 in `[model_filepath]/dual_weights` and `[model_filepath]/deltas`.
        If `load_method` is `dual_from_local`, assumes that dual_weights and deltas are saved locally in [model_filepath] npz file.
    cc_bucket_name : str
        Name of cottoncandy bucket, if load_method is `from_cc`.
    data_parameters_dict : dict
        See load_data_matrices() for details.
    do_undelay_weights : bool
        If True, undelays primal weights.
    do_mean_weights : bool
        If True, computes mean over delays.
    primal_weights_save_filepath : str
        Location for where to save primal weights after computing from dual weights.
        If None, does not save primal weights.
    Returns:
    --------
    primal_weights : list
        List of (num_dims * num_delays) x num_voxels matrices of primal weights.
    '''
    if load_method == 'primal_from_local':
        primal_weights = np.load(model_filepath)['primal_weights']
    else:
        if load_method == 'dual_from_local':
            model = np.load(model_filepath)
            dual_weights = model['dual_weights']
            deltas = model['deltas']
        elif load_method == 'dual_from_cc':
            cci = cc.get_interface(cc_bucket_name)
            dual_weights = cci.download_raw_array(os.path.join(model_filepath, 'dual_weights'))
            deltas = cci.download_raw_array(os.path.join(model_filepath, 'deltas'))
        else:
            raise ValueError(f'load_method {load_method} not supported')

        data_parameters_dict = copy.deepcopy(data_parameters_dict)
        data_parameters_dict['do_load_Y'] = False
        _, _, Xs_train, _ = load_data_matrices(data_parameters_dict)

        primal_weights = primal_weights_weighted_kernel_ridge(dual_weights, deltas, Xs_train)
        if primal_weights_save_filepath:
            np.savez(primal_weights_save_filepath, primal_weights=primal_weights)

    if do_undelay_weights:
        delays = data_parameters_dict['delays']
        primal_weights = [undelay_weights(feature_weights, delays) for feature_weights in primal_weights]  # num_features len list of num_delays x num_feature_dims x num_voxels matrices.
    if do_mean_weights:
        primal_weights = [feature_primal_weights.mean(0) for feature_primal_weights in primal_weights]
    return primal_weights


def compute_weight_correlation(primal_weights_a, primal_weights_b, weight_correlation_save_filepath, scores_a=None, scores_b=None):
    '''Compute correlation between two sets of weights.
    Parameters:
    -----------
    primal_weights_a : array_like
        num_dims x num_voxels matrix containing first set of model weights.
    primal_weights_b : array_like
        num_dims x num_voxels matrix containing second set of model weights.
    weight_correlation_save_filepath : str
        Filepath for where to save weight correlation matrices.
    scores_a : array_like
        num_voxels matrix of model scores corresponding to first set of weights. Potentially used for weight scaling.
    scores_b : array_like
        num_voxels matrix of model scores corresponding to second set of weights. Potentially used for weight scaling.
    '''
    primal_weights_a = np.nan_to_num(primal_weights_a)
    primal_weights_b = np.nan_to_num(primal_weights_b)
    np.savez(f'primal_weights', primal_weights_a=primal_weights_a, primal_weights_b=primal_weights_b, scores_a=scores_a, scores_b=scores_b)

    num_voxels = primal_weights_a.shape[1]
    voxel_weight_correlations = np.zeros(num_voxels)
    for voxel_index in range(num_voxels):
        voxel_weight_correlations[voxel_index] = np.corrcoef(primal_weights_a[:, voxel_index], primal_weights_b[:, voxel_index])[0, 1]

    if weight_correlation_save_filepath:
        np.savez(weight_correlation_save_filepath, weight_correlations=voxel_weight_correlations)


def get_pcs(primal_weights_matrix, n_components=10):
    '''Compute primal weights.
    Parameters:
    -----------
    primal_weights_matrix : array_like
        num_voxels x num_feature_dims matrix of primal weights.
    n_components : int
        The number of PCs to compute.
    Returns:
    --------
    explained_variance: array_like
        num_components array of explained variance for each PC
    components : array_like
        num_components x num_feature_dims matrix of PCs.
    '''
    pca = PCA(n_components=n_components)
    pca.fit(np.nan_to_num(primal_weights_matrix))

    explained_variance, components = pca.explained_variance_ratio_, pca.components_
    return explained_variance, components


def compute_weight_pcs(all_primal_weights_a, all_primal_weights_b, all_scores_a, all_scores_b, pc_computation_method, weight_scaling_method, pc_filepath):
    '''Perform PC analysis for weights.
    Parameters:
    -----------
    all_primal_weights_a : dict
        Dictionary of {participant: num_dims x num_voxels matrix containing first set of model weights}.
    all_primal_weights_b : list
        Dictionary of {participant: num_dims x num_voxels matrix containing second set of model weights}.
    all_scores_a : dict
        Dictionary of {participant: num_voxels matrix of model scores corresponding to first set of weights}.
    all_scores_b : dict
        Dictionary of {participant: num_voxels matrix of model scores corresponding to second set of weights}.
    pc_computation_method : str
        Denotes how to compute the PCs. Options: [`group_both_langs`, `individual_both_langs`]
    weight_scaling_method : str
        Denotes how to scale weights. Options: [`None`, `single_language`, `both_languages`]
    pc_filepath : str
        If provided, denotes where to save PCs.
        If None, then PCs are not saved.
    '''
    if weight_scaling_method == 'None':
        pass
    elif weight_scaling_method == 'single_language':
        all_primal_weights_a = {participant: scale_weights_by_score_sqrt(participant_weights, all_scores_a[participant]) for participant, participant_weights in all_primal_weights_a.items()}
        all_primal_weights_b = {participant: scale_weights_by_score_sqrt(participant_weights, all_scores_b[participant]) for participant, participant_weights in all_primal_weights_b.items()}
    elif weight_scaling_method == 'both_languages':
        all_primal_weights_a = {participant: scale_weights_by_score_sqrt(participant_weights, (all_scores_a[participant] + all_scores_b[participant]) / 2) for participant, participant_weights in all_primal_weights_a.items()}
        all_primal_weights_b = {participant: scale_weights_by_score_sqrt(participant_weights, (all_scores_a[participant] + all_scores_b[participant]) / 2) for participant, participant_weights in all_primal_weights_b.items()}
    else:
        raise ValueError(f'weight_scaling_method {weight_scaling_method} not supported.')
    if pc_computation_method == 'group_both_langs':
        primal_weights_matrix = np.concatenate([all_primal_weights_a[participant] for participant in all_primal_weights_a.keys()] +
                [all_primal_weights_b[participant] for participant in all_primal_weights_a.keys()], axis=1)
        pcs_dict = {'group': get_pcs(primal_weights_matrix.T)}
    elif pc_computation_method == 'individual_both_langs':
        pcs_dict = dict()
        for participant in all_primal_weights_a.keys():
            primal_weights_matrix = np.concatenate([all_primal_weights_a[participant], all_primal_weights_b[participant]], axis=1)
            pcs_dict[participant] = get_pcs(primal_weights_matrix.T)
    else:
        raise ValueError(f'pc_computation_method {pc_computation_method} not supported.')

    with open(pc_filepath, 'wb') as f:
        pickle.dump(pcs_dict, f)