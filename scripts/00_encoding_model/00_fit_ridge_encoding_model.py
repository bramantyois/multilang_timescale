import os

import pickle
import json

import time
import logging

import warnings

import numpy as np
import pandas as pd

from scipy.stats import zscore

from himalaya.ridge import RidgeCV, Ridge
from himalaya.kernel_ridge import KernelRidgeCV, KernelRidge

# from sklearn.linear_model import Ridge
from himalaya.backend import set_backend

from sklearn.model_selection import KFold

from joblib import dump

# adding src to path
import sys

sys.path.insert(0, os.getcwd())

from src.vm_tutorial_sklearn.util import make_delayed, load_dict
from src.vm_tutorial_sklearn.hard_coded_things import (
    test_stories,
    train_stories,
    silence_length,
    noise_trim_length,
    periods,
    features_sets,
)

from src.configurations import (
    feature_sets_en_path,
    reading_data_en_path,
    model_save_path,
    weights_save_path,
    hyperparams_save_path,
    stats_save_path,
)

from argparse import ArgumentParser


logging.basicConfig(level=logging.DEBUG)


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--sub_id", type=str, default="07", help="subject id")
    parser.add_argument("--sub_fmri_train_path", type=str, help="path to train data")
    parser.add_argument("--sub_fmri_test_path", type=str, help="path to test data")
    parser.add_argument("--sub_trim_start", type=int, default=10)
    parser.add_argument("--sub_trim_end", type=int, default=10)

    parser.add_argument("--train_stories", type=str, nargs="+", default=train_stories)
    parser.add_argument("--test_stories", type=str, nargs="+", default=test_stories)


    parser.add_argument("--timescale", type=str, default="2_4_words", choices=periods)
    parser.add_argument(
        "--lm_feature_path", type=str, required=True,)
    parser.add_argument(
        "--sensory_level_feature_path", type=str,required=False,default=None)

    parser.add_argument("--kfold_splits", type=int, default=5)
    parser.add_argument("--alpha_min", type=float, default=1)
    parser.add_argument("--alpha_max", type=float, default=3)
    parser.add_argument("--alpha_num", type=int, default=10)

    parser.add_argument("--feature_delay", type=int, default=4)

    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["numpy", "torch", "torch-cuda", "cupy"],
    )

    parser.add_argument("--weights_save_path", type=str, default=weights_save_path)
    parser.add_argument("--hyperparams_save_path", type=str, default=hyperparams_save_path)

    parser.add_argument("--config_path", type=str, required=False, default=None)

    return parser


def fit_encoding_model(
    sub_fmri_train_path: str,
    sub_fmri_test_path: str,
    sub_trim_start: int,
    sub_trim_end: int,
    train_stories: List[str],
    test_stories: List[str],
    timescale: str,
    lm_feature_path: str,
    sensory_level_feature_path: str,
    kfold_splits: int = 5,
    alpha_min: float = 1,
    alpha_max: float = 3,
    alpha_num: int = 10,
    feature_delay: int = 4,
    backend: str = "numpy",
    weights_save_path: str = weights_save_path,
    hyperparams_save_path: str = hyperparams_save_path,
):
    # backend
    backend = set_backend(backend, on_error="warn")

    # Loading feature set
    ## loading lm-derived feature set
    # if feature_set_name == "BERT_all":
    #     feature_set = os.path.join(feature_sets_en_path, "timescales_BERT_all.npz")
    # elif feature_set_name == "mBERT_all":
    #     feature_set = os.path.join(feature_sets_en_path, "timescales_mBERT_all.npz")

    lm_feature = np.load(lm_feature_path, allow_pickle=True)

    lm_train_feature = feature["train"].tolist()
    lm_test_feature = feature["test"].tolist()

    # TODO: join sensory level feature set here

    # Delaying feature set
    train_feature = lm_train_feature
    test_feature = lm_test_feature
    
    delays = np.arange(1, feature_delay + 1)

    # delaying all features
    train_feature = make_delayed(train_feature[timescale], delays=delays)
    test_feature = make_delayed(test_feature[timescale], delays=delays)

    train_feature = np.nan_to_num(train_feature)
    test_feature = np.nan_to_num(test_feature)

    # loading fmri data
    train_data = load_dict(sub_fmri_train_path)
    test_data = load_dict(sub_fmri_test_path)

    train_data = np.vstack(
        [
            zscore(
                train_data[story][
                    sub_trim_start : -sub_trim_end,
                    :,
                ],
                axis=0,
            )
            for story in train_stories
        ]
    )
    train_data = np.nan_to_num(train_data)

    test_data = zscore(
        np.mean(test_data[test_stories[0]], axis=0)[
            sub_trim_start : -sub_trim_end,
            :,
        ],
        axis=0,
    )
    test_data = np.nan_to_num(test_data)

    assert train_data.shape[0] == train_feature.shape[0]
    assert test_data.shape[0] == test_feature.shape[0]

    # model fitting
    ## cast to float32
    train_feature = train_feature.astype(np.float32)
    test_feature = test_feature.astype(np.float32)

    training_data = training_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    ## Ridge regression
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=0)

    alphas = np.logspace(alpha_min, alpha_max, alpha_num)

    start = time.time()

    model = RidgeCV(
        alphas=alphas, cv=kfold.split(training_data)
    )

    model.fit(train_feature, training_data)

    print(f"Training took {time.time() - start} seconds")

    # saving best performing alpha
    best_alpha = model.best_alphas_

    # create ridge model with best alpha
    final_model = Ridge(alpha=best_alpha, fit_intercept=False)
    final_model.fit(train_feature, train_data)

    # evaluate on test data
    test_score = final_model.score(test_feature, test_data)
    print(f"Test score: {test_score}")

    # save weights if path is available
    if os.path.exists(weights_save_path) == True:
        print("not saving. path already exists")
    else:    
        np.savez(weights_save_path, final_model.coef_.astype(np.float16))

    if os.path.exists(hyperparams_save_path) == True:
        print("not saving hyperparamters file. Path already exists")
    else:
       np.savez(hyperparams_save_path, best_alpha.astype(np.float16))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    fit_encoding_model(
        args.subject,
        args.timescale,
        args.feature_set_name,
        args.kfold_splits,
        args.alpha_min,
        args.alpha_max,
        args.alpha_num,
    )
