import os

import pickle
import yaml

import time
import logging

import numpy as np
import pandas as pd

from scipy.stats import zscore

from himalaya.ridge import RidgeCV
from himalaya.kernel_ridge import KernelRidgeCV, KernelRidge

from sklearn.linear_model import Ridge
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

from src.config import (
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

    parser.add_argument("--subject", type=str, default="07")
    parser.add_argument("--timescale", type=str, default="2_4_words", choices=periods)
    parser.add_argument(
        "--feature_set_name", type=str, default="BERT_all", choices=features_sets
    )
    parser.add_argument("--kfold_splits", type=int, default=5)
    parser.add_argument("--alpha_min", type=float, default=1)
    parser.add_argument("--alpha_max", type=float, default=3)
    parser.add_argument("--alpha_num", type=int, default=10)

    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["numpy", "torch", "torch-cuda", "cupy"],
    )

    return parser


def fit_encoding_model(
    subject: str,
    timescale: str,
    feature_set_name: str,
    kfold_splits: int = 5,
    alpha_min: float = 1,
    alpha_max: float = 3,
    alpha_num: int = 10,
    backend: str = "numpy",
):
    # Himalaya backend
    backend = set_backend(backend, on_error="warn")

    # Loading feature set
    if feature_set_name == "BERT_all":
        feature_set = os.path.join(feature_sets_en_path, "timescales_BERT_all.npz")
    elif feature_set_name == "mBERT_all":
        feature_set = os.path.join(feature_sets_en_path, "timescales_mBERT_all.npz")

    feature = np.load(feature_set, allow_pickle=True)

    train_feature = feature["train"].tolist()
    test_feature = feature["test"].tolist()

    # Delaying feature set
    ndelays = 4
    delays = np.arange(1, ndelays + 1)

    # delaying all features
    delayed_train_feature = make_delayed(train_feature[timescale], delays=delays)
    delayed_test_feature = make_delayed(test_feature[timescale], delays=delays)

    delayed_train_feature = np.nan_to_num(delayed_train_feature)
    delayed_test_feature = np.nan_to_num(delayed_test_feature)

    # loading fmri data
    train_fn = f"subject{subject}_reading_fmri_data_trn.hdf"
    test_fn = f"subject{subject}_reading_fmri_data_val.hdf"

    training_data = load_dict(os.path.join(reading_data_en_path, train_fn))
    test_data = load_dict(os.path.join(reading_data_en_path, test_fn))

    ztraining_data = np.vstack(
        [
            zscore(
                training_data[story][
                    silence_length + noise_trim_length : -(noise_trim_length + silence_length),
                    :,
                ],
                axis=0,
            )
            for story in list(training_data.keys())
        ]
    )
    ztraining_data = np.nan_to_num(ztraining_data)

    ztest_data = zscore(
        np.mean(test_data["story_11"], axis=0)[
            silence_length + noise_trim_length : -(noise_trim_length + silence_length),
            :,
        ],
        axis=0,
    )
    ztest_data = np.nan_to_num(ztest_data)

    assert ztraining_data.shape[0] == delayed_train_feature.shape[0]
    assert ztest_data.shape[0] == delayed_test_feature.shape[0]

    # model fitting
    ## cast to float32
    delayed_train_feature = delayed_train_feature.astype(np.float32)
    delayed_test_feature = delayed_test_feature.astype(np.float32)

    ztraining_data = ztraining_data.astype(np.float32)
    ztest_data = ztest_data.astype(np.float32)

    ## Ridge regression
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=0)

    alphas = np.logspace(alpha_min, alpha_max, alpha_num)

    start = time.time()

    model = RidgeCV(
        alphas=alphas, cv=kfold.split(ztraining_data)
    )

    model.fit(delayed_train_feature, ztraining_data)

    print(f"Training took {time.time() - start} seconds")

    # saving best performing alpha
    best_alpha = model.best_alphas_

    # create ridge model with best alpha
    final_model = Ridge(alpha=best_alpha, fit_intercept=False)
    final_model.fit(delayed_train_feature, ztraining_data)

    # evaluate on test data
    test_score = final_model.score(delayed_test_feature, ztest_data)
    print(f"Test score: {test_score}")

    # save weights
    if os.path.exists(weights_save_path) == False:
        os.makedirs(weights_save_path)

    weight_name = f"{subject}-{timescale}-{feature_set_name}-weight.npz"
    weight_path = os.path.join(weights_save_path, weight_name)

    ## save in npz
    np.savez(weight_path, final_model.coef_.astype(np.float16))

    # save hyperparams in yaml file
    if os.path.exists(hyperparams_save_path) == False:
        os.makedirs(hyperparams_save_path)

    hyperparams_name = f"{subject}-{timescale}-{feature_set_name}-hyperparams.yaml"
    hyperparams_path = os.path.join(hyperparams_save_path, hyperparams_name)

    ## save best alphas array in yaml
    with open(hyperparams_path, "w") as f:
        yaml.dump({"best_alphas": best_alpha.tolist()}, f)

    # reg.alphas = list[reg.alphas]

    # # saving model through pickle
    # if os.path.exists(model_save_path) == False:
    #     os.makedirs(model_save_path)

    # model_name = f"subject{subject}_timescale_{timescale}_feature_set_{feature_set_name}_model.joblib"
    # model_path = os.path.join(model_save_path, model_name)

    # dump(reg, model_path)
    # # # model_path = os.path.join(model_save_path, model_name)

    # # with open(model_path, "wb") as f:
    # #     pickle.dump(reg, f)

    # # saving model weights
    # weights = reg.coef_
    # weights_name = f"subject{subject}_timescale_{timescale}_feature_set_{feature_set_name}_weights.csv"
    # weights_path = os.path.join(model_save_path, weights_name)

    # np.savetxt(weights_path, weights, delimiter=",")

    # # saving model intercept
    # intercept = reg.intercept_
    # intercept_name = f"subject{subject}_timescale_{timescale}_feature_set_{feature_set_name}_intercept.csv"
    # intercept_path = os.path.join(model_save_path, intercept_name)

    # np.savetxt(intercept_path, intercept, delimiter=",")


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
