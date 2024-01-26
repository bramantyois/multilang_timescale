import os
import time
import uuid
import json

from typing import Optional

import numpy as np
from scipy.stats import zscore

import torch

from himalaya.backend import set_backend
from himalaya.kernel_ridge import (
    Kernelizer,
    ColumnKernelizer,
    MultipleKernelRidgeCV,
    WeightedKernelRidge,
)
from himalaya.scoring import r2_score_split, correlation_score_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config

from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.utils import explainable_variance
from voxelwise_tutorials.viz import (
    plot_2d_flatmap_from_mapper,
    plot_flatmap_from_mapper,
)

import matplotlib.pyplot as plt

from src.utils import load_dict
from src.settings import TrainerConfig, SubjectConfig, FeatureConfig, ResultConfig


class Trainer:
    def __init__(
        self,
        sub_config_json: Optional[str] = None,
        feature_config_json: Optional[str] = None,
        trainer_config_json: Optional[str] = None,
        result_config_json: str = None,
    ):
        if result_config_json is not None:
            temp = self._load_json(result_config_json)
            self.result_config = ResultConfig(**temp)

            sub_config_json = self.result_config.subject_config_path
            feature_config_json = self.result_config.feature_config_path
            trainer_config_json = self.result_config.trainer_config_path

        temp = self._load_json(sub_config_json)
        self.sub_config = SubjectConfig(**temp)

        temp = self._load_json(feature_config_json)
        self.feature_config = FeatureConfig(**temp)

        temp = self._load_json(trainer_config_json)
        self.trainer_config = TrainerConfig(**temp)

        self._generate_output_config(
            sub_config_json, feature_config_json, trainer_config_json
        )

        self._prepare_data()
        self._prepare_features()

        set_config(assume_finite=True)

    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def _generate_output_config(
        self, sub_config_json: str, feature_config_json: str, trainer_config_json: str
    ):
        id = str(uuid.uuid4())

        result_dir = os.path.join(self.trainer_config.result_save_dir, id)

        result_config = ResultConfig()
        result_config.subject_config_path = sub_config_json
        result_config.feature_config_path = feature_config_json
        result_config.trainer_config_path = trainer_config_json

        result_config.result_dir = result_dir
        result_config.hyperparam_path = os.path.join(result_dir, "hyperparams.npz")
        result_config.stats_path = os.path.join(result_dir, "stats.npz")
        result_config.plot_dir = os.path.join(result_dir, "plots")

        # creating dirs
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(result_config.plot_dir):
            os.makedirs(result_config.plot_dir)

        # to json
        if not os.path.exists(self.trainer_config.result_meta_save_dir):
            os.makedirs(self.trainer_config.result_meta_save_dir)
        result_config_json = os.path.join(
            self.trainer_config.result_meta_save_dir, f"{id}.json"
        )

        with open(result_config_json, "w") as f:
            json.dump(result_config.__dict__, f, indent=4)

        self.result_config = result_config

    def _prepare_data(self):
        train_data = load_dict(self.sub_config.sub_fmri_train_path)
        test_data = load_dict(self.sub_config.sub_fmri_test_path)

        # zscore data
        ## zscore data
        train_data = np.vstack(
            [
                zscore(
                    train_data[story][
                        self.sub_config.sub_trim_start : -self.sub_config.sub_trim_end,
                        :,
                    ],
                    axis=0,
                )
                for story in list(train_data.keys())
            ]
        )
        self.train_data = np.nan_to_num(train_data)

        # computing ev before masking
        ev = explainable_variance(test_data["story_11"])
        self.mask = ev > self.sub_config.ev_threshold

        test_data = zscore(
            np.mean(test_data["story_11"], axis=0)[
                self.sub_config.sub_trim_start : -self.sub_config.sub_trim_end, :
            ],
            axis=0,
        )
        self.test_data = np.nan_to_num(test_data)

    def _prepare_features(self):
        train_features = []
        test_features = []

        # lm-derived feature
        ## train
        lm_feature_train_test = np.load(
            self.feature_config.lm_feature_path, allow_pickle=True
        )

        for t in self.feature_config.timescale:
            lm_features = lm_feature_train_test["train"].tolist()[t]
            train_features.append(
                {
                    "name": f"lm_{t}",
                    "size": lm_features.shape[1],
                    "feature": np.nan_to_num(lm_features),
                }
            )
            ## test
            lm_features = lm_feature_train_test["test"].tolist()[t]
            test_features.append(
                {
                    "name": f"lm_{t}",
                    "size": lm_features.shape[1],
                    "feature": np.nan_to_num(lm_features),
                }
            )

        # sensory-level features
        ## train
        if self.feature_config.sensory_feature_train_paths is not None:
            sensory_level_train_feature = load_dict(
                self.feature_config.sensory_feature_train_paths
            )

            for feature in self.feature_config.sensory_features:
                story_stacked = []
                for story in list(sensory_level_train_feature.keys()):
                    story_stacked.append(
                        sensory_level_train_feature[story][feature][
                            self.feature_config.sensory_feature_trim_start : -self.feature_config.sensory_feature_trim_end
                        ]
                    )
                story_stacked = np.vstack(story_stacked)
                train_features.append(
                    {
                        "name": feature,
                        "size": story_stacked.shape[1],
                        "feature": np.nan_to_num(story_stacked),
                    }
                )
        ## test
        if self.feature_config.sensory_feature_test_paths is not None:
            sensory_level_test_feature = load_dict(
                self.feature_config.sensory_feature_test_paths
            )

            for feature in self.feature_config.sensory_features:
                story_stacked = []
                for story in list(sensory_level_test_feature.keys()):
                    story_stacked.append(
                        sensory_level_test_feature[story][feature][
                            self.feature_config.sensory_feature_trim_start : -self.feature_config.sensory_feature_trim_end
                        ]
                    )
                story_stacked = np.vstack(story_stacked)
                test_features.append(
                    {
                        "name": feature,
                        "size": story_stacked.shape[1],
                        "feature": np.nan_to_num(story_stacked),
                    }
                )

        # motion-energy features
        if self.feature_config.motion_energy_feature_paths is not None:
            moten = np.load(
                self.feature_config.motion_energy_feature_paths, allow_pickle=True
            )
            ## train
            moten_train = moten["train"].tolist()

            for f in self.feature_config.motion_energy_features:
                train_features.append(
                    {
                        "name": f"motion energy : {f}",
                        "size": moten_train[f].shape[1],
                        "feature": np.nan_to_num(moten_train[f]),
                    }
                )

            ## test
            moten_test = moten["test"].tolist()

            for f in self.feature_config.motion_energy_features:
                test_features.append(
                    {
                        "name": f"motion energy : {f}",
                        "size": moten_test[f].shape[1],
                        "feature": np.nan_to_num(moten_test[f]),
                    }
                )

        # join features
        self.train_feature = np.hstack([f["feature"] for f in train_features])
        self.test_feature = np.hstack([f["feature"] for f in test_features])

        # delays = np.arange(1, self.trainer_config.feature_delay + 1)
        # self.train_feature = make_delayed(train_feature, delays)

        assert (
            self.train_feature.shape[0] == self.train_data.shape[0]
        ), "Feature and data shape mismatch"

        assert (
            self.test_feature.shape[0] == self.test_data.shape[0]
        ), "Feature and data shape mismatch"

        # remove 'feature' key, save memory
        for f in train_features:
            del f["feature"]

        for f in test_features:
            del f["feature"]

        self.train_feature_info = train_features
        self.test_feature_info = test_features

    def get_kernelizer(self):
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            Delayer(delays=[1, 2, 3, 4]),
            Kernelizer(kernel="linear"),
        )

        n_feature_list = [f["size"] for f in self.train_feature_info]
        start_and_end = np.concatenate([[0], np.cumsum(n_feature_list)])
        slices = [
            slice(start, end)
            for start, end in zip(start_and_end[:-1], start_and_end[1:])
        ]

        kernelizers_tuples = [
            (name, preprocess_pipeline, slice_)
            for name, slice_ in zip(
                [f["name"] for f in self.train_feature_info],
                slices,
            )
        ]
        return ColumnKernelizer(kernelizers_tuples)

    def prepare_training_pipeline(self):
        # preprocess
        columnn_kernelizer = self.get_kernelizer()
        # model
        solver_params = dict(
            n_iter=self.trainer_config.n_iter,
            alphas=np.logspace(
                self.trainer_config.alpha_min,
                self.trainer_config.alpha_max,
                self.trainer_config.alpha_num,
            ),
            n_targets_batch=self.trainer_config.n_targets_batch,
            n_alphas_batch=self.trainer_config.n_alphas_batch,
            n_targets_batch_refit=self.trainer_config.n_targets_batch_refit,
        )

        mkr_model = MultipleKernelRidgeCV(
            kernels="precomputed",
            solver=self.trainer_config.solver,
            solver_params=solver_params,
            cv=self.trainer_config.kfolds,
        )

        return make_pipeline(columnn_kernelizer, mkr_model, verbose=False)

    def train(self, force_cpu: bool = False):
        if force_cpu:
            set_backend("numpy", on_error="warn")
        else:
            set_backend(self.trainer_config.backend, on_error="warn")

        pipeline = self.prepare_training_pipeline()

        # casting
        train_feature = self.train_feature.astype("float32")
        train_data = self.train_data.astype("float32")

        if self.trainer_config.fit_on_mask:
            train_data = train_data[:, self.mask]
            # print("using mask size of {train_data.shape}")

        # Fitting
        print("Fitting model...")
        start = time.time()

        pipeline.fit(train_feature, train_data)

        print(f"training took {time.time() - start} seconds")

        # save hyperparams
        deltas = pipeline[-1].deltas_.cpu().numpy()
        best_alphas = pipeline[-1].best_alphas_.cpu().numpy()
        np.savez(
            self.result_config.hyperparam_path, deltas=deltas, best_alphas=best_alphas
        )

        # clear cuda memory
        if self.trainer_config.backend == "torch_cuda":
            del pipeline
            torch.cuda.empty_cache()

    def refit_and_evaluate(self, force_cpu: bool = False):
        if force_cpu:
            backend = set_backend("numpy", on_error="warn")
        else:
            backend = set_backend(self.trainer_config.backend, on_error="warn")

        # load hyperparams
        hyperparams_fn = os.path.join(
            self.result_config.result_dir,
            f"hyperparams.npz",
        )

        hyperparams = np.load(hyperparams_fn)
        deltas = (
            hyperparams["deltas"].astype("float32")
            if self.trainer_config.use_fitted_deltas
            else "zeros"
        )
        best_alphas = (
            hyperparams["best_alphas"].astype("float32")
            if self.trainer_config.use_fitted_alphas
            else 1
        )

        # load kernelizer
        columnn_kernelizer = self.get_kernelizer()

        model = WeightedKernelRidge(
            alpha=best_alphas,
            deltas=deltas,
            kernels="precomputed",
            solver="conjugate_gradient",
            solver_params={
                "n_targets_batch": self.trainer_config.n_targets_batch_refit
            },
        )

        pipeline = make_pipeline(columnn_kernelizer, model, verbose=False)

        # casting
        train_feature = self.train_feature.astype("float32")
        train_data = self.train_data.astype("float32")

        test_feature = self.test_feature.astype("float32")
        test_data = self.test_data.astype("float32")

        if self.trainer_config.fit_on_mask:
            train_data = train_data[:, self.mask]
            test_data = test_data[:, self.mask]
            # print("using mask size of {train_data.shape}")

        # prepare features
        pipeline.fit(train_feature, train_data)

        # score on train
        ## predict in batches
        def predict_in_batches(
            model, X, batch_size=self.trainer_config.n_targets_batch_refit
        ):
            n_samples = X.shape[0]
            n_batches = int(np.ceil(n_samples / batch_size))
            y_pred = []
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = (batch_idx + 1) * batch_size
                y_pred_batch = model.predict(X[start:end], split=True)
                y_pred_batch = backend.to_numpy(y_pred_batch)
                y_pred.append(y_pred_batch)
            y_pred = np.concatenate(y_pred, axis=1)
            return y_pred

        train_pred_split = predict_in_batches(pipeline, train_feature)
        test_pred_split = predict_in_batches(pipeline, test_feature)

        # now do it in cpu
        backend = set_backend("numpy", on_error="warn")

        train_r2_score_mask = r2_score_split(train_data, train_pred_split)
        train_r_score_mask = correlation_score_split(train_data, train_pred_split)

        # score on test
        test_r2_score_mask = r2_score_split(test_data, test_pred_split)
        test_r_score_mask = correlation_score_split(test_data, test_pred_split)

        if self.trainer_config.fit_on_mask:
            n_kernels = train_r2_score_mask.shape[0]
            n_voxels = self.test_data.shape[1]

            train_r2_split_scores = np.zeros((n_kernels, n_voxels))
            train_r_split_scores = np.zeros((n_kernels, n_voxels))

            test_r2_split_scores = np.zeros((n_kernels, n_voxels))
            test_r_split_scores = np.zeros((n_kernels, n_voxels))

            train_r2_split_scores[:, self.mask] = backend.to_numpy(train_r2_score_mask)
            train_r_split_scores[:, self.mask] = backend.to_numpy(train_r_score_mask)

            test_r2_split_scores[:, self.mask] = backend.to_numpy(test_r2_score_mask)
            test_r_split_scores[:, self.mask] = backend.to_numpy(test_r_score_mask)
        else:
            train_r2_split_scores = train_r2_score_mask
            train_r_split_scores = train_r_score_mask
            test_r2_split_scores = test_r2_score_mask
            test_r_split_scores = test_r_score_mask

        # saving stat
        np.savez_compressed(
            self.result_config.stats_path,
            train_r2_split_scores=train_r2_split_scores,
            train_r_split_scores=train_r_split_scores,
            test_r2_split_scores=test_r2_split_scores,
            test_r_split_scores=test_r_split_scores,
        )

        # clear cuda memory
        if self.trainer_config.backend == "torch_cuda":
            del pipeline
            torch.cuda.empty_cache()

    def plot(
        self,
        feature_index: int = 0,
        is_corr: bool = False,
        is_train: bool = False,
    ):
        # load statfile
        stat = np.load(self.result_config.stats_path)

        data_mode = "test"
        if is_train:
            data_mode = "train"

        score_mode = "r2"
        if is_corr:
            score_mode = "r"

        scores = stat[f"{data_mode}_{score_mode}_split_scores"]

        fig, ax = plt.subplots(figsize=(10, 10))

        plot_flatmap_from_mapper(
            voxels=scores[feature_index],
            mapper_file=self.sub_config.sub_fmri_mapper_path,
            vmin=0,
            vmax=0.5,
            ax=ax,
        )

        plt.show()

    def plot2d(
        self,
        feature_indices: list = [0, 1],
        is_corr: bool = False,
        is_train: bool = False,
    ):
        # load statfile
        stat_fn = os.path.join(
            self.result_config.stat_dir,
            f"{self.sub_config.sub_id}-{self.sub_config.task}-{self.feature_config.timescale}.npz",
        )

        stat = np.load(stat_fn)

        data_mode = "test"
        if is_train:
            data_mode = "train"

        score_mode = "r2"
        if is_corr:
            score_mode = "r"

        scores = stat[f"{data_mode}_{score_mode}_split_scores"]

        fig, ax = plt.subplots(figsize=(10, 10))

        plot_2d_flatmap_from_mapper(
            voxels_1=scores[feature_indices[0]],
            voxels_2=scores[feature_indices[1]],
            mapper_file=self.sub_config.sub_fmri_mapper_path,
            vmin=0,
            vmax=0.5,
            vmin2=0,
            vmax2=0.5,
            label_1=self.train_feature_info[feature_indices[0]]["name"],
            label_2=self.train_feature_info[feature_indices[1]]["name"],
            ax=ax,
        )
        plt.show()
