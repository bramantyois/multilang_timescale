import os

import time

import numpy as np
from scipy.stats import zscore

from himalaya.backend import set_backend
from himalaya.kernel_ridge import (
    Kernelizer,
    ColumnKernelizer,
    MultipleKernelRidgeCV,
    WeightedKernelRidge
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
from src.settings import TrainerConfig, SubjectConfig, FeatureConfig


class Trainer:
    def __init__(self, sub_config: SubjectConfig, feature_config: FeatureConfig):
        self.sub_config = sub_config
        self.feature_config = feature_config

        self.prepare_data()
        self.prepare_features()
        
        set_config(assume_finite=True)

    def prepare_data(self):
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
        self.mask = ev > 0.1

        test_data = zscore(
            np.mean(test_data["story_11"], axis=0)[
                self.sub_config.sub_trim_start : -self.sub_config.sub_trim_end, :
            ],
            axis=0,
        )
        self.test_data = np.nan_to_num(test_data)

    def prepare_features(self):
        train_features = []
        test_features = []

        # lm-derived feature
        ## train
        lm_feature_train_test = np.load(
            self.feature_config.lm_feature_path, allow_pickle=True
        )
        lm_features = lm_feature_train_test["train"].tolist()[
            self.feature_config.timescale
        ]
        train_features.append(
            {
                "name": "lm",
                "size": lm_features.shape[1],
                "feature": np.nan_to_num(lm_features),
            }
        )
        ### test
        lm_features = lm_feature_train_test["test"].tolist()[
            self.feature_config.timescale
        ]
        test_features.append(
            {
                "name": "lm",
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

        # delays = np.arange(1, trainer_config.feature_delay + 1)
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

    def prepare_training_pipeline(self, trainer_config: TrainerConfig):
        # preprocess
        columnn_kernelizer = self.get_kernelizer()
        # model
        solver_params = dict(
            n_iter=trainer_config.n_iter,
            alphas=np.logspace(
                trainer_config.alpha_min,
                trainer_config.alpha_max,
                trainer_config.alpha_num,
            ),
            n_targets_batch=trainer_config.n_targets_batch,
            n_alphas_batch=trainer_config.n_alphas_batch,
            n_targets_batch_refit=trainer_config.n_targets_batch_refit,
        )

        mkr_model = MultipleKernelRidgeCV(
            kernels="precomputed",
            solver=trainer_config.solver,
            solver_params=solver_params,
            cv=trainer_config.kfolds,
        )

        return make_pipeline(columnn_kernelizer, mkr_model, verbose=False)

    def train(self, trainer_config: TrainerConfig, force_cpu: bool = False):
        if force_cpu:
            set_backend("numpy", on_error="warn")
        else:
            set_backend(trainer_config.backend, on_error="warn")

        pipeline = self.prepare_training_pipeline(trainer_config)

        # casting
        train_feature = self.train_feature.astype("float32")
        train_data = self.train_data.astype("float32")

        if trainer_config.fit_on_mask:
            train_data = train_data[:, self.mask]
            # print("using mask size of {train_data.shape}")

        # Fitting
        print("Fitting model...")
        start = time.time()

        pipeline.fit(train_feature, train_data)

        print(f"training took {time.time() - start} seconds")

        if not os.path.exists(trainer_config.hyperparams_save_dir):
            os.makedirs(trainer_config.hyperparams_save_dir)

        hyperparams_fn = os.path.join(
            trainer_config.hyperparams_save_dir,
            f"{self.sub_config.sub_id}-{self.sub_config.task}-{self.feature_config.timescale}.npz",
        )

        # save hyperparams
        deltas = pipeline[-1].deltas_.cpu().numpy()
        best_alphas = pipeline[-1].best_alphas_.cpu().numpy()
        np.savez(hyperparams_fn, deltas=deltas, best_alphas=best_alphas)

    def refit_and_evaluate(self, trainer_config: TrainerConfig, force_cpu: bool = False):
        if force_cpu:
            backend = set_backend("numpy", on_error="warn")
        else:
            backend = set_backend(trainer_config.backend, on_error="warn")

        # load hyperparams
        hyperparams_fn = os.path.join(
            trainer_config.hyperparams_save_dir,
            f"{self.sub_config.sub_id}-{self.sub_config.task}-{self.feature_config.timescale}.npz",
        )

        hyperparams = np.load(hyperparams_fn)
        deltas = (
            hyperparams["deltas"].astype("float32")
            if trainer_config.use_fitted_deltas
            else "zeros"
        )
        best_alphas = (
            hyperparams["best_alphas"].astype("float32")
            if trainer_config.use_fitted_alphas
            else 1
        )

        # load kernelizer
        columnn_kernelizer = self.get_kernelizer()

        model = WeightedKernelRidge(
            alpha=best_alphas,
            deltas=deltas,
            kernels="precomputed",
            solver="conjugate_gradient",
            solver_params={'n_targets_batch':trainer_config.n_targets_batch_refit},
        )

        pipeline = make_pipeline(columnn_kernelizer, model, verbose=False)

        # casting
        train_feature = self.train_feature.astype("float32")
        train_data = self.train_data.astype("float32")

        test_feature = self.test_feature.astype("float32")
        test_data = self.test_data.astype("float32")

        if trainer_config.fit_on_mask:
            train_data = train_data[:, self.mask]
            test_data = test_data[:, self.mask]
            # print("using mask size of {train_data.shape}")

        # prepare features
        pipeline.fit(train_feature, train_data)

        # score on train
        train_pred_split = pipeline.predict(train_feature, split=True)
        train_r2_score_mask = r2_score_split(train_data, train_pred_split)
        train_r_score_mask = correlation_score_split(train_data, train_pred_split)

        # score on test
        test_pred_split = pipeline.predict(test_feature, split=True)
        test_r2_score_mask = r2_score_split(test_data, test_pred_split)
        test_r_score_mask = correlation_score_split(test_data, test_pred_split)

        if trainer_config.fit_on_mask:
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
        if not os.path.exists(trainer_config.stats_save_dir):
            os.makedirs(trainer_config.stats_save_dir)

        stat_fn = os.path.join(
            trainer_config.stats_save_dir,
            f"{self.sub_config.sub_id}-{self.sub_config.task}-{self.feature_config.timescale}.npz",
        )

        np.savez_compressed(
            stat_fn,
            train_r2_split_scores=train_r2_split_scores,
            train_r_split_scores=train_r_split_scores,
            test_r2_split_scores=test_r2_split_scores,
            test_r_split_scores=test_r_split_scores,
        )

    def plot(
        self,
        trainer_config: TrainerConfig,
        feature_index: int = 0,
        is_corr: bool = False,
        is_train: bool = False,
    ):
        # load statfile
        stat_fn = os.path.join(
            trainer_config.stats_save_dir,
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

        ax = plot_flatmap_from_mapper(
            voxels=scores[feature_index],
            mapper_file=self.sub_config.sub_fmri_mapper_path,
            vmin=0,
            vmax=0.2,
        )
        plt.show()

    def plot2d(
        self,
        trainer_config: TrainerConfig,
        feature_indices: list = [0, 1],
        is_corr: bool = False,
    ):
        # load statfile
        stat_fn = os.path.join(
            trainer_config.stats_save_dir,
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

        ax = plot_2d_flatmap_from_mapper(
            voxels_1=scores[feature_indices[0]],
            voxels_2=scores[feature_indices[1]],
            mapper_file=self.sub_config.sub_fmri_mapper_path,
            vmin=0,
            vmax=0.2,
            vmin2=0,
            vmax2=0.2,
            label_1=self.train_feature_info[feature_indices[0]]["name"],
            label_2=self.train_feature_info[feature_indices[1]]["name"],
        )
        plt.show()
