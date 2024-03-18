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
from himalaya.scoring import r2_score, correlation_score

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

from src.utils import (
    load_dict,
    compute_timescale_selectivity,
    permutation_test,
    permutation_test_mp,
    cook_responses,
)
from src.settings import TrainerSetting, SubjectSetting, FeatureSetting, ResultSetting
from src.configurations import (
    timescale_ranges,
    timescales,
    train_stories,
    test_stories,
    train_stories_zh,
    test_stories_zh,
)

# TODO: 
class Trainer:
    def __init__(
        self,
        sub_setting_json: Optional[str] = None,
        feature_setting_json: Optional[str] = None,
        trainer_setting_json: Optional[str] = None,
        result_setting_json: str = None,
        cuda_device_id: int = 1,
    ):
        should_create_result_config = True
        if result_setting_json is not None:
            temp = self._load_json(result_setting_json)
            self.result_config = ResultSetting(**temp)

            sub_setting_json = self.result_config.subject_config_path
            feature_setting_json = self.result_config.feature_config_path
            trainer_setting_json = self.result_config.trainer_config_path

            should_create_result_config = False

        temp = self._load_json(sub_setting_json)
        self.sub_setting = SubjectSetting(**temp)

        temp = self._load_json(feature_setting_json)
        self.feature_setting = FeatureSetting(**temp)

        temp = self._load_json(trainer_setting_json)
        self.trainer_setting = TrainerSetting(**temp)

        if should_create_result_config:
            self._generate_output_config(
                sub_setting_json, feature_setting_json, trainer_setting_json
            )

        self._prepare_data()
        self._prepare_features()

        set_config(assume_finite=True)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)

    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def _generate_output_config(
        self,
        sub_setting_json: str,
        feature_setting_json: str,
        trainer_setting_json: str,
    ):
        id = str(uuid.uuid4())

        result_dir = os.path.join(self.trainer_setting.result_save_dir, id)

        result_config = ResultSetting()
        result_config.subject_config_path = sub_setting_json
        result_config.feature_config_path = feature_setting_json
        result_config.trainer_config_path = trainer_setting_json

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
        if not os.path.exists(self.trainer_setting.result_meta_save_dir):
            os.makedirs(self.trainer_setting.result_meta_save_dir)
        result_setting_json = os.path.join(
            self.trainer_setting.result_meta_save_dir, f"{id}.json"
        )

        with open(result_setting_json, "w") as f:
            json.dump(result_config.__dict__, f, indent=4)

        self.result_config = result_config

    def _prepare_data(self):
        if self.sub_setting.sub_fmri_train_test_path is None:
            train_data = load_dict(self.sub_setting.sub_fmri_train_path)
            test_data = load_dict(self.sub_setting.sub_fmri_test_path)
            # zscore data
            ## zscore data
            train_data = np.vstack(
                [
                    zscore(
                        train_data[story][
                            self.sub_setting.sub_trim_start : -self.sub_setting.sub_trim_end,
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
            self.mask = ev > self.sub_setting.ev_threshold

            test_data = zscore(
                np.mean(test_data["story_11"], axis=0)[
                    self.sub_setting.sub_trim_start : -self.sub_setting.sub_trim_end, :
                ],
                axis=0,
            )
            self.test_data = np.nan_to_num(test_data)
        else:
            if self.sub_setting.lang_code == "en":
                train_strs = train_stories
                test_strs = test_stories
            else:
                train_strs = train_stories_zh
                test_strs = test_stories_zh

            data = load_dict(self.sub_setting.sub_fmri_train_test_path)
            train_data, test_data = cook_responses(
                data,
                test_runs=test_strs,
                train_runs=train_strs,
                trim_start_length=self.sub_setting.sub_trim_start,
                trim_end_length=self.sub_setting.sub_trim_end,
                multiseries="average_across",
                do_zscore=True,
            )

            self.train_data = np.nan_to_num(train_data)
            self.test_data = np.nan_to_num(test_data[0])

            # ev = explainable_variance(test_data)
            self.mask = np.ones(self.train_data.shape[1], dtype=bool)

    def _prepare_features(self):
        train_features = []
        test_features = []

        # lm-derived feature
        if self.feature_setting.lm_feature_type is None:
            print('skipping lm feature')
        else:    
            ## if feature is trimmed
            if (
                self.feature_setting.is_lm_feature_trimmed is None
                or self.feature_setting.is_lm_feature_trimmed is True
            ):
                    lm_feature_train_test = np.load(
                        self.feature_setting.lm_feature_path, allow_pickle=True
                    )

                    for t in self.feature_setting.timescale:
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
            else: # feature is not trimmed
                lm_feature_train_test = np.load(
                    self.feature_setting.lm_feature_path, allow_pickle=True
                )

                lm_train_untrimmed = lm_feature_train_test["train"].tolist()
                lm_test_untrimmed = lm_feature_train_test["test"].tolist()

                for t in self.feature_setting.timescale:
                    lm_train_feature = []
                    lm_test_feature = []
                    for s in train_stories:
                        train = lm_train_untrimmed[t][s][
                            self.feature_setting.lm_feature_trim_start : -self.feature_setting.lm_feature_trim_end
                        ]
                        lm_train_feature.append(train)

                    for s in test_stories:
                        test = lm_test_untrimmed[t][s][
                            self.feature_setting.lm_feature_trim_start : -self.feature_setting.lm_feature_trim_end
                        ]
                        lm_test_feature.append(test)

                    lm_train_feature = np.concatenate(lm_train_feature, axis=0)
                    lm_test_feature = np.concatenate(lm_test_feature, axis=0)

                    train_features.append(
                        {
                            "name": f"lm_{t}",
                            "size": lm_train_feature.shape[1],
                            "feature": np.nan_to_num(lm_train_feature),
                        }
                    )
                    test_features.append(
                        {
                            "name": f"lm_{t}",
                            "size": lm_test_feature.shape[1],
                            "feature": np.nan_to_num(lm_test_feature),
                        }
                    )

        # joint sensory-level features
        ## train
        if self.feature_setting.sensory_feature_train_paths is not None:
            sensory_level_train_feature = load_dict(
                self.feature_setting.sensory_feature_train_paths
            )

            for feature in self.feature_setting.sensory_features:
                story_stacked = []
                for story in list(sensory_level_train_feature.keys()):
                    story_stacked.append(
                        sensory_level_train_feature[story][feature][
                            self.feature_setting.sensory_feature_trim_start : -self.feature_setting.sensory_feature_trim_end
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
        if self.feature_setting.sensory_feature_test_paths is not None:
            sensory_level_test_feature = load_dict(
                self.feature_setting.sensory_feature_test_paths
            )

            for feature in self.feature_setting.sensory_features:
                story_stacked = []
                for story in list(sensory_level_test_feature.keys()):
                    story_stacked.append(
                        sensory_level_test_feature[story][feature][
                            self.feature_setting.sensory_feature_trim_start : -self.feature_setting.sensory_feature_trim_end
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
        if self.feature_setting.motion_energy_feature_paths is not None:
            moten = np.load(
                self.feature_setting.motion_energy_feature_paths, allow_pickle=True
            )
            ## train
            moten_train = moten["train"].tolist()

            for f in self.feature_setting.motion_energy_features:
                train_features.append(
                    {
                        "name": f"motion energy : {f}",
                        "size": moten_train[f].shape[1],
                        "feature": np.nan_to_num(moten_train[f]),
                    }
                )

            ## test
            moten_test = moten["test"].tolist()

            for f in self.feature_setting.motion_energy_features:
                test_features.append(
                    {
                        "name": f"motion energy : {f}",
                        "size": moten_test[f].shape[1],
                        "feature": np.nan_to_num(moten_test[f]),
                    }
                )

        # join sensory features
        if self.feature_setting.join_sensory_feature_path is not None:
            sensory_features = np.load(
                self.feature_setting.join_sensory_feature_path, allow_pickle=True
            )
            sensory_train_features = sensory_features["train_features"].tolist()
            sensory_test_features = sensory_features["test_features"].tolist()

            ## number of words
            if "numwords" in self.feature_setting.join_sensory_feature_list:
                train_features.append(
                    {
                        "name": "numwords",
                        "size": sensory_train_features["numwords"].shape[1],
                        "feature": sensory_train_features["numwords"],
                    }
                )
                test_features.append(
                    {
                        "name": "numwords",
                        "size": sensory_test_features["numwords"][0].shape[1],
                        "feature": sensory_test_features["numwords"][0],
                    }
                )

            ## number of letters
            if "numletters" in self.feature_setting.join_sensory_feature_list:
                train_features.append(
                    {
                        "name": "numletters",
                        "size": sensory_train_features["numletters"].shape[1],
                        "feature": sensory_train_features["numletters"],
                    }
                )
                test_features.append(
                    {
                        "name": "numletters",
                        "size": sensory_test_features["numletters"][0].shape[1],
                        "feature": sensory_test_features["numletters"][0],
                    }
                )

            ## moten
            if "moten" in self.feature_setting.join_sensory_feature_list:
                sub_id = self.sub_setting.sub_id
                train_features.append(
                    {
                        "name": "moten",
                        "size": sensory_train_features[f"motion_energy_{sub_id}"].shape[
                            1
                        ],
                        "feature": sensory_train_features[f"motion_energy_{sub_id}"],
                    }
                )
                test_features.append(
                    {
                        "name": "moten",
                        "size": sensory_test_features[f"motion_energy_{sub_id}"][
                            0
                        ].shape[1],
                        "feature": sensory_test_features[f"motion_energy_{sub_id}"][0],
                    }
                )

        # add index to feature
        for i, f in enumerate(train_features):
            f["index"] = i
        for i, f in enumerate(test_features):
            f["index"] = i

        # join features
        self.train_feature = np.hstack([f["feature"] for f in train_features])
        self.test_feature = np.hstack([f["feature"] for f in test_features])

        # delays = np.arange(1, self.trainer_setting.feature_delay + 1)
        # self.train_feature = make_delayed(train_feature, delays)

        assert (
            self.train_feature.shape[0] == self.train_data.shape[0]
        ), f"Feature and data shape mismatch {self.train_feature.shape[0]} != {self.train_data.shape[0]}"

        assert (
            self.test_feature.shape[0] == self.test_data.shape[0]
        ), f"Feature and data shape mismatch {self.test_feature.shape[0]} != {self.test_data.shape[0]}"

        # remove 'feature' key, save memory
        for f in train_features:
            del f["feature"]

        for f in test_features:
            del f["feature"]

        self.train_feature_info = train_features
        self.test_feature_info = test_features

    def get_kernelizer(self):
        preprocess_pipeline = make_pipeline(
            StandardScaler(
                with_mean=self.feature_setting.zscore_use_mean,
                with_std=self.feature_setting.zscore_use_std,
            ),
            Delayer(delays=np.arange(1, self.trainer_setting.feature_delay + 1)),
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
            n_iter=self.trainer_setting.n_iter,
            alphas=np.logspace(
                self.trainer_setting.alpha_min,
                self.trainer_setting.alpha_max,
                self.trainer_setting.alpha_num,
            ),
            n_targets_batch=self.trainer_setting.n_targets_batch,
            n_alphas_batch=self.trainer_setting.n_alphas_batch,
            n_targets_batch_refit=self.trainer_setting.n_targets_batch_refit,
        )

        mkr_model = MultipleKernelRidgeCV(
            kernels="precomputed",
            solver=self.trainer_setting.solver,
            solver_params=solver_params,
            cv=self.trainer_setting.kfolds,
        )

        return make_pipeline(columnn_kernelizer, mkr_model, verbose=False)

    def train(self, force_cpu: bool = False):
        if force_cpu:
            set_backend("torch", on_error="warn")
        else:
            set_backend(self.trainer_setting.backend, on_error="warn")

        pipeline = self.prepare_training_pipeline()

        # casting
        train_feature = self.train_feature.astype("float32")
        train_data = self.train_data.astype("float32")

        if self.trainer_setting.fit_on_mask:
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
        if self.trainer_setting.backend == "torch_cuda":
            del pipeline
            torch.cuda.empty_cache()

    # def compute_stats(self, prediction_split: np.ndarray, prediction: np.ndarray, target: np.ndarray force_cpu: bool=False):

    #     r2_score_mask = r2_score_split(target, prediction_split)
    #     r_score_mask = correlation_score_split(target, prediction_split)

    #     r_timescale_selectivity = compute_timescale_selectivity(
    #         r_score_mask[0:8]
    #     )
    #     r2_timescale_selectivity = compute_timescale_selectivity(
    #         r2_score_mask[0:8]
    #     )

    #     if self.trainer_setting.fit_on_mask:
    #         n_kernels = r2_score_mask.shape[0]
    #         n_voxels = self.test_data.shape[1]

    #         r2_split_scores = np.zeros((n_kernels, n_voxels))
    #         r_split_scores = np.zeros((n_kernels, n_voxels))

    #         r2_split_scores[:, self.mask] = backend.to_numpy(r2_score_mask)
    #         r_split_scores[:, self.mask] = backend.to_numpy(r_score_mask)

    #         r_selectivity = np.zeros(n_voxels)
    #         r2_selectivity = np.zeros(n_voxels)

    #         r_selectivity[self.mask] = backend.to_numpy(
    #             r_timescale_selectivity
    #         )
    #         r2_selectivity[self.mask] = backend.to_numpy(
    #             r2_timescale_selectivity
    #         )
    #     else:
    #         test_r2_split_scores = test_r2_score_mask
    #         test_r_split_scores = test_r_score_mask

    #         test_r2_selectivity = test_r2_timescale_selectivity
    #         test_r_selectivity = test_r_timescale_selectivity

    #     test_r2_selectivity = np.power(2, test_r2_selectivity)
    #     test_r_selectivity = np.power(2, test_r_selectivity)

    def refit_and_evaluate(self, force_cpu: bool = False, return_pred: bool = False):
        if force_cpu:
            backend = set_backend("torch", on_error="warn")
        else:
            backend = set_backend(self.trainer_setting.backend, on_error="warn")

        # load hyperparams
        hyperparams_fn = os.path.join(
            self.result_config.result_dir,
            f"hyperparams.npz",
        )

        hyperparams = np.load(hyperparams_fn)
        deltas = (
            hyperparams["deltas"].astype("float32")
            if self.trainer_setting.use_fitted_deltas
            else "zeros"
        )
        best_alphas = (
            hyperparams["best_alphas"].astype("float32")
            if self.trainer_setting.use_fitted_alphas
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
                "n_targets_batch": self.trainer_setting.n_targets_batch_refit
            },
        )

        pipeline = make_pipeline(columnn_kernelizer, model, verbose=False)

        # casting
        train_feature = self.train_feature.astype("float32")
        train_data = self.train_data.astype("float32")

        test_feature = self.test_feature.astype("float32")
        test_data = self.test_data.astype("float32")

        if self.trainer_setting.fit_on_mask:
            train_data = train_data[:, self.mask]
            test_data = test_data[:, self.mask]
            # print("using mask size of {train_data.shape}")

        # prepare features
        print("refitting...")
        pipeline.fit(train_feature, train_data)

        # score on train
        ## predict in batches
        print("predicting in batches...")

        def predict_in_batches(
            model, X, batch_size=self.trainer_setting.n_targets_batch_refit, split=True
        ):
            n_samples = X.shape[0]
            n_batches = int(np.ceil(n_samples / batch_size))
            y_pred = []
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = (batch_idx + 1) * batch_size
                y_pred_batch = model.predict(X[start:end], split=split)
                y_pred_batch = backend.to_numpy(y_pred_batch)
                y_pred.append(y_pred_batch)
            if split is True:
                y_pred = np.concatenate(y_pred, axis=1)
            else:
                y_pred = np.concatenate(y_pred)
            return y_pred

        # train_pred_split = predict_in_batches(pipeline, train_feature)
        test_pred_split = predict_in_batches(pipeline, test_feature)
        test_pred = predict_in_batches(pipeline, test_feature, split=False)

        # now do it in cpu
        print("computing scores...")
        if force_cpu:
            backend = set_backend("numpy", on_error="warn")

        # score on test
        test_r_score_mask = correlation_score_split(test_data, test_pred_split)
        test_r2_score_mask = r2_score_split(test_data, test_pred_split)

        test_joint_r_score_mask = correlation_score(test_data, test_pred)
        test_joint_r2_score_mask = r2_score(test_data, test_pred)

        # do permutation test
        if force_cpu:
            permutation_text_func = permutation_test_mp
        else:
            permutation_text_func = permutation_test
        
        print("computing permutation test...")
        test_p_values_r_mask = permutation_text_func(
            test_data, test_pred, score_func=correlation_score, num_permutations=2000
        )
        test_p_values_r2_mask = permutation_text_func(
            test_data, test_pred, score_func=r2_score, num_permutations=2000
        )
        
        # to numpy 
        test_r_score_mask = backend.to_numpy(test_r_score_mask)
        test_r2_score_mask = backend.to_numpy(test_r2_score_mask)
        test_joint_r_score_mask = backend.to_numpy(test_joint_r_score_mask)
        test_joint_r2_score_mask = backend.to_numpy(test_joint_r2_score_mask)
        test_p_values_r_mask = backend.to_numpy(test_p_values_r_mask)
        test_p_values_r2_mask = backend.to_numpy(test_p_values_r2_mask)

        # compute timescale selectivity
        if self.feature_setting.lm_feature_type is None:            
            print("skip timescale selectivity")
            test_r_selectivity_mask = 0
            test_r2_selectivity_mask = 0
        else:
            test_r_timescale_selectivity_mask = compute_timescale_selectivity(
                test_r_score_mask[0:8]
            )
            test_r2_timescale_selectivity_mask = compute_timescale_selectivity(
                test_r2_score_mask[0:8]
            )

            test_r_selectivity_mask = np.power(2, test_r_timescale_selectivity_mask)
            test_r2_selectivity_mask = np.power(2, test_r2_timescale_selectivity_mask)

        # saving stat
        print("saving stat...")
        np.savez_compressed(
            self.result_config.stats_path,
            test_r_score_mask=test_r_score_mask,
            test_r2_score_mask=test_r2_score_mask,
            test_joint_r_score_mask=test_joint_r_score_mask,
            test_joint_r2_score_mask=test_joint_r2_score_mask,
            test_p_values_r_mask=test_p_values_r_mask,
            test_p_values_r2_mask=test_p_values_r2_mask,
            test_r_selectivity_mask=test_r_selectivity_mask,
            test_r2_selectivity_mask=test_r2_selectivity_mask,
            mask=self.mask,
            test_info=self.test_feature_info,
            train_info=self.train_feature_info, 
        )

        # clear cuda memory
        if self.trainer_setting.backend == "torch_cuda":
            del pipeline
            torch.cuda.empty_cache()

        if return_pred:
            return {
                "test split predictions": test_pred_split,
                "test predictions": test_pred,
            }

        # if self.trainer_setting.fit_on_mask:
        #     n_kernels = test_r2_score_mask.shape[0]
        #     n_voxels = self.test_data.shape[1]

        #     test_r2_split_scores = np.zeros((n_kernels, n_voxels))
        #     test_r_split_scores = np.zeros((n_kernels, n_voxels))

        #     test_r2_split_scores[:, self.mask] = backend.to_numpy(test_r2_score_mask)
        #     test_r_split_scores[:, self.mask] = backend.to_numpy(test_r_score_mask)

        #     test_r_selectivity = np.zeros(n_voxels)
        #     test_r2_selectivity = np.zeros(n_voxels)

        #     test_r_selectivity[self.mask] = backend.to_numpy(
        #         test_r_timescale_selectivity
        #     )
        #     test_r2_selectivity[self.mask] = backend.to_numpy(
        #         test_r2_timescale_selectivity
        #     )

        # else:
        #     test_r2_split_scores = test_r2_score_mask
        #     test_r_split_scores = test_r_score_mask

        #     test_r2_selectivity = test_r2_timescale_selectivity
        #     test_r_selectivity = test_r_timescale_selectivity

    def get_scores(self):
        return np.load(self.result_config.stats_path)

    # def plot(
    #     self,
    #     feature_index: int = 0,
    #     is_corr: bool = False,
    #     is_train: bool = False,
    # ):
    #     # load statfile
    #     stat = np.load(self.result_config.stats_path)

    #     data_mode = "test"
    #     if is_train:
    #         data_mode = "train"

    #     score_mode = "r2"
    #     if is_corr:
    #         score_mode = "r"

    #     scores = stat[f"{data_mode}_{score_mode}_split_scores"]

    #     fig, ax = plt.subplots(figsize=(10, 10))

    #     plot_flatmap_from_mapper(
    #         voxels=scores[feature_index],
    #         mapper_file=self.sub_setting.sub_fmri_mapper_path,
    #         vmin=0,
    #         vmax=0.5,
    #         ax=ax,
    #     )

    #     plt.show()

    # def plot2d(
    #     self,
    #     feature_indices: list = [0, 1],
    #     is_corr: bool = False,
    #     is_train: bool = False,
    # ):
    #     # load statfile
    #     stat_fn = os.path.join(
    #         self.result_config.stat_dir,
    #         f"{self.sub_setting.sub_id}-{self.sub_setting.task}-{ self.feature_setting.timescale}.npz",
    #     )

    #     stat = np.load(stat_fn)

    #     data_mode = "test"
    #     if is_train:
    #         data_mode = "train"

    #     score_mode = "r2"
    #     if is_corr:
    #         score_mode = "r"

    #     scores = stat[f"{data_mode}_{score_mode}_split_scores"]

    #     fig, ax = plt.subplots(figsize=(10, 10))

    #     plot_2d_flatmap_from_mapper(
    #         voxels_1=scores[feature_indices[0]],
    #         voxels_2=scores[feature_indices[1]],
    #         mapper_file=self.sub_setting.sub_fmri_mapper_path,
    #         vmin=0,
    #         vmax=0.5,
    #         vmin2=0,
    #         vmax2=0.5,
    #         label_1=self.train_feature_info[feature_indices[0]]["name"],
    #         label_2=self.train_feature_info[feature_indices[1]]["name"],
    #         ax=ax,
    #     )
    #     plt.show()
