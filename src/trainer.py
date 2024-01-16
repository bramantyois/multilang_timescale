import os

import time
import logging

import numpy as np
import pandas as pd

from scipy.stats import zscore

from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV, KernelRidge

from joblib import dump

from .utils import make_delayed, load_dict
from .settings import FeatureSet, TrainerConfig


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

        self._set_backend()

        self._prepare_data()

        self._prepare_feature_sets()

        self._set_model()

    def _set_backend(self):
        set_backend(self.config.backend, on_error="warn")

    def _prepare_data(self):
        self.data_train = load_dict(self.config.sub_fmri_train_path)
        self.data_test = load_dict(self.config.sub_fmri_test_path)

        # zscore data
        self.data_train = np.vstack(
            [
                zscore(self.data_train[story][sub_trim_start:-sub_trim_end, :], axis=0)
                for story in train_stories
            ]
        )
        self.data_train = np.nan_to_num(self.data_train)
        
        self.data_test = np.vstack(
            [
                zscore(self.data_test[story][sub_trim_start:-sub_trim_end, :], axis=0)
                for story in test_stories
            ]
        )
        self.data_test = np.nan_to_num(self.data_test)
        
    def _prepare_feature_sets(self):
        train_features = []
        test_features = []  
        for feature in self.config.feature_sets:
            feature_set = np.load(feature.path, allow_pickle=True)
            
            
            