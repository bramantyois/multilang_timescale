import os
import json

from itertools import product

import sys

sys.path.insert(0, os.getcwd())

from src.trainer import Trainer


def train(
    config_subject_path: str,
    config_feature_path: str,
    config_train_path: str,
):
    trainer = Trainer(
        sub_config_json=config_subject_path,
        feature_config_json=config_feature_path,
        trainer_config_json=config_train_path,
    )
    trainer.train()
    trainer.refit_and_evaluate()


feature_config_paths = [
    ".temp/config/bling/feature/mBERT_all_timescale_en.json",
    ".temp/config/bling/feature/mBERT_all_timescale_zh.json",
    ".temp/config/bling/feature/BERT_all_timescale_en.json",
]

sub_config_paths = [
    ".temp/config/bling/subject/COL_en.json",
    ".temp/config/bling/subject/COL_zh.json",
]

train_config_paths = [
    ".temp/config/bling/train/trainer.json",
]

# train(
#     config_subject_path=sub_config_paths[0],
#     config_feature_path=feature_config_paths[0],
#     config_train_path=train_config_paths[0],
# )

# train(
#     config_subject_path=sub_config_paths[1],
#     config_feature_path=feature_config_paths[1],
#     config_train_path=train_config_paths[0],
# )

train(
    config_subject_path=sub_config_paths[0],
    config_feature_path=feature_config_paths[2],
    config_train_path=train_config_paths[0],
)
