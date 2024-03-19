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
        sub_setting_json=config_subject_path,
        feature_setting_json=config_feature_path,
        trainer_setting_json=config_train_path,
    )
    trainer.train()
    trainer.refit_and_evaluate()


feature_config_paths = [ 
    # ".temp/config/bling/feature/mBERT_all_untrimmed_timescale_en_COL.json",
    # ".temp/config/bling/feature/mBERT_all_untrimmed_timescale_zh_COL.json",
    ".temp/config/bling/feature/COL/mBERT_all_untrimmed_timescale_stepwise_en.json",
    ".temp/config/bling/feature/COL/mBERT_all_untrimmed_timescale_stepwise_zh.json"
    
]

sub_config_paths = [
    ".temp/config/bling/subject/COL_en.json",
    ".temp/config/bling/subject/COL_zh.json",
]

train_config_paths = [
    ".temp/config/bling/train/stepwise/col_en_timescale.json",
    ".temp/config/bling/train/stepwise/col_zh_timescale.json",
]

# # COL training
# train(
#     config_feature_path=feature_config_paths[0],
#     config_subject_path=sub_config_paths[0],
#     config_train_path=train_config_paths[0],
# )

# # ZH 
train(
    config_feature_path=feature_config_paths[1],
    config_subject_path=sub_config_paths[1],
    config_train_path=train_config_paths[1],
)

