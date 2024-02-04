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
    ".temp/config/feature/BERT-all_timescales-7-feature_config.json",
    ".temp/config/feature/mBERT-all_timescales-7-feature_config.json",
]

sub_config_paths = [
    ".temp/config/subject/subject-07-reading.json",
    #".temp/config/subject/subject-07-listening.json",
]


train_config_paths = [
    #".temp/config/train/trainer_config_shorttime.json",
    ".temp/config/train/trainer_config.json",
]


# now make all possible combinations
configs = list(product(sub_config_paths, feature_config_paths, train_config_paths))

# train
for c in configs:
    print("training for: ")
    print(c)
    train(
        config_subject_path=c[0],
        config_feature_path=c[1],
        config_train_path=c[2],
    )
    

# # now for mBERT
# config_train_path = ".temp/config/train/mbert_trainer_config.json"

# train_json = [config_train_path]

# config_feature_dir = ".temp/config/feature/mbert"

# feature_json = os.listdir(config_feature_dir)
# feature_json = [j for j in feature_json if j.endswith(".json")]
# feature_json = [os.path.join(config_feature_dir, j) for j in feature_json]

# # now make all possible combinations
# configs = list(product(sub_json, train_json, feature_json))

# # iterate over the list
# for c in configs:
#     print('training for: ')
#     print(c)

#     # load subject config
#     with open(c[0]) as f:
#         sub_config = json.load(f)
#     sub_config = SubjectConfig(**sub_config)

#     # load train config
#     with open(c[1]) as f:
#         trainer_config = json.load(f)
#     trainer_config = TrainerConfig(**trainer_config)

#     # load feature config
#     with open(c[2]) as f:
#         feature_config = json.load(f)
#     feature_config = FeatureConfig(**feature_config)

#     trainer = Trainer(sub_config=sub_config, feature_config=feature_config)
#     trainer.train(trainer_config)
#     trainer.refit_and_evaluate(trainer_config)
