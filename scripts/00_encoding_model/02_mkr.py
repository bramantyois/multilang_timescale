import os
import json

from itertools import product

import sys

sys.path.insert(0, os.getcwd())
 
from src.trainer import Trainer
from src.settings import TrainerConfig, SubjectConfig, FeatureConfig


def train(config_train_path: str, config_subject_dir: str, config_feature_dir: str):
    # now load the config files and train
    ## get list of subject config files
    sub_json = os.listdir(config_subject_dir)
    sub_json = [j for j in sub_json if j.endswith(".json")]
    sub_json = [os.path.join(config_subject_dir, j) for j in sub_json]

    ## get list of train .json config files
    train_json = [config_train_path]

    feature_json = os.listdir(config_feature_dir)  
    feature_json = [j for j in feature_json if j.endswith(".json")]
    feature_json = [os.path.join(config_feature_dir, j) for j in feature_json]

    # now make all possible combinations
    configs = list(product(sub_json, train_json, feature_json))

    # iterate over the list
    for c in configs:
        print('training for: ')
        print(c)
        # load subject config
        with open(c[0]) as f:
            sub_config = json.load(f)
        sub_config = SubjectConfig(**sub_config)
        
        # load train config
        with open(c[1]) as f:
            trainer_config = json.load(f)
        trainer_config = TrainerConfig(**trainer_config)
        
        # load feature config
        with open(c[2]) as f:
            feature_config = json.load(f)
        feature_config = FeatureConfig(**feature_config)
        
        trainer = Trainer(sub_config=sub_config, feature_config=feature_config)
        trainer.train(trainer_config)
        trainer.refit_and_evaluate(trainer_config)

subject_config_dir = ".temp/config/subject/"
 
train_config_bert_paths = [
    ".temp/config/train/bert_trainer_config_100.json",
    #".temp/config/train/bert_trainer_config.json", d
]

train_config_mbert_paths = [
    
    ".temp/config/train/mbert_trainer_config_100.json",
    #".temp/config/train/mbert_trainer_config.json"
]

feature_config_bert_dir = ".temp/config/feature/bert"
feature_config_mbert_dir = ".temp/config/feature/mbert"

for c in train_config_mbert_paths:
    train(c, subject_config_dir, feature_config_mbert_dir)

# train for all configs
for c in train_config_bert_paths:
    train(c, subject_config_dir, feature_config_bert_dir)
    
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