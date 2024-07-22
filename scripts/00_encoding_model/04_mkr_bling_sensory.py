import os
import sys

sys.path.insert(0, os.getcwd())

from src.trainer import Trainer


# en_subject_config = ".temp/config/bling/subject/COL_en.json"
# zh_subject_config = ".temp/config/bling/subject/COL_zh.json"

en_subject_config = ".temp/config/bling/subject/GFW_en.json"
zh_subject_config = ".temp/config/bling/subject/GFW_zh.json"

# en_subject_config = ".temp/config/bling/subject/TYE_en.json"
# zh_subject_config = ".temp/config/bling/subject/TYE_zh.json"

# en_feature_config = ".temp/config/bling/feature/sensory_feature_en.json"
# zh_feature_config = ".temp/config/bling/feature/sensory_feature_zh.json"  

# en_feature_config = ".temp/config/bling/feature/TYE/sensory_feature_en.json"
# zh_feature_config = ".temp/config/bling/feature/TYE/sensory_feature_zh.json"    

en_feature_config = ".temp/config/bling/feature/GFW/sensory_feature_en.json"
zh_feature_config = ".temp/config/bling/feature/GFW/sensory_feature_zh.json"    

trainer_config = ".temp/config/bling/train/trainer_lowlevel_stepwise.json"

for user in ['COL']:
    en_subject_config = f".temp/config/bling/subject/{user}_en.json"
    zh_subject_config = f".temp/config/bling/subject/{user}_zh.json"
    
    en_feature_config = f".temp/config/bling/feature/{user}/sensory_feature_en.json"
    zh_feature_config = f".temp/config/bling/feature/{user}/sensory_feature_zh.json"

    trainer = Trainer(
        sub_setting_json=en_subject_config,
        feature_setting_json=en_feature_config,
        trainer_setting_json=trainer_config)
    trainer.train()
    trainer.refit_and_evaluate()

    # now for the chinese
    trainer = Trainer(
        sub_setting_json=zh_subject_config,
        feature_setting_json=zh_feature_config,
        trainer_setting_json=trainer_config)
    trainer.train()
    trainer.refit_and_evaluate()