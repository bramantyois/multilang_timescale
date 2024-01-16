from typing import List, Optional, Literal

from pydantic import BaseSettings

from .config import train_stories, test_stories, timescales

# class FeatureSet(BaseSettings):
#     name: str
#     path: str
#     description: Optional[str] = None

# class TimescaleFeatureSet(FeatureSet):
#     timescale: Literal[] = "2_4_words"
        
class TrainerConfig(BaseSettings):
    # subject related
    sub_id: str
    sub_fmri_train_path: str
    sub_fmri_test_path: str
    # sub_fmri_mapper_path: str
    sub_trim_start: int
    sub_trim_end: int
    
    # story related
    train_stories: List[str] 
    test_stories: List[str]
    
    # feature related
    timescale: Literal[timescales]
    lm_feature_path: str
    sensory_level_feature_path: str
        
    # training related
    kfolds: int
    alpha_min: float
    alpha_max: float 
    alpha_num: int
    feature_delay: int = 4
    
    # backend related
    backend: Literal["numpy", "torch", "torch-cuda", "cupy"] = "numpy"
    
    # output related
    weights_save_path: str
    hyperparams_save_path: str
    
    