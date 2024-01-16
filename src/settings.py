from typing import List, Optional, Literal

from pydantic import BaseSettings


class FeatureSet(BaseSettings):
    name: str
    path: str
    description: Optional[str] = None

class TimescaleFeatureSet(FeatureSet):
    timescale: Literal[
        "2_4_words",
        "4_8_words",
        "8_16_words",
        "16_32_words",
        "32_64_words",
        "64_128_words",
        "128_256_words",
        "256+ words",] = "2_4_words"
        
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
    feature_sets: List[FeatureSet]
    
    # training related
    kfolds: int
    alpha_num: int
    alpha_min: float
    alpha_max: float 
    feature_delay: int = 4
    
    # backend related
    backend: Literal["numpy", "torch", "torch-cuda", "cupy"] = "numpy"
    
    # output related
    weights_save_path: str
    hyperparams_save_path: str
    