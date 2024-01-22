from typing import List, Literal, Optional

from pydantic import BaseSettings


# class FeatureSet(BaseSettings):
#     name: str
#     path: str
#     description: Optional[str] = None

# class TimescaleFeatureSet(FeatureSet):
#     timescale: Literal[] = "2_4_words"

class FeatureConfig(BaseSettings):
    timescale: str = "2_4_words"

    # lm feature
    lm_feature_path: str = ""
    
    # sensory feature
    sensory_feature_train_paths: Optional[str] = None
    sensory_feature_test_paths: Optional[str] = None
    sensory_features: Optional[List[str]] = None
    sensory_feature_trim_start: int = 10
    sensory_feature_trim_end: int = 5

    # motion energy feature
    motion_energy_feature_paths: Optional[str] = None
    motion_energy_features: Optional[List[str]] = ["0", "1", "2", "3", "4", "5", "6", "7"]


class TrainerConfig(BaseSettings):
    # training related
    kfolds: int = 5
    alpha_min: float = -10
    alpha_max: float = 10
    alpha_num: int = 21
    feature_delay: int = 4
    
    # multi kernel related
    n_targets_batch: int = 1000
    n_alphas_batch: int = 10
    n_targets_batch_refit: int = 500
    n_iter: int = 1000
    solver: str = "random_search"

    # re-fit related
    use_fitted_alphas: bool = False
    use_fitted_deltas: bool = True

    # mask
    fit_on_mask: bool = True    
    
    # backend related
    backend: Literal["numpy", "torch", "torch_cuda", "cupy"] = "numpy"

    # output related
    #weights_save_dir: str = "./models/weights"
    hyperparams_save_dir: str = "./models/hyperparams"
    #model_save_path: str = "./models/model"
    
    ## stats
    stats_save_dir: str = "./models/stats" 


class SubjectConfig(BaseSettings):
    # subject related
    sub_id: str = ""
    sub_fmri_train_path: str = ""
    sub_fmri_test_path: str = ""
    sub_fmri_mapper_path: str = ""

    # sub_fmri_mapper_path: str
    sub_trim_start: int = 10
    sub_trim_end: int = 10

    # modality related
    task: str = "reading"
