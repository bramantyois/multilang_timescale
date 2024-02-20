from typing import List, Literal, Optional

from pydantic import BaseSettings


class FeatureConfig(BaseSettings):
    timescale: List[str] = [
        "2_4_words",
        "4_8_words",
        "8_16_words",
        "16_32_words",
        "32_64_words",
        "64_128_words",
        "128_256_words",
        "256+ words",
    ]

    # lm feature
    lm_feature_type: str = ""
    lm_feature_path: str = ""

    # join sensory feature
    join_sensory_feature_path: Optional[str] = None
    join_sensory_feature_list: Optional[List[str]] = ["numwords", "numletters", "moten"]

    # sensory feature
    sensory_feature_train_paths: Optional[str] = None
    sensory_feature_test_paths: Optional[str] = None
    sensory_features: Optional[List[str]] = None

    sensory_feature_trim_start: int = 10
    sensory_feature_trim_end: int = 5

    # motion energy feature
    motion_energy_feature_paths: Optional[str] = None
    motion_energy_features: Optional[List[str]] = (
        None  # ["0", "1", "2", "3", "4", "5", "6", "7"]
    )

    # preprocessing related
    zscore_use_mean: bool = True
    zscore_use_std: bool = True


class TrainerConfig(BaseSettings):
    # training related
    kfolds: int = 5
    alpha_min: float = -10
    alpha_max: float = 10
    alpha_num: int = 21
    feature_delay: int = 4

    # multi kernel related
    n_targets_batch: int = 512
    n_alphas_batch: int = 8
    n_targets_batch_refit: int = 512
    n_iter: int = 1000
    solver: str = "random_search"

    # re-fit related
    use_fitted_alphas: bool = True
    use_fitted_deltas: bool = True

    # mask
    fit_on_mask: bool = True

    # backend related
    backend: Literal["numpy", "torch", "torch_cuda", "cupy"] = "numpy"

    # output related
    result_save_dir: str = ".temp/result"
    result_meta_save_dir: str = ".temp/result_meta"

    # #weights_save_dir: str = "./models/weights"
    # hyperparams_save_dir: str = "./models/hyperparams"
    # #model_save_path: str = "./models/model"

    # ## stats
    # stats_save_dir: str = "./models/stats"


class SubjectConfig(BaseSettings):
    # subject related
    sub_id: str = ""
    sub_fmri_train_path: Optional[str] = ""
    sub_fmri_test_path: Optional[str] = ""

    sub_fmri_train_test_path: Optional[str] = ""

    sub_fmri_mapper_path: str = ""
    
    lang_code: str = "en"

    # sub_fmri_mapper_path: str
    sub_trim_start: int = 10
    sub_trim_end: int = 10

    # modality related
    task: str = "reading"

    # mask related
    ev_threshold: float = 1e-9


class ResultConfig(BaseSettings):
    subject_config_path: str = ""
    feature_config_path: str = ""
    trainer_config_path: str = ""

    # output related
    result_dir: str = ""
    hyperparam_path: str = ""
    stats_path: str = ""
    plot_dir: str = ""