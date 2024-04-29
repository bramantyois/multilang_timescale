import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np
import cortex
import os

from bling.data_loading.load_metadata import surface_metadata
from bling.utils import get_bh_excluded_voxels
import weight_analysis

subject = "COL"

experiment_name = "bling_reading"
weights_dir = "/mnt/raid/bling/call_scripts/weights/"
results_dir = "/mnt/raid/bling/call_scripts/results/"
feature_dir = "/mnt/raid/bling/data/features/"
figures_dir = "/mnt/antares_raid/home/mathislmrr/figures/weights_projected_vector"

delays = range(1, 5)
data_parameters_dict = {
    "do_load_Xs": True,
    "delays": delays,
    "do_load_Y": False,
    "Xs_load_method": "from_local",
    "Xs_save_filepath": "",
    "features": {"feature_names": []},
}

file = "Baseline_bling_reading_zh_COL_regress_out_Baseline_bling_reading_zh_COL_numwords-numletters-moten_preds_fTRS_results.npz"
results = np.load(os.path.join(results_dir, file.replace("_results", "_corr")))
excluded_voxels = get_bh_excluded_voxels(results["pvalues_residuals"], 0.05)[0]

data_parameters_dict["Xs_save_filepath"] = os.path.join(
    feature_dir, experiment_name, file.replace("_results", "")
)

feature_names = ["fastText_zh_RCSLS_TUB"]

data_parameters_dict["features"]["feature_names"] = feature_names

primal_weights = weight_analysis.get_primal_weights(
    "dual_from_local",
    os.path.join(weights_dir, file),
    data_parameters_dict,
    do_undelay_weights=True,
    do_mean_weights=True,
    primal_weights_save_filepath=None,
)
primal_weight = primal_weights[0].squeeze()

results = np.load(os.path.join(results_dir, file.replace("results", "corr")))
joint_score = results["scores_residuals"]

surface_dict = [
    surface_dict
    for surface_dict in surface_metadata[subject]
    if surface_dict["description"] == "alternative"
][0]
surface = surface_dict["surface"]
transform = surface_dict["transform"]

scaled_weights = weight_analysis.scale_weights_by_score_sqrt(primal_weight, joint_score)
