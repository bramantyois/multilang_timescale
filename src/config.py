import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def config_plotting(context="paper"):
    sns.set_style("darkgrid")
    if context == "paper":
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    elif context == "talk":
        sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
    else:
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    sns.set_palette("Set3")

train_stories = [
    "alternateithicatom",
    "avatar",
    "howtodraw",
    "legacy",
    "life",
    "myfirstdaywiththeyankees",
    "naked",
    "odetostepfather",
    "souls",
    "undertheinfluence",
]

test_stories = ["wheretheressmoke"]


timescales = [
    "2_4_words",
    "4_8_words",
    "8_16_words",
    "16_32_words",
    "32_64_words",
    "64_128_words",
    "128_256_words",
    "256+ words",
]

# # data directories
# feature_sets_path = "/media/data/dataset/timescale/feature_sets/"
# feature_sets_en_path = os.path.join(feature_sets_path, "en")
# feature_sets_zh_path = os.path.join(feature_sets_path, "zh")

# intermediate_output_path = os.path.join(feature_sets_path, "intermediate_output")

# fmri_path = "/media/data/dataset/timescale/fmri/"
# mapper_path = os.path.join(fmri_path, "mappers")

# reading_data_path = os.path.join(fmri_path, "bling_reading")
# reading_data_en_path = os.path.join(reading_data_path, "en")
# reading_data_zh_path = os.path.join(reading_data_path, "zh")

# grids_path = os.path.join(fmri_path, "grids")
# grids_en_path = os.path.join(grids_path, "grids_en")
# grids_zh_path = os.path.join(grids_path, "grids_zh")

# trs_path = os.path.join(fmri_path, "trfiles")
# trs_en_path = os.path.join(trs_path, "trfiles_en")
# trs_zh_path = os.path.join(trs_path, "trfiles_zh")

# model_save_path = "/media/data/dataset/timescale/models"
# weights_save_path = os.path.join(model_save_path, "weights")
# hyperparams_save_path = os.path.join(model_save_path, "hyperparams")
# stats_save_path = os.path.join(model_save_path, "stats")

