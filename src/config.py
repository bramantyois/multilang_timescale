# plotting config

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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

periods = [
    "2_4_words",
    "4_8_words",
    "8_16_words",
    "16_32_words",
    "32_64_words",
    "64_128_words",
    "128_256_words",
    "256+ words",
]


frequency_to_period_name_dict = {
    0.375: "2_4_words",
    0.1875: "4_8_words",
    0.09375: "8_16_words",
    0.046875: "16_32_words",
    0.0234375: "32_64_words",
    0.01171875: "64_128_words",
    0.005859375: "128_256_words",
    0.00390625: "256+ words",
}


frequency_str_to_period_name_dict = {
    "0.375": "2_4_words",
    "0.1875": "4_8_words",
    "0.09375": "8_16_words",
    "0.046875": "16_32_words",
    "0.0234375": "32_64_words",
    "0.01171875": "64_128_words",
    "0.005859375": "128_256_words",
    "0.00390625": "256+ words",

}


def config_plotting(context="paper"):
    sns.set_style("darkgrid")
    if context == "paper":
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    elif context == "talk":
        sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
    else:
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    sns.set_palette("Set3")


# data directoru
intermediate_output_path = "/media/data/dataset/timescale/intermediate_outputs/"