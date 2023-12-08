# plotting config

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def config_plotting():
    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_palette("husl")

