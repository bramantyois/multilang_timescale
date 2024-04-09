from typing import Dict, Literal, Optional, List, Tuple

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable

from sklearn.linear_model import LinearRegression

import cortex
from cortex import quickshow
from cortex import Volume, VolumeRGB, Vertex
from cortex.quickflat import make_figure

from .utils import put_values_on_mask


def plot_timescale_flatmap_from_volume(
    volume: Volume,
    title: str = "Timescale Selectivity",
    mask: np.ndarray = None,
    ax=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    make_figure(
        volume,
        fig=ax,
        with_colorbar=False,
        with_curvature=True,
        nanmean=True,
        ax=ax,
        **kwargs,
    )

    ax.axis("off")

    # add vertical colorbar
    cax = fig.add_axes([0.4, 0.9, 0.2, 0.05])
    cbar = plt.colorbar(ax.images[0], cax=cax, orientation="horizontal")

    # cbar = plt.colorbar(ax.images[0], cax=cax)
    cbar.set_label("Number of Words")
    cbar.set_ticks([8, 16, 32, 64, 128, 256])
    cbar.set_ticklabels([8, 16, 32, 64, 128, 256])

    return ax


def plot_flatmap_from_vertex(
    vertex: Vertex,
    title: str = "Timescale Selectivity",
    ax=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    make_figure(
        vertex,
        fig=ax,
        with_colorbar=False,
        with_curvature=True,
        nanmean=True,
        ax=ax,
        **kwargs,
    )

    ax.axis("off")

    # add vertical colorbar
    cax = fig.add_axes([0.4, 0.9, 0.2, 0.05])
    cbar = plt.colorbar(ax.images[0], cax=cax, orientation="horizontal")

    # cbar = plt.colorbar(ax.images[0], cax=cax)
    cbar.set_label("Number of Words")
    cbar.set_ticks([8, 16, 32, 64, 128, 256])
    cbar.set_ticklabels([8, 16, 32, 64, 128, 256])

    return ax


def get_timescale_rgb(
    timescale: np.ndarray,
    vmin: int = 8,
    vmax: int = 256,
    cmap: str = "rainbow",
    is_log: bool = False,
    is_symmetric: bool = False,
):
    """get timescale rgb value from timescale value"""
    if is_log:
        if is_symmetric:
            norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.01)
        else:
            norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    rgb = scalar_mappable.to_rgba(np.nan_to_num(timescale))
    rgb = rgb[:, :3]

    # set to uint8
    rgb = (rgb * 255).astype(np.uint8)

    return rgb, scalar_mappable


def get_alpha_mask(score: np.ndarray, is_r2: bool = True):
    """
    get alpha mask from score
    """
    alpha_mask = score.copy()

    alpha_mask[alpha_mask < 0] = 0
    alpha_mask = np.nan_to_num(alpha_mask)

    if is_r2:
        alpha_mask = np.sqrt(alpha_mask)

    alpha_mask /= alpha_mask.max()

    return alpha_mask


def plot_volume_rgb(
    timescale: np.ndarray,
    surface_dict: dict,
    vmin: int = 8,
    vmax: int = 256,
    cmap: str = "rainbow",
    is_log: bool = False,
    is_symmetric: bool = False,
    is_r2: bool = True,
    is_timescale: bool = True,
    mask: np.ndarray = None,
    ax=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if mask is not None:
        alpha_mask = get_alpha_mask(mask, is_r2=is_r2)
    else:
        alpha_mask = np.ones_like(timescale)

    rgb, scalar_mappable = get_timescale_rgb(
        timescale,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        is_log=is_log,
        is_symmetric=is_symmetric,
    )

    red = Volume(rgb[:, 0], surface_dict["surface"], surface_dict["transform"])
    green = Volume(rgb[:, 1], surface_dict["surface"], surface_dict["transform"])
    blue = Volume(rgb[:, 2], surface_dict["surface"], surface_dict["transform"])

    vol_rgb = VolumeRGB(
        red,
        green,
        blue,
        surface_dict["surface"],
        surface_dict["transform"],
        alpha=alpha_mask,
    )

    quickshow(
        vol_rgb,
        with_curvature=True,
        with_colorbar=False,
        nanmean=True,
        fig=ax,
        **kwargs,
    )
    # add vertical colorbar
    cax = fig.add_axes([0.4, 0.9, 0.2, 0.05])
    cbar = plt.colorbar(scalar_mappable, cax=cax, orientation="horizontal")

    if is_timescale:
        cbar.set_label("Number of Words")
        cbar.set_ticks([8, 16, 32, 64, 128, 256])
        cbar.set_ticklabels([8, 16, 32, 64, 128, 256])

    return ax


def get_joint_result(
    first_stat,
    second_stat,
    result_type: Literal["timescale", "accuracy"],
    result_metric: Literal["r", "r2"],
    alpha: float = 0.05,
    ev_mask: Optional[np.ndarray] = None,
    mask_type: Literal["p-values", "prediction_accuracy"] = "p-values",
) -> Tuple:
    if result_type == "timescale":
        keyword = f"test_{result_metric}_selectivity_mask"
        p_val_keyword = f"test_p_values_{result_metric}_mask"
        value_range = (8, 256)
    elif result_type == "accuracy":
        keyword = f"test_joint_{result_metric}_score_mask"
        p_val_keyword = f"test_p_values_{result_metric}_mask"
        value_range = (0, 1)

    pred_acc_keyword = f"test_joint_{result_metric}_score_mask"

    if mask_type == "p-values":
        result_first, valid_voxel_first = put_values_on_mask(
            first_stat[keyword],
            first_stat[p_val_keyword],
            ev_mask=ev_mask,
            alpha=alpha,
            valid_range=value_range,
        )

        result_second, valid_voxel_second = put_values_on_mask(
            second_stat[keyword],
            second_stat[p_val_keyword],
            ev_mask=ev_mask,
            alpha=alpha,
            valid_range=value_range,
        )
    elif mask_type == "prediction_accuracy":
        first_mask = first_stat[pred_acc_keyword]
        second_mask = second_stat[pred_acc_keyword]
        if result_metric == "r2":
            first_mask[first_mask < 0] = 0
            second_mask[second_mask < 0] = 0

            first_mask = np.nan_to_num(first_mask)
            second_mask = np.nan_to_num(second_mask)

            first_mask = np.sqrt(first_mask)
            second_mask = np.sqrt(second_mask)

        valid_voxel_first = first_mask > alpha
        valid_voxel_second = second_mask > alpha

        valid_voxel_first = np.where(valid_voxel_first)[0]
        valid_voxel_second = np.where(valid_voxel_second)[0]

        result_first = first_stat[keyword]
        result_second = second_stat[keyword]

    common_voxels = np.intersect1d(valid_voxel_first, valid_voxel_second)

    result_first = result_first[common_voxels]
    result_second = result_second[common_voxels]

    return result_first, result_second


def plot_joint_result(
    first_stat,
    second_stat,
    result_type: Literal["timescale", "accuracy"],
    result_metric: Literal["r", "r2"],
    alpha: int = 0.05,
    ev_mask: Optional[np.ndarray] = None,
    mask_type: Literal["p-values", "prediction_accuracy"] = "p-values",
    labels: List[str] = ["EN", "ZH"],
    add_regression_line: bool = False,
):
    result_first, result_second = get_joint_result(
        first_stat,
        second_stat,
        result_type,
        result_metric,
        alpha,
        ev_mask,
        mask_type=mask_type,
    )

    if result_type == "accuracy" and result_metric == "r2":
        result_first = np.sqrt(result_first)
        result_second = np.sqrt(result_second)

    diff = np.abs(result_first - result_second)

    sns.scatterplot(x=result_first, y=result_second, hue=diff, palette="viridis")

    if result_type == "timescale":
        plt.xlim(8, 256)
        plt.ylim(8, 256)
    elif result_type == "accuracy":
        # get max from result first and result second
        max = np.max([result_first, result_second])
        min = np.min([result_first, result_second])
        plt.xlim(min, max)
        plt.ylim(min, max)

    plt.ylabel(labels[1])
    plt.xlabel(labels[0])

    slope = 0
    if add_regression_line:
        # add regression line without intercept/bias
        reg = LinearRegression(fit_intercept=False).fit(
            result_first.reshape(-1, 1), result_second
        )
        plt.plot(
            result_first,
            reg.predict(result_first.reshape(-1, 1)),
            color="red",
            linestyle="--",
        )
        slope = reg.coef_
    else:
        # add diagonal line
        if result_type == "timescale":
            plt.plot([8, 256], [8, 256], color="red", linestyle="--")
        elif result_type == "accuracy":
            plt.plot([0, 1], [0, 1], color="red", linestyle="--")

    if result_type == "timescale":
        type_title = "timescale selectivity"
        if result_metric == "r2":
            type_title += " (r2)"
        else:
            type_title += " (R)"
    elif result_type == "accuracy":
        if result_metric == "r2":
            type_title = "R2 Score"
        else:
            type_title = "Correlation Score (R)"

    # type_title = "timescale selectivity" if result_type == "timescale" else "accuracy"
    title = f"Pairwise Comparison of {type_title.capitalize()}"

    if add_regression_line:
        title += " with Regression Line (Slope: {:.2f})".format(slope[0])

    plt.title(title)

    plt.legend(title="Absolute Difference", loc="upper right")

    plt.show()


def plot_density(
    en_stats,
    zh_stats,
    result_type: Literal["timescale", "accuracy"] = "timescale",
    result_metric: Literal["r", "r2"] = "r2",
    alpha: float = 0.01,
    subject_id="COL",
    mask_type: Literal["p-values", "prediction_accuracy"] = "p-values",
):
    if result_type == "timescale":
        keyword = f"test_{result_metric}_selectivity_mask"
        p_val_keyword = f"test_p_values_{result_metric}_mask"
        valid_range = (8, 256)
    else:
        keyword = f"test_joint_{result_metric}_score_mask"
        p_val_keyword = f"test_p_values_{result_metric}_mask"
        valid_range = (0, 1)

    pred_acc_keyword = f"test_joint_{result_metric}_score_mask"

    en_stat = en_stats[keyword]
    zh_stat = zh_stats[keyword]
    if result_type == "accuracy" and result_metric == "r2":
        en_stat = np.sqrt(en_stat)
        zh_stat = np.sqrt(zh_stat)

    if mask_type == "p-values":
        result_en, _ = put_values_on_mask(
            en_stat,
            en_stats[p_val_keyword],
            ev_mask=None,
            alpha=alpha,
            valid_range=valid_range,
        )
        en_stat = result_en[~np.isnan(result_en)]

        result_zh, _ = put_values_on_mask(
            zh_stat,
            zh_stats[p_val_keyword],
            ev_mask=None,
            alpha=alpha,
            valid_range=valid_range,
        )
        zh_stat = result_zh[~np.isnan(result_zh)]
    else:
        en_mask = en_stats[pred_acc_keyword]
        zh_mask = zh_stats[pred_acc_keyword]

        en_mask[en_mask < 0] = 0
        zh_mask[zh_mask < 0] = 0
        if result_metric == "r2":
            en_mask = np.nan_to_num(en_mask)
            zh_mask = np.nan_to_num(zh_mask)

            en_mask = np.sqrt(en_mask)
            zh_mask = np.sqrt(zh_mask)

        en_mask = en_mask > alpha
        zh_mask = zh_mask > alpha

        result_en = en_stat.copy()
        result_en[~en_mask] = np.nan

        result_zh = zh_stat.copy()
        result_zh[~zh_mask] = np.nan

        en_valid_voxel = np.where(en_mask)[0]
        zh_valid_voxel = np.where(zh_mask)[0]

        en_stat = en_stat[en_valid_voxel]
        zh_stat = zh_stat[zh_valid_voxel]

    diff = result_en - result_zh
    # drop nan
    diff = diff[~np.isnan(diff)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()

    # plot density of both en_stat and zh_stat in one plot
    sns.kdeplot(en_stat, color="blue", label="EN", fill=True, ax=axs[0])
    sns.kdeplot(zh_stat, color="red", label="ZH", fill=True, ax=axs[0])

    subtitle = "Timescale Selectivity" if result_type == "timescale" else "Accuracy"

    axs[0].set_title(f"Density of {subtitle} (Subject: {subject_id})")
    axs[0].set_xlabel(f"{subtitle} Value")
    axs[0].legend()

    sns.kdeplot(diff, color="green", label="EN - ZH", fill=True, ax=axs[1])
    axs[1].set_title(f"Density of {subtitle} Difference (Subject: {subject_id})")
    axs[1].set_xlabel(f"{subtitle} Difference")
    axs[1].legend()

    if result_type == "timescale":
        axs[0].set_xlim(0, 260)
        axs[1].set_xlim(-128, 128)

    plt.show()
