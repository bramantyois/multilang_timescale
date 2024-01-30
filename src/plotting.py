import numpy as np

import matplotlib.pyplot as plt

from voxelwise_tutorials.viz import map_voxels_to_flatmap
from voxelwise_tutorials.viz import _plot_addition_layers
from voxelwise_tutorials.io import load_hdf5_array

from src.trainer import Trainer


def plot_flatmap_from_mapper(
    voxels,
    mapper_file,
    ax=None,
    alpha=0.7,
    cmap="inferno",
    vmin=None,
    vmax=None,
    with_curvature=True,
    with_rois=True,
    with_colorbar=True,
    colorbar_location=(0.4, 0.9, 0.2, 0.05),
):
    """Plot a flatmap from a mapper file, with 1D data.

    This function is equivalent to the pycortex functions:
    cortex.quickshow(cortex.Volume(voxels, ...), ...)

    Note that this function does not have the full capability of pycortex,
    since it is based on flatmap mappers and not on the original brain
    surface of the subject.

    Parameters
    ----------
    voxels : array of shape (n_voxels, )
        Data to be plotted.
    mapper_file : str
        File name of the mapper.
    ax : matplotlib Axes or None.
        Axes where the figure will be plotted.
        If None, a new figure is created.
    alpha : float in [0, 1], or array of shape (n_voxels, )
        Transparency of the flatmap.
    cmap : str
        Name of the matplotlib colormap.
    vmin : float or None
        Minimum value of the colormap. If None, use the 1st percentile of the
        `voxels` array.
    vmax : float or None
        Minimum value of the colormap. If None, use the 99th percentile of the
        `voxels` array.
    with_curvature : bool
        If True, show the curvature below the data layer.
    with_rois : bool
        If True, show the ROIs labels above the data layer.
    colorbar_location : [left, bottom, width, height]
        Location of the colorbar. All quantities are in fractions of figure
        width and height.

    Returns
    -------
    ax : matplotlib Axes
        Axes where the figure has been plotted.
    """
    # create a figure
    if ax is None:
        flatmap_mask = load_hdf5_array(mapper_file, key="flatmap_mask")
        figsize = np.array(flatmap_mask.shape) / 100.0
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")

    # process plotting parameters
    vmin = np.percentile(voxels, 1) if vmin is None else vmin
    vmax = np.percentile(voxels, 99) if vmax is None else vmax
    if isinstance(alpha, np.ndarray):
        alpha = map_voxels_to_flatmap(alpha, mapper_file)

    alpha = np.nan_to_num(alpha, nan=0, posinf=0, neginf=0)
    # plot the data
    image = map_voxels_to_flatmap(voxels, mapper_file)
    cimg = ax.imshow(
        image, aspect="equal", zorder=1, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax
    )

    if with_colorbar:
        try:
            cbar = ax.inset_axes(colorbar_location)
        except AttributeError:  # for matplotlib < 3.0
            cbar = ax.figure.add_axes(colorbar_location)
        ax.figure.colorbar(cimg, cax=cbar, orientation="horizontal")

    # plot additional layers if present
    _plot_addition_layers(
        ax=ax,
        n_voxels=voxels.shape[0],
        mapper_file=mapper_file,
        with_curvature=with_curvature,
        with_rois=with_rois,
    )

    return ax


def plot_timeline_flatmaps(
    result_config_json, appendage: str = "Feature Set: BERT", is_corr=True
):
    trainer = Trainer(result_config_json=result_config_json)
    # plotting timescale
    stat = np.load(trainer.result_config.stats_path)
    if is_corr:
        scores = stat["test_r_split_scores"]
    else:
        scores = stat["test_r2_split_scores"]
    scores_timescale = scores[:8]

    max_scores = np.argmax(scores_timescale, axis=0)

    alpha = trainer.mask.astype(float)

    ax = plot_flatmap_from_mapper(
        max_scores, trainer.sub_config.sub_fmri_mapper_path, alpha=alpha, cmap="rainbow"
    )

    # plotting mask
    # plot_flatmap_from_mapper(
    #     non_mask, trainer.sub_config.sub_fmri_mapper_path,
    #     cmap="grey", ax=ax)
    is_corr_str = "pearson's r" if is_corr else "r2"
    ax.set_title(
        f"Timescale selectivity ({is_corr_str}) , Subject: {trainer.sub_config.sub_id}-{trainer.sub_config.task}, {appendage}"
    )
    plt.show()