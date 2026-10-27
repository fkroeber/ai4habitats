import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_samples(
    res,
    ds,
    out_path=None,
    prob_cmap="magma",
    mask_ignore_in_pred=True,
    rgb_bands=("B4", "B3", "B2"),
    rgb_time_idx=12,
    rgb_clip=0.2,
):
    """
    Args:
        res: apply_model(return_res=True) output with keys 'indices', 'inputs', 'preds', 'probs', 'targets'.
        ds:  S2_Dataset the loader was built from (to fetch colour LUT + provenance + norm stats).
        out_path: if specified, the path to save the plot image. Otherwise, the plot is shown interactively.
        prob_cmap: colormap for the confidence map (p(argmax class)).
        mask_ignore_in_pred: if True, ignore pixels are masked in the prediction and confidence maps.
        rgb_bands: tuple of 3 band names to use for RGB reconstruction from the input data.
        rgb_time_idx: index of the time step to use for RGB reconstruction (0-based).
        rgb_clip: maximum reflectance value for RGB reconstruction (values above this are clipped to 1).

    Returns:
        A plot with 4 rows and N columns, where
        - columns = samples
        - rows = [input RGB, ground truth, prediction, p(argmax)]
    """
    inputs = res["inputs"].numpy()  # (N, C=bands*times, H, W), normalised
    preds = res["preds"].numpy()
    probs = res["probs"]
    targets = res["targets"].numpy()
    indices = res["indices"].numpy()

    # --- colour LUT, ordered by encoding; ignore_index -> white ---
    lut = (
        ds.label_map_df[["target_encoding", "target_semantics", "target_color"]]
        .drop_duplicates("target_encoding")
        .sort_values("target_encoding")
    )
    enc = lut["target_encoding"].to_numpy()
    colours = lut["target_color"].tolist()
    names = {int(e): str(s) for e, s in zip(enc, lut["target_semantics"])}
    label_cmap = ListedColormap(colours)
    label_cmap.set_bad("white")  # (1) NaN / ignore -> white
    K = int(enc.max())
    label_norm = BoundaryNorm(np.arange(-0.5, K + 1.5), label_cmap.N)

    # --- confidence + label maps ---
    if probs.shape[1] == 1:
        p1 = probs[:, 0]
        conf_map = np.maximum(p1, 1 - p1).numpy()
        pred_map = preds[:, 0] if preds.ndim == 4 else preds
        gt_map = targets[:, 0] if targets.ndim == 4 else targets
    else:
        conf_map = probs.max(1).values.numpy()
        pred_map, gt_map = preds, targets

    # --- reconstruct one-timestep RGB from the flattened, normalised input ---
    # channel layout (from S2_Dataset): band-major, time-minor -> band*n_times + time
    n_times = len(ds.i_time)
    band_names = [ds.BAND_ORDER[b] for b in ds.i_band]  # selected bands, in order
    mean = np.asarray(ds.norm_stats.mean)  # per selected channel
    std = np.asarray(ds.norm_stats.std)

    missing = [b for b in rgb_bands if b not in band_names]
    if missing:
        raise ValueError(
            f"rgb_bands {missing} are not among the selected bands {band_names}"
        )
    rgb_time_idx = min(rgb_time_idx, n_times - 1)  # fewer time steps selected

    def sel_channel(band):  # local index into i_band
        return band_names.index(band) * n_times + rgb_time_idx

    rgb_ch = [sel_channel(b) for b in rgb_bands]

    def to_rgb(sample):  # (C,H,W) -> (H,W,3) display
        chans = []
        for c in rgb_ch:
            x = sample[c] * std[c] + mean[c]  # undo A.Normalize
            chans.append(x)
        rgb = np.stack(chans, axis=-1)
        return np.clip(rgb / rgb_clip, 0, 1)  # scale reflectance to [0,1]

    # --- provenance ---
    meta = ds.meta_file
    has_pid = "patch_id" in meta.columns
    col_labels = [
        str(meta.iloc[int(i)]["patch_id"]) if has_pid else str(meta.index[int(i)])
        for i in indices
    ]

    n = pred_map.shape[0]
    row_titles = ["input (RGB)", "ground truth", "prediction", "p(argmax class)"]
    fig, axes = plt.subplots(
        4, n, figsize=(4 * n, 16), squeeze=False, constrained_layout=True
    )
    present = set()

    for j in range(n):
        gt = gt_map[j].astype(float)
        pr = pred_map[j].astype(float)
        conf = conf_map[j].astype(float)
        ignore = gt < 0
        gt[ignore] = np.nan
        if mask_ignore_in_pred:
            pr[ignore] = np.nan
            conf[ignore] = np.nan
        present.update(np.unique(gt[~np.isnan(gt)]).astype(int).tolist())
        present.update(np.unique(pr[~np.isnan(pr)]).astype(int).tolist())

        axes[0, j].imshow(to_rgb(inputs[j]), interpolation="nearest")
        axes[1, j].imshow(gt, cmap=label_cmap, norm=label_norm, interpolation="nearest")
        axes[2, j].imshow(pr, cmap=label_cmap, norm=label_norm, interpolation="nearest")
        im = axes[3, j].imshow(
            conf, cmap=prob_cmap, vmin=0, vmax=1, interpolation="nearest"
        )

        axes[0, j].set_title(f"patch {col_labels[j]}", fontsize=10)
        for r in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            if j == 0:
                axes[r, j].set_ylabel(row_titles[r], fontsize=11)

    fig.colorbar(
        im,
        ax=axes[3, :].tolist(),
        fraction=0.02,
        pad=0.01,
        location="right",
        label="confidence",
    )

    handles = [
        mpatches.Patch(
            facecolor=colours[e], edgecolor="black", label=names.get(e, str(e))
        )
        for e in sorted(present)
        if 0 <= e < len(colours)
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(handles), 6),
            fontsize=9,
            frameon=False,
        )

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()
