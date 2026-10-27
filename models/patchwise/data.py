import albumentations as A
import json
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import uuid
import xarray as xr
from matplotlib import patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from models.paths import data_dir_of, resolve_paths


class S2_Dataset(torch.utils.data.Dataset):
    COVAR_ORDER = ["polygon_id", "lrt_status", "acquisition_date"]
    BAND_ORDER = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    TIME_ORDER = [
        "y1-03",
        "y1-04",
        "y1-05",
        "y1-06",
        "y1-07",
        "y1-08",
        "y1-09",
        "y1-10",
        "y2-03",
        "y2-04",
        "y2-05",
        "y2-06",
        "y2-07",
        "y2-08",
        "y2-09",
        "y2-10",
        "y3-03",
        "y3-04",
        "y3-05",
        "y3-06",
        "y3-07",
        "y3-08",
        "y3-09",
        "y3-10",
    ]

    def __init__(
        self,
        meta_file,
        sel_time=None,
        sel_band=None,
        sel_task="lvl0_bic",
        normalize=True,
        transform=None,
        use_cache=None,
        cache_dir=None,
        read_covars=False,
    ):
        """
        Base Dataset class.

        Args:
            meta_file (str, os.PathLike or pd.DataFrame): Patch metadata, i.e. the data set's
                patches_meta.csv or a DataFrame derived from it, with the columns 'path_input'
                (Sentinel-2 NetCDF) and 'path_label' (label NetCDF). Paths in a CSV are relative
                to the CSV's folder. A DataFrame is used as is, so its paths must already be
                valid (e.g. the frames produced by SplitGenerator). The normalisation statistics
                (s2_normalization_stats.json) are read from the data folder, i.e. the parent
                folder of the Sentinel-2 files.
            sel_time (list, optional): List of time steps to select from the dataset.
            sel_band (list, optional): List of bands to select from the dataset.
            sel_task (str, optional): Task type for the dataset.
                - "lvl0_bic": binary label data (habitat vs. non-habitat) - background-inclusive classification
                - "lvl1_bic": habitat supersets (1st level), max. 9 classes - background-inclusive classification
                - "lvl1_bec": habitat supersets (1st level), max. 9 classes - background-exclusive classification
                - "lvl2_bic": habitat supersets (2nd level), max. 33 classes - background-inclusive classification
                - "lvl2_bec": habitat supersets (2nd level), max. 33 classes - background-exclusive classification
                - "lvl4_bic": habitats (full 4-digit codes), max. 233 classes - background-inclusive classification
                - "lvl4_bec": habitats (full 4-digit codes), max. 233 classes - background-exclusive classification
            normalize (bool, optional): Whether to standardise the input data channel-wise. Defaults to True.
            transform (callable, optional): A function/transform to apply to the images
            use_cache (str, optional): Whether to cache data in memory to speed up data loading. Defaults to None.
                None:   no caching
                "cpu":  cache in CPU memory (mind required RAM)
                "disk": cache on disk using .pt files (requires disk space)
            cache_dir (str, optional): Directory to store disk cache files if use_cache is set to "disk".
            read_covars (bool, optional): Whether __getitem__ also returns the label covariates (see COVAR_ORDER) as
                a (n_covar, H, W) float32 tensor with NaN where undefined. The covariates are always read and cached
                regardless; this only controls the return signature (4-tuple instead of 3-tuple), so it can
                be toggled on an existing instance. Defaults to False.
        """
        # parse input parameters
        if isinstance(meta_file, (str, os.PathLike)):
            # paths in the CSV are relative to the CSV's folder
            self.meta_file = resolve_paths(
                pd.read_csv(meta_file), data_dir_of(meta_file)
            )
        else:
            self.meta_file = meta_file
        self.sel_time = sel_time
        self.sel_band = sel_band
        self.sel_task = sel_task
        self.transform = transform
        self.normalize = normalize
        self.read_covars = read_covars

        # init caching
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        if self.use_cache not in (None, "cpu", "disk"):
            raise ValueError("cache must be one of: None, 'cpu', 'disk'")
        if self.use_cache == "disk" and self.cache_dir is None:
            raise ValueError("cache_dir must be specified when use_cache is 'disk'")

        if self.use_cache == "cpu":
            self._cache = {}

        if self.use_cache == "disk":
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # define label map
        self._get_label_map()

        # check that selected time steps and bands are valid
        if self.sel_time is not None:
            if not all(t in self.TIME_ORDER for t in self.sel_time):
                raise ValueError(
                    f"Invalid time steps selected. \n"
                    f"Valid options are: {self.TIME_ORDER}\n"
                    f"Selected options are: {self.sel_time}"
                )
        else:
            self.sel_time = self.TIME_ORDER
        if self.sel_band is not None:
            if not all(b in self.BAND_ORDER for b in self.sel_band):
                raise ValueError(
                    f"Invalid bands selected. \n"
                    f"Valid options are: {self.BAND_ORDER}\n"
                    f"Selected options are: {self.sel_band}"
                )
        else:
            self.sel_band = self.BAND_ORDER

        # convert time steps & bands to indices for selection later on
        self.i_band = [self.BAND_ORDER.index(b) for b in self.sel_band]
        self.i_time = [self.TIME_ORDER.index(t) for t in self.sel_time]

        # derive flattened global channel index
        # band-major, time-minor ordering, i.e. channel = band_idx * n_times + time_idx
        self.i_channel = np.asarray(
            [
                band_idx * len(self.TIME_ORDER) + time_idx
                for band_idx in self.i_band
                for time_idx in self.i_time
            ],
            dtype=np.int64,
        )

        # useful for checking the resulting order
        self.channel_names = [
            f"{self.BAND_ORDER[band_idx]}__{self.TIME_ORDER[time_idx]}"
            for band_idx in self.i_band
            for time_idx in self.i_time
        ]

        # fetch data set statistics for normalization
        if self.normalize:
            data_dir = os.path.dirname(
                os.path.dirname(self.meta_file["path_input"].iloc[0])
            )
            with open(os.path.join(data_dir, "s2_normalization_stats.json")) as f:
                norm_stats = json.load(f)
                self.norm_stats = A.Normalize(
                    mean=list(np.array(norm_stats["mean"])[self.i_channel]),
                    std=list(np.array(norm_stats["std"])[self.i_channel]),
                    max_pixel_value=1.0,
                )
        else:
            self.norm_stats = None

    def __len__(self):
        return len(self.meta_file)

    def _get_raw_item(self, idx):
        # load input data
        row = self.meta_file.iloc[idx]
        in_da = xr.open_dataarray(row["path_input"])
        # subset & convert from (band, time, y, x) to (z, y, x)
        target_dims = ("band", "time", "y", "x")
        in_da = in_da.transpose(*target_dims)
        in_da = in_da.isel(band=self.i_band, time=self.i_time)
        in_arr = in_da.values.astype(np.float32)
        in_arr = in_arr.reshape(-1, *in_arr.shape[2:])
        # apply normalization
        in_arr /= 10000
        if self.normalize:
            in_arr = self.norm_stats(image=np.transpose(in_arr, (1, 2, 0)))["image"]
            in_arr = np.transpose(in_arr, (2, 0, 1))
        # load & encode labels
        label_ds = xr.open_dataset(row["path_label"], mask_and_scale=False)
        aoi = label_ds["aoi_mask"].values
        label_arr = self.label_map_lut[label_ds["lrt_code"].values.astype(int)]
        # insert ignore values for outside AoI and optionally background
        label_arr[aoi == 0] = -99
        if self.sel_task.endswith("_bec"):
            label_arr[label_arr == 0] = -99
        # read covariates, NaN outside the AoI
        covar_arr = np.stack(
            [self._read_covar(label_ds[c]) for c in self.COVAR_ORDER]
        ).astype(np.float32)
        covar_arr[:, np.nan_to_num(aoi) == 0] = np.nan
        # convert to torch tensors
        return (
            torch.from_numpy(in_arr),
            torch.from_numpy(label_arr),
            torch.from_numpy(covar_arr),
        )

    @staticmethod
    def _read_covar(da):
        """One covariate as float32 with NaN for missing values."""
        arr = da.values.astype(np.float32)
        fill = da.attrs.get("_FillValue", da.encoding.get("_FillValue"))
        if fill is not None and np.isreal(fill) and np.isfinite(fill):
            arr[arr == np.float32(fill)] = np.nan
        return arr

    def __getitem__(self, idx):
        # load data, labels & covariates, optionally using caching
        if self.use_cache is None:
            in_arr, label_arr, covar_arr = self._get_raw_item(idx)
        elif self.use_cache == "cpu":
            if idx not in self._cache:
                self._cache[idx] = self._get_raw_item(idx)
            in_arr, label_arr, covar_arr = self._cache[idx]
        elif self.use_cache == "disk":
            in_arr, label_arr, covar_arr = self._load_or_build_disk(
                idx,
                lambda: self._get_raw_item(idx),
            )
        else:
            raise ValueError(f"Invalid cache option: {self.use_cache}")
        if self.transform:
            mask = np.concatenate(
                [
                    label_arr.numpy()[..., None].astype(np.float32),
                    np.transpose(covar_arr.numpy(), (1, 2, 0)),
                ],
                axis=-1,
            )
            aug = self.transform(
                image=np.transpose(in_arr.numpy(), (1, 2, 0)),
                mask=mask,
            )
            in_arr = torch.from_numpy(np.transpose(aug["image"], (2, 0, 1)))
            label_arr = torch.from_numpy(aug["mask"][..., 0].astype(np.int64))
            covar_arr = torch.from_numpy(
                np.ascontiguousarray(np.transpose(aug["mask"][..., 1:], (2, 0, 1)))
            )
        if self.read_covars:
            return in_arr, label_arr, covar_arr, torch.tensor(idx).long()
        return in_arr, label_arr, torch.tensor(idx).long()

    def plot(self, idx=None, seed=42):
        """
        Plot samples from the dataset.

        Args:
            idx (int, optional): Index of the sample to plot. If None, a random sample is selected.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        # get rgb version of dataset
        rgb_instance = self.__class__(
            meta_file=self.meta_file,
            sel_time=self.sel_time,
            sel_band=["B4", "B3", "B2"],
            sel_task=self.sel_task,
            normalize=False,
            transform=self.transform,
        )

        # define sample idxs
        if idx is not None:
            sample_idx = idx
        else:
            np.random.seed(seed)
            sample_idx = np.random.choice(np.arange(len(self)), 1)[0]

        # fetch data and labels
        print(f"Plotting sample {sample_idx} from dataset as RGB timeseries...")
        print(self.meta_file.iloc[sample_idx])
        data, label, _ = rgb_instance[sample_idx]

        # define plot layout
        ncols = min(len(rgb_instance.i_time), 8)
        nrows = int(np.ceil(len(rgb_instance.i_time) / ncols)) + 1
        t_names = (
            rgb_instance.TIME_ORDER
            if rgb_instance.sel_time is None
            else rgb_instance.sel_time
        )
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(3 * ncols, 3 * nrows),
            constrained_layout=True,
        )

        # helper function to clean up axes
        def clean_map_axis(ax, title):
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        # fetch cmap for habitat labels
        if rgb_instance.sel_task.endswith("_bec"):
            habitat_cmap = ListedColormap(rgb_instance.label_cmap.colors[1:])
            habitat_norm = BoundaryNorm(
                boundaries=np.arange(0.5, rgb_instance.labels_unique[-1] + 1.5),
                ncolors=habitat_cmap.N,
            )
        else:
            habitat_cmap = rgb_instance.label_cmap
            habitat_norm = BoundaryNorm(
                boundaries=np.arange(-0.5, rgb_instance.labels_unique[-1] + 1.5),
                ncolors=habitat_cmap.N,
            )

        habitat_cmap.set_bad("black")
        habitat_cmap.set_under("black")
        habitat_cmap.set_over("black")

        # create legend
        habitat_ints = (
            pd.Series(label.flatten()).value_counts().sort_index().index.values
        )
        habitat_lut = rgb_instance.label_map_df[
            rgb_instance.label_map_df["target_encoding"].isin(habitat_ints)
        ]
        habitat_patches = [
            mpatches.Patch(
                facecolor=row["target_color"],
                edgecolor="black",
                label=str(row["target_semantics"]),
            )
            for _, row in habitat_lut.drop_duplicates("target_encoding").iterrows()
        ]

        # plot label data & legend
        axs[0, 0].imshow(
            label,
            cmap=habitat_cmap,
            norm=habitat_norm,
            interpolation="nearest",
        )
        legend = axs[0, 0].legend(
            handles=habitat_patches,
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            borderaxespad=0,
            fontsize=12,
            frameon=False,
            ncol=min(2 * (ncols - 1), 5),
        )
        legend.set_in_layout(False)

        clean_map_axis(axs[0, 0], "habitats")
        for col in np.arange(1, ncols):
            axs[0, col].axis("off")

        # plot input satellite data
        arr_reshaped = data.reshape(
            len(rgb_instance.i_band), len(rgb_instance.i_time), *data.shape[1:]
        )
        for i, t in enumerate(t_names):
            row = (i // ncols) + 1
            col = i % ncols
            t_arr = np.array(arr_reshaped[:, i, :, :]).transpose(1, 2, 0)
            t_arr = 5 * np.clip(t_arr, 0, 0.2)
            axs[row, col].imshow(t_arr, interpolation="nearest")
            clean_map_axis(axs[row, col], t)

        # return the figure and axes for further manipulation if needed
        return fig, axs

    def _get_label_map(self):
        """
        Defines a mapping of original label encodings to their target encodings.
        Note: Background class is always encoded as 0, regardless of the task.
        This means that the mapping and therefore model architecture for the settings
        background-inclusive (bic) and background-exclusive (bec) looks the same.
        The differentiation between settings is done...
            * either during __getitem__ & via ignore_index for the loss definition
              and the metrics computation for semantic segmentation data
            * or upon TimeSeriesDataset generation for pixelwise models, where the
              background-exclusive setting removes all background pixels from the dataset.
        """
        # read legend file & add original encoding
        lrt_legend_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "lrt_legend.csv",
        )
        lrt_legend = pd.read_csv(lrt_legend_path)
        lrt_legend.columns = ["lrt_code", "lrt_name", "lrt_color"]
        zero_row = pd.DataFrame(
            {
                "lrt_code": [0],
                "lrt_name": ["Hintergrund (kein Habitat)"],
                "lrt_color": ["#ffffff"],
            }
        )
        lrt_legend = pd.concat([zero_row, lrt_legend], ignore_index=True)
        lrt_legend.insert(0, "encoded_int", lrt_legend.index.values)

        # define different level mappings
        lgds = {}
        lgds["lvl0"] = lrt_legend.copy()
        ts = lgds["lvl0"].encoded_int.apply(lambda x: 0 if x == 0 else 1)
        lgds["lvl0"].insert(1, "target_semantics", ts)
        te = LabelEncoder().fit_transform(ts)
        lgds["lvl0"].insert(2, "target_encoding", te)
        tcolors = lgds["lvl0"].encoded_int.apply(
            lambda x: "#ffffff" if x == 0 else "#61ab76"
        )
        lgds["lvl0"].insert(3, "target_color", tcolors)

        lgds["lvl1"] = lrt_legend.copy()
        ts = lgds["lvl1"].lrt_code.apply(lambda x: int(str(x)[:1]))
        lgds["lvl1"].insert(1, "target_semantics", ts)
        te = LabelEncoder().fit_transform(ts)
        lgds["lvl1"].insert(2, "target_encoding", te)
        tcolors = (
            lgds["lvl1"]
            .groupby("target_encoding")
            .apply(lambda x: x.iloc[(len(x)) // 2], include_groups=False)
            .reset_index(drop=True)
            .merge(
                lgds["lvl1"], on="target_semantics", how="right", suffixes=("", "_orig")
            )["lrt_color"]
        )
        lgds["lvl1"].insert(3, "target_color", tcolors)

        lgds["lvl2"] = lrt_legend.copy()
        ts = lgds["lvl2"].lrt_code.apply(lambda x: int(str(x)[:2]))
        lgds["lvl2"].insert(1, "target_semantics", ts)
        te = LabelEncoder().fit_transform(ts)
        lgds["lvl2"].insert(2, "target_encoding", te)
        tcolors = (
            lgds["lvl2"]
            .groupby("target_encoding")
            .apply(lambda x: x.iloc[(len(x)) // 2], include_groups=False)
            .reset_index(drop=True)
            .merge(
                lgds["lvl2"], on="target_semantics", how="right", suffixes=("", "_orig")
            )["lrt_color"]
        )
        lgds["lvl2"].insert(3, "target_color", tcolors)

        lgds["lvl3"] = lrt_legend.copy()
        ts = lgds["lvl3"].lrt_code.apply(lambda x: str(x)[:3])
        lgds["lvl3"].insert(1, "target_semantics", ts)
        te = LabelEncoder().fit_transform(ts)
        lgds["lvl3"].insert(2, "target_encoding", te)
        tcolors = (
            lgds["lvl3"]
            .groupby("target_encoding")
            .apply(lambda x: x.iloc[(len(x)) // 2], include_groups=False)
            .reset_index(drop=True)
            .merge(
                lgds["lvl3"], on="target_semantics", how="right", suffixes=("", "_orig")
            )["lrt_color"]
        )
        lgds["lvl3"].insert(3, "target_color", tcolors)

        lgds["lvl4"] = lrt_legend.copy()
        ts = lgds["lvl4"].lrt_code.apply(lambda x: str(x)[:4])
        lgds["lvl4"].insert(1, "target_semantics", ts)
        te = LabelEncoder().fit_transform(ts)
        lgds["lvl4"].insert(2, "target_encoding", te)
        tcolors = (
            lgds["lvl4"]
            .groupby("target_encoding")
            .apply(lambda x: x.iloc[(len(x)) // 2], include_groups=False)
            .reset_index(drop=True)
            .merge(
                lgds["lvl4"], on="target_semantics", how="right", suffixes=("", "_orig")
            )["lrt_color"]
        )
        lgds["lvl4"].insert(3, "target_color", tcolors)

        # choose label map
        self.label_map_df = lgds[self.sel_task.split("_")[0]]

        # label mapping as look-up-table for on-the-fly conversion
        orig = self.label_map_df["encoded_int"].values
        target = self.label_map_df["target_encoding"].values
        max_val = self.label_map_df["encoded_int"].max()
        self.label_map_lut = np.zeros(max_val + 1, dtype=int)
        self.label_map_lut[orig] = target

        # other attributes
        self.labels_unique = self.label_map_df["target_encoding"].unique()
        hex_colors = self.label_map_df.drop_duplicates("target_encoding")[
            "target_color"
        ].tolist()
        self.label_cmap = ListedColormap(hex_colors)

    def _disk_cache_key(self, idx):
        row = self.meta_file.iloc[idx]

        key_dict = {
            "dataset": self.__class__.__name__,
            "idx": int(row.name),
            "path_input": row["path_input"],
            "path_label": row["path_label"],
            "sel_band": self.sel_band,
            "sel_time": self.sel_time,
            "sel_task": self.sel_task,
            "normalize": self.normalize,
            "covars": self.COVAR_ORDER,
        }

        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        return hashlib.sha1(key_str.encode()).hexdigest()

    def _disk_cache_path(self, key):
        # set up disk cache directory for pt files
        base = os.path.join(self.cache_dir, ".pt_cache")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{key}.pt")

    def _load_or_build_disk(self, idx, original_fn):
        key = self._disk_cache_key(idx)
        path = self._disk_cache_path(key)

        if os.path.exists(path):
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )

        # Write the data in float32 to a temporary file first
        data = original_fn()
        data = (
            data[0].to(torch.float32),
            data[1],
            data[2].to(torch.float32),
        )
        tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"

        # Atomically publish the completed cache entry
        try:
            torch.save(data, tmp, _use_new_zipfile_serialization=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        return data
