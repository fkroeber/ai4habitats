import datetime
import hashlib
import json
import numpy as np
import os
import shutil
import tempfile
import tqdm
import warnings
import uuid
import xarray as xr

from models.patchwise.data import S2_Dataset
from models.patchwise.train import Trainer
from models.pixelwise.sharded import (
    MANIFEST,
    ShardedArray,
    ShardWriter,
    dir_size,
    load_field,
    write_manifest,
)
from models.pixelwise.utils import extract_pickled, save_pickled


class TimeSeriesDatasetCreator:

    IGNORE_INDEX = -99

    def __init__(
        self,
        meta_file,
        sel_time=None,
        sel_band=None,
        sel_task="lvl0_bic",
        split_strategy="random",
        split_subset=1.0,
        split_trainsize=0.7,
        split_fold=0,
        split_kwargs=None,
        use_cache=None,
        cache_dir=None,
        seed=42,
        sample_size=None,
        read_covars=False,
        shard_size_mb=None,
    ):
        """
        TimeSeriesDatasetCreator is a class to sample temporal pixel stacks from a given patch-based dataset.
        Using the Trainer & Dataset classes from the patchwise model, it is ensured that the sampled pixel stacks
        are consistent with the patch-based datasets.

        Args:
            meta_file (str): Patch metadata CSV (see S2_Dataset); relative paths are
                interpreted relative to the CSV's folder.
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
            split_strategy (str, optional): Strategy for splitting the dataset into train/val/test.
                Defaults to "random".
            split_trainsize (float, optional): Target share assigned to training. The remaining share is divided
                equally between validation and test. Defaults to 0.7.
            split_fold (int, optional): Ignored, kept for backward compatibility; splits differ
                by `seed` only. Defaults to 0.
            split_subset (float, optional): Fraction of metadata rows retained before splitting, in (0, 1].
                Defaults to 1.0.
            split_kwargs (dict, optional): Additional keyword arguments for the split strategy. Default None.
            use_cache (str, optional): Whether to cache data in memory to speed up data loading. Defaults to None.
                None:   no caching
                "disk": cache on disk using .pkl files (requires disk space)
            cache_dir (str, optional): Directory to store disk cache files if use_cache is set to "disk".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            sample_size (int): Absolute number of samples to be drawn from the dataset for the train fold.
                Validation and test fold samples will be drawn proportionally as defined in the dataloader.
                If None, the entire dataset will be used.
            read_covars (bool, optional): Whether to sample the label covariates alongside the pixel stacks.
                If True, each split gains a "covar" array of shape [samples, n_covar]. Defaults to False.
            shard_size_mb (float, optional): If set, the pixel stacks are streamed to disk as .npy shards of
                roughly this size (in MB of "input" data) instead of being assembled in memory, and "input"
                is returned as a lazily loaded ShardedArray. Requires use_cache="disk". The shard size does
                not change the sampled data and is therefore not part of the cache key. Defaults to None
                (single in-memory array, cached as one .pkl file).

        Returns:
            dict: A dictionary containing the sampled pixel stacks and metadata.

            The created pixel stacks will be saved as .pkl files with the following structure:
                meta             -> dictionary with metadata (input arguments used for sampling)
                data             -> dictionary with data arrays
                |-- train
                    |--  input     -> array of shape [sample_size, bands, times] with pixel stacks
                    |--  label     -> array of shape [sample_size] with labels
                |-- val
                    |--  input     -> array of shape [sample_size * val_ratio, bands, times] with pixel stacks
                    |--  label     -> array of shape [sample_size] with labels
                |-- test
                    |--  input     -> array of shape [sample_size * test_ratio, bands, times] with pixel stacks
                    |--  label     -> array of shape [sample_size] with labels

            where
                * sample_size is the absolute number of pixels to be drawn from the dataset
                * val_ratio is the ratio of validation samples to training samples as defined in the dataloader
                * test_ratio is the ratio of test samples to training samples as defined in the dataloader
                * bands is the number of spectral channels
                * times is the number of time steps

            With shard_size_mb set, the same dictionary is returned, but "input" is a ShardedArray (lazy,
            numpy-like row indexing) and the data live in <cache_dir>/.pixel_cache/<key>.shards/:
                manifest.json               -> layout, row counts, dtypes, parameters
                meta.pkl                    -> the "meta" dictionary
                <split>/<field>_<i>.npy     -> shards of input/label/(covar/tile/coords_x/coords_y)
        """
        # parse arguments
        self.meta_file = meta_file
        self.sel_time = sel_time
        self.sel_band = sel_band
        self.sel_task = sel_task
        self.split_strategy = split_strategy
        self.split_subset = split_subset
        self.split_trainsize = split_trainsize
        self.split_fold = split_fold
        self.split_kwargs = split_kwargs or {}
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.seed = seed
        self.sample_size = sample_size
        self.read_covars = read_covars
        self.shard_size_mb = shard_size_mb
        # check cache dir & create if it doesn't exist
        if self.use_cache not in (None, "disk"):
            raise ValueError("use_cache must be one of: None, 'disk'")
        if self.shard_size_mb is not None:
            if self.use_cache != "disk":
                raise ValueError(
                    "shard_size_mb requires use_cache='disk', sharded data live on disk"
                )
            if not self.shard_size_mb > 0:
                raise ValueError("shard_size_mb must be > 0")
        if self.use_cache == "disk":
            if self.cache_dir is None:
                self.cache_dir = os.path.dirname(os.path.abspath(self.meta_file))
            os.makedirs(self.cache_dir, exist_ok=True)
        # initialise return object
        self.fetched = {"meta": {}, "data": {}}

    def _tmp_dir(self):
        """Scratch save_dir for the internal Trainer (it writes a config.json)."""
        if getattr(self, "_tmp", None) is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="ai4habitats_tsdc_")
        return self._tmp

    def run(self):
        rng = np.random.default_rng(self.seed)

        # define default configuration to fetch a dataloader from the Trainer class
        trainer_cfg = {
            "meta_file": self.meta_file,
            "sel_time": self.sel_time,
            "sel_band": self.sel_band,
            "sel_task": self.sel_task,
            "split_strategy": self.split_strategy,
            "split_subset": self.split_subset,
            "split_kwargs": self.split_kwargs,
            "split_trainsize": self.split_trainsize,
            "split_fold": self.split_fold,
            "train_augmentations": False,
            "use_cache": self.use_cache,
            "seed": self.seed,
            "verbose": False,
            "save_dir": self._tmp_dir().name,
            "wandb_project": None,
        }

        # cache lookup before any expensive work
        sharded = self.shard_size_mb is not None
        cache_path = None
        if self.use_cache == "disk":
            key = self._cache_key()
            cache_path = self._shard_dir(key) if sharded else self._cache_path(key)
            if sharded and os.path.isfile(os.path.join(cache_path, MANIFEST)):
                print(f"Loading sharded cache from {cache_path}...")
                self.fetched = self._load_shards(cache_path)
                return self.fetched
            if not sharded and os.path.exists(cache_path):
                print(f"Loading cache from {cache_path}...")
                self.fetched = extract_pickled(cache_path)
                return self.fetched

        # cache miss: sample from scratch
        self.trainer = Trainer(**trainer_cfg)
        self.trainer.prep_model()

        # set read_covars flag in the datasets so they return covariates if requested
        splits = list(self.trainer.dataloaders.keys())
        for split in splits:
            self.trainer.dataloaders[split].dataset.read_covars = self.read_covars

        # get relevant parameters of dataset
        exp_ds = self.trainer.dataloaders["test"].dataset
        covars = exp_ds.COVAR_ORDER
        n_bands, n_time = len(exp_ds.i_band), len(exp_ds.i_time)
        tile_hw = exp_ds[0][0].shape[1:]
        pxls_per_tile = np.prod(tile_hw)

        # fetch how many eligible pixels each tile holds
        available = {}
        caps = {}
        for split in splits:
            caps[split] = self._count_valid_pixels(split, pxls_per_tile)
            available[split] = int(caps[split].sum())

        # resolve per-split targets depending on requested sample_size
        targets = {}
        for split in splits:
            if self.sample_size is None:
                want = available[split]
            else:
                share = available[split] / available["train"]
                want = int(share * self.sample_size)
            if want > available[split]:
                warnings.warn(
                    f"Requested {want} {split} pixels but only {available[split]} "
                    f"eligible pixels exist (sel_task='{self.sel_task}'); "
                    f"sampling {available[split]} instead."
                )
                want = available[split]
            if want == 0:
                warnings.warn(
                    f"Resolved sample size for split '{split}' is 0 - the returned "
                    "arrays will be empty, which will fail downstream model fitting."
                )
            targets[split] = want

        # fields sampled per pixel: name -> (shape of one row, dtype)
        fields = {
            "input": ((n_bands, n_time), "float32"),
            "label": ((), "int16"),
        }
        if self.read_covars:
            fields["covar"] = ((len(covars),), "float32")
            fields["tile"] = ((), "int32")
            fields["coords_x"] = ((), "float32")
            fields["coords_y"] = ((), "float32")

        # query number of pixels to be sampled from each tile
        pxls_to_sample = {}
        for i, split in enumerate(splits):
            pxls_to_sample[split] = self._pixels_per_tile(
                targets[split], caps[split], seed=self.seed + i
            )

        # metadata initialization
        meta = {
            k: v for k, v in self.trainer.dataloaders["train"].dataset.__dict__.items()
        }
        meta["covar_names"] = list(covars) if self.read_covars else []

        if not sharded:
            # in-memory: preallocate full arrays, fill tile by tile
            arrs = {
                split: {
                    name: np.zeros((targets[split], *shape), dtype=dtype)
                    for name, (shape, dtype) in fields.items()
                }
                for split in splits
            }
            offsets = {
                split: np.concatenate(([0], np.cumsum(counts)))
                for split, counts in pxls_to_sample.items()
            }
            for split in splits:
                for i, chunk in self._iter_tile_samples(
                    split, pxls_to_sample[split], rng, n_bands, n_time, covars, tile_hw
                ):
                    lo, hi = offsets[split][i], offsets[split][i + 1]
                    for name, values in chunk.items():
                        arrs[split][name][lo:hi] = values

            self.fetched["meta"] = meta
            self.fetched["data"] = arrs

            # write cache
            if cache_path is not None:
                self._atomic_save(cache_path, self.fetched)
                self._write_manifest(cache_path)
            return self.fetched

        # sharded: stream tile samples into fixed-size shards in a temporary
        # directory, which is renamed atomically once it is complete
        row_bytes = n_bands * n_time * np.dtype("float32").itemsize
        rows_per_shard = max(1, int(self.shard_size_mb * 2**20 // row_bytes))
        tmp_dir = f"{cache_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            groups = {}
            for split in splits:
                writer = ShardWriter(tmp_dir, split, fields, rows_per_shard)
                for _, chunk in self._iter_tile_samples(
                    split, pxls_to_sample[split], rng, n_bands, n_time, covars, tile_hw
                ):
                    writer.append(**chunk)
                groups[split] = writer.close()
                if groups[split]["n_rows"] != targets[split]:
                    raise RuntimeError(
                        f"wrote {groups[split]['n_rows']} {split} rows, "
                        f"expected {targets[split]}"
                    )
            save_pickled(os.path.join(tmp_dir, "meta.pkl"), meta)
            write_manifest(
                tmp_dir,
                groups,
                extra={
                    "key": self._cache_key(),
                    "params": self._cache_params(),
                    "meta_file": os.path.abspath(self.meta_file),
                    "rows_per_shard": rows_per_shard,
                    "created_utc": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(timespec="seconds"),
                    "size_bytes": dir_size(tmp_dir),
                },
            )
            try:
                # atomic; fails if target exists
                os.replace(tmp_dir, cache_path)
            except OSError:
                if not os.path.isfile(os.path.join(cache_path, MANIFEST)):
                    raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.fetched = self._load_shards(cache_path)
        return self.fetched

    # ------------------------------------------------------------------ #
    # sampling helpers
    # ------------------------------------------------------------------ #

    def _iter_tile_samples(self, split, counts, rng, n_bands, n_time, covars, tile_hw):
        """Yield (tile index, dict of sampled rows per field) for every tile of `split`.

        Tiles are visited in order and `rng` is consumed exactly as before, so the in-memory
        and the sharded storage mode produce identical samples for the same seed.
        """
        ds = self.trainer.dataloaders[split].dataset
        for i, k in tqdm.tqdm(
            enumerate(counts), total=len(counts), desc=f"Sampling {split} pixels"
        ):
            if k == 0:
                continue
            # read the tile
            if self.read_covars:
                x, y, c, _ = ds[i]
            else:
                x, y, _ = ds[i]
            # reshape into (n_bands, n_time, -1) assumes band-major channel layout
            x = np.asarray(x).reshape(n_bands, n_time, -1)
            y = np.asarray(y).reshape(-1)
            # drop ignore pixels
            pool = np.flatnonzero(y != self.IGNORE_INDEX)
            sel = rng.choice(pool, size=k, replace=False)
            chunk = {"input": x[:, :, sel].transpose(2, 0, 1), "label": y[sel]}
            # if covariates are requested, reshape and store them as well
            if self.read_covars:
                c = np.asarray(c).reshape(max(1, len(covars)), -1)
                chunk["covar"] = c[:, sel].T
                chunk["tile"] = np.full(k, i, dtype="int32")
                rows, cols = np.unravel_index(sel, tile_hw)
                with xr.open_dataarray(ds.meta_file["path_input"].iloc[i]) as da:
                    if (da.sizes["y"], da.sizes["x"]) != tuple(tile_hw):
                        raise ValueError(
                            f"Nc file {ds.meta_file['path_input'].iloc[i]}: "
                            f"Does not match tile shape {tile_hw}."
                        )
                    chunk["coords_x"] = da["x"].values[cols]
                    chunk["coords_y"] = da["y"].values[rows]
            yield i, chunk

    def _count_valid_pixels(self, split, pxls_per_tile):
        """Count pixels with a label other than IGNORE_INDEX for every tile of `split`.

        Doubles as a check that all tiles really share the same pixel count, which the
        rest of the sampling logic assumes.
        """
        ds = self.trainer.dataloaders[split].dataset
        counts = np.zeros(len(ds), dtype=np.int64)
        for i in tqdm.tqdm(
            range(len(ds)), total=len(ds), desc=f"Counting valid {split} pixels"
        ):
            if ds.read_covars:
                _, y, _, _ = ds[i]
            else:
                _, y, _ = ds[i]
            y = np.asarray(y).reshape(-1)
            if y.size != pxls_per_tile:
                raise ValueError(
                    f"Tile {i} of split '{split}' holds {y.size} pixels, expected "
                    f"{pxls_per_tile}. Ragged tiles are not supported."
                )
            counts[i] = int((y != self.IGNORE_INDEX).sum())
        return counts

    def _pixels_per_tile(self, total, caps, seed=None):
        """
        Distribute `total` pixels over the tiles as evenly as possible, never taking
        more than `caps[i]` pixels from tile i. Returns an int array of the same
        length as `caps` summing exactly to `total`.

        `caps` may be a scalar (uniform cap) or a per-tile array; a cap of 0 excludes
        the tile entirely.
        """
        caps = np.asarray(caps, dtype=np.int64)
        if caps.ndim == 0:
            raise ValueError(
                "caps must be a per-tile array; pass np.full(n_tiles, cap)"
            )
        n_tiles = caps.size
        budget = int(caps.sum())
        if total > budget:
            raise ValueError(
                f"Cannot draw {total} pixels from {n_tiles} tiles offering "
                f"{budget} eligible pixels."
            )
        rng = np.random.default_rng(seed=seed)
        counts = np.zeros(n_tiles, dtype=np.int64)
        # exclude zero-cap tiles up front
        active = caps > 0
        remaining = int(total)
        while remaining > 0:
            idx = np.flatnonzero(active)
            if idx.size == 0:
                raise RuntimeError(
                    f"Ran out of eligible tiles with {remaining} pixels still to place."
                )
            base, rem = divmod(remaining, idx.size)
            add = np.full(idx.size, base, dtype=np.int64)
            # scatter the leftover pixels randomly so no tile is systematically favoured
            if rem:
                add[rng.choice(idx.size, rem, replace=False)] += 1
            counts[idx] += add
            # clip tiles that overshot their cap and redistribute their excess
            over = counts > caps
            remaining = int((counts[over] - caps[over]).sum())
            counts[over] = caps[over]
            active &= ~over
        return counts

    # ------------------------------------------------------------------ #
    # caching & metadata helpers
    # ------------------------------------------------------------------ #

    def _atomic_save(self, path, obj):
        """Write under a temp name and rename, so a crash never leaves a
        half-written cache file that a later run would treat as valid."""
        tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.pkl"
        try:
            save_pickled(tmp, obj)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _cache_params(self):
        """The parameters that determine the sampled output.

        Only fields that change the pixel stacks are included: the S2_Dataset-defining
        params, the SplitGenerator params, & `sample_size`. Other params are excluded,
        since they don't affect the data.
        """
        return {
            "meta_file": os.path.basename(self.meta_file),
            "sel_time": self.sel_time,
            "sel_band": self.sel_band,
            "sel_task": self.sel_task,
            "split_strategy": self.split_strategy,
            "split_subset": self.split_subset,
            "split_trainsize": self.split_trainsize,
            "split_fold": self.split_fold,
            "split_kwargs": self.split_kwargs,
            "seed": self.seed,
            "sample_size": self.sample_size,
            "read_covars": self.read_covars,
        }

    def _cache_key(self):
        """Stable hash over `_cache_params()`."""
        key_str = json.dumps(self._cache_params(), sort_keys=True, default=str)
        return hashlib.sha1(key_str.encode()).hexdigest()

    def _cache_path(self, key):
        base = os.path.join(self.cache_dir, ".pixel_cache")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{key}.pkl")

    def _shard_dir(self, key):
        base = os.path.join(self.cache_dir, ".pixel_cache")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{key}.shards")

    @staticmethod
    def _load_shards(root):
        """Open a sharded cache: "input" stays lazy, all other fields are small and
        are loaded into memory."""
        with open(os.path.join(root, MANIFEST)) as f:
            groups = json.load(f)["groups"]
        data = {}
        for split, entry in groups.items():
            data[split] = {
                field: (
                    ShardedArray.from_manifest(root, split, field)
                    if field == "input"
                    else load_field(root, split, field)
                )
                for field in entry["fields"]
            }
        return {"meta": extract_pickled(os.path.join(root, "meta.pkl")), "data": data}

    def _write_manifest(self, cache_path):
        """Write a human-readable sidecar next to the pickle recording which
        parameter combination produced it. Written *after* the pickle, so a
        manifest never advertises data that isn't there."""
        manifest = {
            "key": self._cache_key(),
            "params": self._cache_params(),
            "meta_file": os.path.abspath(self.meta_file),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "size_bytes": os.path.getsize(cache_path),
        }
        path = self._manifest_path(cache_path)
        tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.json"
        try:
            with open(tmp, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @staticmethod
    def _manifest_path(cache_path):
        return f"{os.path.splitext(cache_path)[0]}.json"


class TimeSeriesDataset:
    def __init__(self, ds, in_memory=False):
        """
        TimeSeriesDataset is a class to load time series data from pre-sampled pixel stacks.
        Labels, covariates and coordinates are always loaded into memory. The pixel stacks
        (X_*) are loaded into memory if they were created in memory; if they were created as
        shards (TimeSeriesDatasetCreator(shard_size_mb=...)), X_* are lazily loaded
        ShardedArrays, which the (patched) sktime deep learning classifiers read batch-wise.
        Pass in_memory=True to materialise them anyway (e.g., for non-deep-learning sktime
        classifiers, which require full numpy arrays).

        The intialised DataSet object will have the following attributes:
            X_train (np.ndarray): Training data of shape [samples, band, time].
            y_train (np.ndarray): Training labels of shape [samples].
            X_val (np.ndarray): Validation data of shape [samples, band, time].
            y_val (np.ndarray): Validation labels of shape [samples].
            X_test (np.ndarray): Test data of shape [samples, band, time].
            y_test (np.ndarray): Test labels of shape [samples].
            labels_dict (dict): Dictionary mapping unique labels to their indices.
            labels_str (str): Semantic names of the classes.

        Args:
            ds (dict): Dictionary containing the pre-sampled pixel stacks and metadata.
            in_memory (bool, optional): Materialise lazily loaded pixel stacks as numpy
                arrays. Defaults to False.
        """
        self.ds = ds
        self.in_memory = in_memory
        self.load_data()

    def get_covar(self, name, split="test"):
        """One covariate as a 1-D array aligned with X_{split} / y_{split}."""
        covar = getattr(self, f"covar_{split}", None)
        if covar is None or name not in self.covar_names:
            return None
        return covar[:, self.covar_names.index(name)]

    def load_data(self):
        self.meta = self.ds["meta"]
        self.covar_names = list(self.meta.get("covar_names") or [])
        # extract splits, dropping any pixel without a valid target
        ignore = TimeSeriesDatasetCreator.IGNORE_INDEX
        for split in ("train", "val", "test"):
            fields = self.ds["data"][split]
            valid = fields["label"] != ignore
            rows = None if valid.all() else np.flatnonzero(valid)
            for name in ("input", "label", "covar", "tile", "coords_x", "coords_y"):
                attr = {"input": "X", "label": "y"}.get(name, name)
                setattr(
                    self, f"{attr}_{split}", self._select_rows(fields.get(name), rows)
                )
            if self.in_memory and isinstance(getattr(self, f"X_{split}"), ShardedArray):
                setattr(self, f"X_{split}", np.asarray(getattr(self, f"X_{split}")))
        # create mapping -> given value: encoded value for training the classifier
        self.labels_dict = {
            k: i for i, k in enumerate(self.ds["meta"]["labels_unique"])
        }
        # fetch labels as strings
        label_map = self.ds["meta"]["label_map_df"]
        label_map = label_map[["target_encoding", "target_semantics"]].copy()
        label_map.sort_values(by="target_encoding", inplace=True)
        labels = label_map["target_semantics"].drop_duplicates().to_numpy()
        self.labels_str = [str(lab) for lab in labels]

    @staticmethod
    def _select_rows(arr, rows):
        """Rows of `arr` selected by `rows`; `rows=None` means all rows, without copying."""
        if arr is None or rows is None:
            return arr
        if isinstance(arr, ShardedArray):
            return arr.subset(rows)  # lazy view, nothing is read from disk
        return arr[rows]  # in-memory copy of the selected rows


class TimeSeriesInferenceSet:
    """Dense, order-preserving pixel stacks for a set of tiles, for patchwise
    plotting of pixelwise model predictions. Unifies TimeSeriesDatasetCreator and
    TimeSeriesDataset functionality.

    Unlike TimeSeriesDataset, this keeps every pixel (including ignore_index)
    in raster order and records per-tile (H, W) so predictions can be reshaped
    back to spatial maps.
    """

    IGNORE_INDEX = -99

    def __init__(
        self,
        meta_file,
        sel_time=None,
        sel_band=None,
        sel_task="lvl0_bic",
        seed=42,
        read_covars=False,
    ):
        self.meta_file = meta_file
        self.sel_time = sel_time
        self.sel_band = sel_band
        self.sel_task = sel_task
        self.seed = seed
        self.read_covars = read_covars

        # build a dataset without transforms
        self.ds = S2_Dataset(
            meta_file=self.meta_file,
            sel_time=self.sel_time,
            sel_band=self.sel_band,
            sel_task=self.sel_task,
            transform=None,
            use_cache=None,
            cache_dir=None,
            read_covars=self.read_covars,
        )
        n_bands, n_time = len(self.ds.i_band), len(self.ds.i_time)

        # fetch all pixel stacks and labels
        X_list, y_list, shapes, tile_ids, covar_list = [], [], [], [], []
        for i in range(len(self.ds)):
            if self.read_covars:
                x, y, c, _ = self.ds[i]
                covar_list.append(np.asarray(c).reshape(len(self.ds.COVAR_ORDER), -1).T)
            else:
                x, y, _ = self.ds[i]
            x = np.asarray(x).reshape(n_bands, n_time, -1)  # (B, T, H*W)
            y = np.asarray(y)  # (H, W)
            H, W = y.shape
            X_list.append(x.transpose(2, 0, 1))  # (H*W, B, T)
            y_list.append(y.reshape(-1))  # (H*W,)
            shapes.append((H, W))
            pid = (
                self.ds.meta_file.iloc[i]["patch_id"]
                if "patch_id" in self.ds.meta_file.columns
                else self.ds.meta_file.index[i]
            )
            tile_ids.append(pid)

        # concatenate all tiles into a single array, preserving raster order
        self.X_all = np.concatenate(X_list, 0).astype("float32")  # (P, B, T)
        self.y_all = np.concatenate(y_list, 0).astype("int16")  # (P,)
        self.covar_all = np.concatenate(covar_list, 0) if covar_list else None
        self.shapes = shapes  # per-tile (H, W)
        self.tile_ids = tile_ids

        # boundaries so a flat (P,) array can be split back into tiles
        self.tile_slices = np.concatenate(([0], np.cumsum([H * W for H, W in shapes])))

        # define a mask for valid pixels & prepare arrays for prediction
        self.valid_mask = self.y_all != self.IGNORE_INDEX
        self.X_test = self.X_all[self.valid_mask]
        self.y_test = self.y_all[self.valid_mask]
        self.covar_test = (
            None if self.covar_all is None else self.covar_all[self.valid_mask]
        )

        # create mapping -> given value: encoded value for training the classifier
        self.labels_dict = {k: i for i, k in enumerate(self.ds.labels_unique)}

        # fetch labels as strings
        label_map = self.ds.label_map_df[["target_encoding", "target_semantics"]].copy()
        label_map.sort_values(by="target_encoding", inplace=True)
        labels = label_map["target_semantics"].drop_duplicates().to_numpy()
        self.labels_str = [str(lab) for lab in labels]

    def get_covar(self, name, split="test"):
        """One covariate as a 1-D array aligned with X_test. `split` is accepted and
        ignored, so this class and TimeSeriesDataset are interchangeable to the
        evaluator."""
        if self.covar_all is None or name not in self.ds.COVAR_ORDER:
            return None
        return self.covar_test[:, self.ds.COVAR_ORDER.index(name)]

    def inputs_to_maps(self, fill=np.nan):
        """Fold the (P, B, T) inputs back into per-tile maps of shape (B*T, H, W).

        Channels are band-major / time-minor as in S2_Dataset. `fill` is unused (all
        pixels are kept) and only there for interface compatibility.

        Returns a list of (B*T, H, W) float arrays
        """
        maps = []
        for t, (H, W) in enumerate(self.shapes):
            lo, hi = self.tile_slices[t], self.tile_slices[t + 1]
            maps.append(self.X_all[lo:hi].reshape(H, W, -1).transpose(2, 0, 1))
        return maps

    def targets_to_maps(self):
        """Original label maps, ignore pixels preserved as ignore_index."""
        return [
            self.y_all[self.tile_slices[t] : self.tile_slices[t + 1]].reshape(H, W)
            for t, (H, W) in enumerate(self.shapes)
        ]

    def predictions_to_maps(self, y_pred_valid, fill=-99):
        """Fold a (n_valid,) prediction vector back into per-tile (H, W) maps.

        Returns a list of (H, W) int arrays, ignore pixels set to `fill`.
        """
        full = np.full(self.y_all.shape, fill, dtype="int64")
        full[self.valid_mask] = np.asarray(y_pred_valid)
        maps = []
        for t, (H, W) in enumerate(self.shapes):
            lo, hi = self.tile_slices[t], self.tile_slices[t + 1]
            maps.append(full[lo:hi].reshape(H, W))
        return maps

    def probs_to_maps(self, prob_valid, fill=0.0):
        """Fold a (n_valid, C) probability matrix into per-tile (C, H, W) maps."""
        C = prob_valid.shape[1]
        full = np.full((self.y_all.shape[0], C), fill, dtype="float32")
        full[self.valid_mask] = np.asarray(prob_valid)
        maps = []
        for t, (H, W) in enumerate(self.shapes):
            lo, hi = self.tile_slices[t], self.tile_slices[t + 1]
            maps.append(full[lo:hi].reshape(H, W, C).transpose(2, 0, 1))
        return maps

    def decode(self, y_encoded):
        inv = {i: k for k, i in self.labels_dict.items()}
        return np.vectorize(inv.get)(y_encoded)
