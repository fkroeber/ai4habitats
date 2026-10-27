import numpy as np
import pandas as pd
from pathlib import Path

from models.paths import data_dir_of, resolve_paths


class SplitGenerator:
    """
    Generate train/validation/test assignments for habitat patches.

    Parameters
    ----------
    meta_file : str, Path, or pandas.DataFrame
        Metadata table or path to a CSV/Parquet/Pickle metadata file. Relative
        'path_input'/'path_label' entries of a file are resolved against the
        file's folder; a DataFrame is used as is.
    strategy : str
        - "random"         : row-level random split.
        - "spatial-block"  : complete grid cells are assigned to one split.
          The grid is generated on the fly from the patch coordinates
          (see `grid_distance`, `grid_crs`, `grid_origin`).
    trainsize : float
        Target share assigned to training. The remaining share is divided
        equally between validation and test.
    fold : int
        Ignored. Kept for backward compatibility of the Trainer interface; use
        different seeds to obtain different splits.
    subset : float
        Fraction of metadata rows retained before splitting, in (0, 1].
    seed : int
        Random seed. Use different seeds to obtain different splits.
    grid_col : str or None
        Name of the column the generated grid index is written to
        ("spatial-block" only). Defaults to "grid_idx". An existing column
        of the same name is overwritten.
    split_col : str
        Name of the generated split column.
    grid_distance : float
        Edge length of the square grid cells in units of `grid_crs`
        (metres for projected CRSs). Default 25600, as in `gen_grid`.
    grid_crs : str
        Projected CRS in which the fishnet grid is laid out. Patch centres
        (`lat_col`/`lon_col`, WGS84) are projected into it before binning.
        Should match the CRS of the AoI used by `gen_grid` if the original
        cell layout is to be reproduced.
    grid_origin : tuple of float or None
        Lower-left corner (x, y) of the grid in `grid_crs`. If None, the
        minimum projected patch coordinates of the full (non-subsetted)
        metadata, floored to 100 m, are used - mirroring the AoI-based
        alignment in `gen_grid`.
    lat_col, lon_col : str
        Columns holding the WGS84 patch-centre coordinates.
    """

    VALID_STRATEGIES = {
        "random",
        "spatial-block",
    }
    SPLIT_NAMES = np.array(
        ["train", "val", "test"],
        dtype=object,
    )
    DEFAULT_GRID_COL = "grid_idx"
    GRID_ALIGN = 100  # round number (in CRS units) the default origin snaps to

    def __init__(
        self,
        meta_file,
        strategy="random",
        trainsize=0.8,
        fold=0,
        subset=1.0,
        seed=42,
        grid_col=None,
        split_col="split",
        grid_distance=25600,
        grid_crs="EPSG:25832",
        grid_origin=None,
        lat_col="lat",
        lon_col="lon",
    ):
        self.meta_file = meta_file
        self.strategy = strategy
        self.trainsize = float(trainsize)
        self.subset = float(subset)
        self.seed = int(seed)
        self.grid_col = grid_col
        self.split_col = split_col
        self.grid_distance = float(grid_distance)
        self.grid_crs = grid_crs
        self.grid_origin = grid_origin
        self.lat_col = lat_col
        self.lon_col = lon_col

        self.meta_df_ = None
        self.summary_ = None
        self.group_assignment_ = None

        self._validate_arguments()

        # note: fold is not used anymore
        # randomness guided by the `seed` parameter
        # kept for backward compatibility

    @property
    def proportions(self):
        """Return target train/validation/test proportions."""
        holdout = (1.0 - self.trainsize) / 2.0
        return np.array(
            [self.trainsize, holdout, holdout],
            dtype=float,
        )

    @property
    def grid_col_name(self):
        """Column the generated grid index is written to."""
        return self.grid_col if self.grid_col is not None else self.DEFAULT_GRID_COL

    # ------------------------------------------------------------------
    # Validation & loading
    # ------------------------------------------------------------------
    def _validate_arguments(self):
        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                "strategy must be one of "
                f"{sorted(self.VALID_STRATEGIES)}, "
                f"but received {self.strategy!r}."
            )

        if not 0 < self.trainsize < 1:
            raise ValueError("trainsize must be strictly between 0 and 1.")

        if not 0 < self.subset <= 1:
            raise ValueError("subset must be in the interval (0, 1].")

        if not np.isfinite(self.grid_distance) or self.grid_distance <= 0:
            raise ValueError("grid_distance must be a positive number.")

        if self.grid_origin is not None and len(self.grid_origin) != 2:
            raise ValueError("grid_origin must be None or an (x, y) pair.")

    def _load_metadata(self):
        if isinstance(self.meta_file, pd.DataFrame):
            return self.meta_file.copy()

        path = Path(self.meta_file)

        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()

        if suffix == ".csv":
            meta_df = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            meta_df = pd.read_parquet(path)
        elif suffix in {".pkl", ".pickle"}:
            meta_df = pd.read_pickle(path)
        else:
            raise ValueError(
                f"Unsupported metadata file format: {suffix!r}. "
                "Use CSV, Parquet, Pickle, or pass a DataFrame."
            )

        # paths in the file are relative to the file's folder (same rule as S2_Dataset)
        return resolve_paths(meta_df, data_dir_of(path))

    # ------------------------------------------------------------------
    # Dynamic grid generation
    # ------------------------------------------------------------------
    def _compute_grid_index(self, meta_df):
        """
        Bin patch centres into a square fishnet grid of size `grid_distance`.

        Cells are numbered row-major starting at the lower-left corner
        (x varies fastest), as in `gen_grid`:
            grid_idx = row * n_cols + col
        """
        for col in (self.lat_col, self.lon_col):
            if col not in meta_df.columns:
                raise KeyError(f"Coordinate column {col!r} is missing.")

        lat = pd.to_numeric(meta_df[self.lat_col], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(meta_df[self.lon_col], errors="coerce").to_numpy(float)

        if np.isnan(lat).any() or np.isnan(lon).any():
            raise ValueError(
                f"{self.lat_col!r}/{self.lon_col!r} contain missing "
                "or non-numeric values."
            )

        # imported lazily so that the random strategy has no pyproj dependency
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            self.grid_crs,
            always_xy=True,
        )
        x, y = transformer.transform(lon, lat)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(
                f"Some coordinates could not be projected to {self.grid_crs!r}."
            )

        if self.grid_origin is None:
            x0 = np.floor(x.min() / self.GRID_ALIGN) * self.GRID_ALIGN
            y0 = np.floor(y.min() / self.GRID_ALIGN) * self.GRID_ALIGN
        else:
            x0, y0 = (float(v) for v in self.grid_origin)

        col_idx = np.floor((x - x0) / self.grid_distance).astype(np.int64)
        row_idx = np.floor((y - y0) / self.grid_distance).astype(np.int64)

        if (col_idx < 0).any() or (row_idx < 0).any():
            raise ValueError(
                "grid_origin must lie south-west of all patch centres "
                f"(in {self.grid_crs!r})."
            )

        n_cols = int(col_idx.max()) + 1
        n_rows = int(row_idx.max()) + 1

        self.grid_origin_ = (float(x0), float(y0))
        self.grid_shape_ = (n_rows, n_cols)

        return pd.Series(
            row_idx * n_cols + col_idx,
            index=meta_df.index,
            name=self.grid_col_name,
        )

    # ------------------------------------------------------------------
    # Subsetting & random split (unchanged)
    # ------------------------------------------------------------------
    def _subset_metadata(self, meta_df, rng):
        if self.subset == 1:
            return meta_df.copy()
        else:
            n_selected = max(
                1,
                int(round(len(meta_df) * self.subset)),
            )
            positions = rng.choice(
                len(meta_df),
                size=n_selected,
                replace=False,
            )
            positions = np.sort(positions)
            return meta_df.iloc[positions].copy()

    def _integer_target_counts(self, n_samples):
        """
        Convert fractional target shares into integer counts while ensuring
        that their sum equals n_samples.
        """
        raw_counts = self.proportions * n_samples
        counts = np.floor(raw_counts).astype(int)

        remainder = n_samples - counts.sum()

        fractional_parts = raw_counts - counts
        order = np.argsort(
            -fractional_parts,
            kind="stable",
        )

        counts[order[:remainder]] += 1

        # Where possible, ensure all three splits receive at least one row.
        if n_samples >= 3:
            for split_index in range(3):
                if counts[split_index] == 0:
                    donors = np.flatnonzero(counts > 1)

                    if len(donors):
                        donor = donors[np.argmax(counts[donors])]

                        counts[donor] -= 1
                        counts[split_index] += 1

        return counts

    def _assign_random(self, meta_df, rng):
        counts = self._integer_target_counts(len(meta_df))

        permutation = rng.permutation(len(meta_df))

        assignments = np.empty(
            len(meta_df),
            dtype=object,
        )

        start = 0

        for split_name, count in zip(
            self.SPLIT_NAMES,
            counts,
        ):
            positions = permutation[start : start + count]

            assignments[positions] = split_name
            start += count

        return pd.Series(
            assignments,
            index=meta_df.index,
            name=self.split_col,
        )

    # ------------------------------------------------------------------
    # Spatial-block split
    # ------------------------------------------------------------------
    def _assign_groups(self, group_ids, rng):
        """
        Assign complete groups (grid cells) to train, val, and test.

        Cells are shuffled (seed-dependent) and cut where the cumulative
        patch count comes closest to the train and train+val targets.
        Each cut deviates from its target by at most half a cell.
        """
        if group_ids.isna().any():
            raise ValueError("group_ids contains missing values.")

        # sorted so the result does not depend on the row order of the metadata
        group_sizes = group_ids.value_counts(sort=False).sort_index()

        n_splits = len(self.SPLIT_NAMES)

        if len(group_sizes) < n_splits:
            raise ValueError(
                f"At least {n_splits} occupied grid cells are required, "
                f"but only {len(group_sizes)} were found. "
                "Reduce grid_distance."
            )

        group_names = group_sizes.index.to_numpy()
        sizes = group_sizes.to_numpy()

        n_groups = len(sizes)

        # random cell order creates variation between seeds
        order = rng.permutation(n_groups)

        # cum_counts[i] = number of patches in the first i shuffled cells
        cum_counts = np.concatenate([[0], np.cumsum(sizes[order])])
        bounds = np.cumsum(self.proportions)[:-1] * sizes.sum()
        cut_train, cut_val = (
            int(np.argmin(np.abs(cum_counts - bound))) for bound in bounds
        )

        # every split receives at least one cell
        cut_train = min(max(cut_train, 1), n_groups - 2)
        cut_val = min(max(cut_val, cut_train + 1), n_groups - 1)

        split_idx = np.searchsorted(
            [cut_train, cut_val],
            np.arange(n_groups),
            side="right",
        )

        mapping = dict(zip(group_names[order], self.SPLIT_NAMES[split_idx]))
        counts = np.bincount(split_idx, weights=sizes[order], minlength=n_splits)

        self.group_assignment_ = mapping
        self.group_counts_ = dict(zip(self.SPLIT_NAMES, counts.astype(int)))

        return group_ids.map(mapping).rename(self.split_col)

    def _validate_block_integrity(self, meta_df):
        if self.strategy == "spatial-block":
            n_splits_per_grid = meta_df.groupby(self.grid_col_name)[
                self.split_col
            ].nunique()
            if n_splits_per_grid.max() > 1:
                raise RuntimeError(
                    "Geographic leakage detected: at least one "
                    "grid cell occurs in multiple splits."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self):
        """Generate and return the subsetted metadata with split assignments."""
        meta_df = self._load_metadata()

        if meta_df.empty:
            raise ValueError("The metadata table is empty.")

        # grid is built on the full table, so cell layout is subset-independent
        if self.strategy == "spatial-block":
            meta_df[self.grid_col_name] = self._compute_grid_index(meta_df)

        # setup independent reproducible streams for subsetting and splitting
        seed_sequence = np.random.SeedSequence(self.seed)
        subset_seed, split_seed = seed_sequence.spawn(2)
        subset_rng = np.random.default_rng(subset_seed)
        split_rng = np.random.default_rng(split_seed)

        # subset meta_df
        meta_df = self._subset_metadata(
            meta_df,
            subset_rng,
        )

        # assign splits
        if self.strategy == "random":
            assignments = self._assign_random(
                meta_df,
                split_rng,
            )
        else:
            assignments = self._assign_groups(
                meta_df[self.grid_col_name],
                split_rng,
            )
        meta_df[self.split_col] = assignments

        # validation & summary
        self._validate_block_integrity(meta_df)
        counts = (
            meta_df[self.split_col]
            .value_counts()
            .reindex(
                self.SPLIT_NAMES,
                fill_value=0,
            )
        )
        self.summary_ = pd.DataFrame(
            {
                "n_patches": counts,
                "share": counts / len(meta_df),
                "target_share": self.proportions,
            }
        )
        self.summary_.index.name = self.split_col
        self.meta_df_ = meta_df
        return meta_df

    def save(self, output_file, index=False):
        """Save the generated metadata table."""
        if self.meta_df_ is None:
            raise RuntimeError("Call generate() before save().")

        output_file = Path(output_file)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_file.suffix.lower() == ".csv":
            self.meta_df_.to_csv(
                output_file,
                index=index,
            )
        elif output_file.suffix.lower() in {
            ".parquet",
            ".pq",
        }:
            self.meta_df_.to_parquet(
                output_file,
                index=index,
            )
        else:
            raise ValueError("Output must be CSV or Parquet.")

        return output_file
