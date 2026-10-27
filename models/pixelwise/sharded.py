"""Sharded on-disk storage for large pixel-stack arrays.

Arrays are written in row-shards of fixed size as plain ``.npy`` files, so that
every shard can be memory-mapped and random rows can be read without loading
whole shards (which is what makes per-sample shuffling across the full dataset
affordable). Nothing ever holds the full array in memory, neither when writing
nor when reading.

    <root>/
      manifest.json                 -> layout, row counts, dtypes, NaN flags
      meta.pkl                      -> free-form metadata (written by the caller)
      <group>/<field>_<shard>.npy   -> e.g. train/input_00000.npy, train/label_00000.npy
"""

import json
import os

import numpy as np

MANIFEST = "manifest.json"


class ShardWriter:
    """Append rows of several aligned fields and flush them as fixed-size shards.

    Args:
        root (str): Directory the shards are written to (``<root>/<group>/``).
        group (str): Sub-directory name, e.g. the split name.
        fields (dict): ``{name: (row_shape, dtype)}``, e.g.
            ``{"input": ((n_bands, n_time), "float32"), "label": ((), "int16")}``.
        rows_per_shard (int): Number of rows per shard (the last one may be shorter).
    """

    def __init__(self, root, group, fields, rows_per_shard):
        if rows_per_shard < 1:
            raise ValueError("rows_per_shard must be >= 1")
        self.root = root
        self.group = group
        self.dir = os.path.join(root, group)
        os.makedirs(self.dir, exist_ok=True)
        self.fields = {
            k: (tuple(shape), np.dtype(dtype)) for k, (shape, dtype) in fields.items()
        }
        self.rows_per_shard = int(rows_per_shard)
        self._buf = {
            k: np.empty((self.rows_per_shard, *shape), dtype=dtype)
            for k, (shape, dtype) in self.fields.items()
        }
        self._fill = 0
        self._shards = []  # row counts of flushed shards
        self._has_nans = {k: False for k in self.fields}
        self._closed = False

    def append(self, **arrays):
        """Append aligned rows; every field must be given with the same length."""
        if self._closed:
            raise RuntimeError("ShardWriter is closed")
        if set(arrays) != set(self.fields):
            raise ValueError(
                f"expected fields {sorted(self.fields)}, got {sorted(arrays)}"
            )
        n = {len(a) for a in arrays.values()}
        if len(n) != 1:
            raise ValueError("all fields must have the same number of rows")
        n = n.pop()
        pos = 0
        while pos < n:
            take = min(n - pos, self.rows_per_shard - self._fill)
            for k, a in arrays.items():
                self._buf[k][self._fill : self._fill + take] = a[pos : pos + take]
            self._fill += take
            pos += take
            if self._fill == self.rows_per_shard:
                self._flush()

    def _flush(self):
        if self._fill == 0:
            return
        i = len(self._shards)
        for k, buf in self._buf.items():
            part = buf[: self._fill]
            if part.dtype.kind == "f" and not self._has_nans[k]:
                self._has_nans[k] = bool(np.isnan(part).any())
            np.save(os.path.join(self.dir, _shard_name(k, i)), part)
        self._shards.append(self._fill)
        self._fill = 0

    def close(self):
        """Flush the remainder and return this group's manifest entry."""
        if not self._closed:
            self._flush()
            self._closed = True
            self._buf = None
        return {
            "n_rows": int(sum(self._shards)),
            "shard_rows": [int(r) for r in self._shards],
            "fields": {
                k: {
                    "row_shape": list(shape),
                    "dtype": dtype.str,
                    "has_nans": self._has_nans[k],
                }
                for k, (shape, dtype) in self.fields.items()
            },
        }


class ShardedArray:
    """Read-only, lazily loaded concatenation of ``.npy`` shards along axis 0.

    Behaves like a numpy array for ``len``, ``shape``, ``dtype`` and row indexing
    (``int``, ``slice``, boolean mask, integer array): indexing returns an
    in-memory ``np.ndarray`` holding only the requested rows, in the requested
    order. ``np.asarray(arr)`` materialises everything (only for data that fit).

    With 3D rows ``(n, bands, times)`` it declares itself a lazy sktime panel,
    which the patched sktime deep-learning classifiers read batch-wise.

    Args:
        paths (list of str): Shard files, in row order.
        shard_rows (list of int): Number of rows of each shard.
        row_shape (tuple): Shape of a single row.
        dtype: dtype of the shards.
        index (np.ndarray, optional): Global row ids selected by this view.
        has_nans (bool, optional): Whether the data contain NaNs (from the writer).
    """

    def __init__(self, paths, shard_rows, row_shape, dtype, index=None, has_nans=None):
        if len(paths) != len(shard_rows):
            raise ValueError("paths and shard_rows must have the same length")
        self.paths = list(paths)
        self.shard_rows = np.asarray(shard_rows, dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.shard_rows)))
        self.row_shape = tuple(row_shape)
        self.dtype = np.dtype(dtype)
        self.index = None if index is None else np.asarray(index, dtype=np.int64)
        self.has_nans = has_nans
        self._mmaps = {}

    # --- construction ------------------------------------------------------- #

    @classmethod
    def from_manifest(cls, root, group, field):
        """Open ``field`` of ``group`` from a directory written by ShardWriter."""
        with open(os.path.join(root, MANIFEST)) as f:
            entry = json.load(f)["groups"][group]
        spec = entry["fields"][field]
        paths = [
            os.path.join(root, group, _shard_name(field, i))
            for i in range(len(entry["shard_rows"]))
        ]
        return cls(
            paths,
            entry["shard_rows"],
            spec["row_shape"],
            np.dtype(spec["dtype"]),
            has_nans=spec["has_nans"],
        )

    def subset(self, rows):
        """Lazy view on selected rows (int array or boolean mask), no data read."""
        rows = self._normalize_rows(rows)
        index = rows if self.index is None else self.index[rows]
        return ShardedArray(
            self.paths,
            self.shard_rows,
            self.row_shape,
            self.dtype,
            index=index,
            has_nans=self.has_nans,
        )

    # --- array protocol ------------------------------------------------------ #

    @property
    def _sktime_lazy_panel(self):
        return len(self.shape) == 3

    @property
    def shape(self):
        n = int(self.offsets[-1]) if self.index is None else len(self.index)
        return (n, *self.row_shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        rest = ()
        if isinstance(key, tuple):
            key, rest = key[0], key[1:]
        if isinstance(key, (int, np.integer)):
            out = self._read(self._normalize_rows(np.array([key])))[0]
            return out[rest] if rest else out
        out = self._read(self._normalize_rows(key))
        return out[(slice(None),) + rest] if rest else out

    def __array__(self, dtype=None, copy=None):
        out = self._read(np.arange(len(self)))
        return out if dtype is None else out.astype(dtype, copy=False)

    def __repr__(self):
        return (
            f"ShardedArray(shape={self.shape}, dtype={self.dtype}, "
            f"shards={len(self.paths)}, view={self.index is not None})"
        )

    def iter_chunks(self, chunk_rows):
        """Yield ``(start, stop, array)`` over consecutive row chunks."""
        for lo in range(0, len(self), chunk_rows):
            hi = min(lo + chunk_rows, len(self))
            yield lo, hi, self._read(np.arange(lo, hi))

    # --- internals ----------------------------------------------------------- #

    def _normalize_rows(self, key):
        n = len(self)
        if isinstance(key, slice):
            return np.arange(n)[key]
        rows = np.asarray(key)
        if rows.dtype == bool:
            if rows.shape != (n,):
                raise IndexError(f"boolean index of shape {rows.shape} for length {n}")
            return np.flatnonzero(rows)
        if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu"):
            raise IndexError("rows must be an int, slice, 1D int array or bool mask")
        rows = rows.astype(np.int64, copy=False)
        if rows.size and (rows.min() < -n or rows.max() >= n):
            raise IndexError(f"row index out of range for length {n}")
        return np.where(rows < 0, rows + n, rows)

    def _mmap(self, s):
        mm = self._mmaps.get(s)
        if mm is None:
            mm = np.load(self.paths[s], mmap_mode="r")
            self._mmaps[s] = mm
        return mm

    def _read(self, rows):
        """Read the given (already normalised) rows of this view, in order."""
        out = np.empty((len(rows), *self.row_shape), dtype=self.dtype)
        if len(rows) == 0:
            return out
        glob = rows if self.index is None else self.index[rows]
        order = np.argsort(glob, kind="stable")
        glob_sorted = glob[order]
        shard = np.searchsorted(self.offsets, glob_sorted, side="right") - 1
        bounds = np.flatnonzero(np.diff(shard)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(shard)]):
            s = int(shard[lo])
            local = glob_sorted[lo:hi] - self.offsets[s]
            mm = self._mmap(s)
            if np.all(np.diff(local) == 1):  # contiguous & unique -> cheap slice
                block = mm[local[0] : local[-1] + 1]
            else:
                block = mm[local]
            out[order[lo:hi]] = block
        return out

    def __getstate__(self):
        # memory maps are not shipped to other processes, they are reopened
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        return state


def load_field(root, group, field):
    """Load a (small) field completely into memory, e.g. labels or covariates."""
    return np.asarray(ShardedArray.from_manifest(root, group, field))


def write_manifest(root, groups, extra=None):
    """Write the manifest after all shards are flushed."""
    manifest = {"format": "npy-shards-v1", "groups": groups}
    if extra:
        manifest.update(extra)
    with open(os.path.join(root, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def dir_size(root):
    return sum(
        os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(root) for f in fs
    )


def _shard_name(field, i):
    return f"{field}_{i:05d}.npy"
