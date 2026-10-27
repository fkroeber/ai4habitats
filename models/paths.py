"""Path handling for the patch metadata (patches_meta.csv).

The data set ships patches_meta.csv with paths relative to the CSV's folder::

    <data_dir>/patches_meta.csv
    <data_dir>/s2_normalization_stats.json
    <data_dir>/sentinel/patch_<id>.nc        <- path_input
    <data_dir>/labels/patch_<id>.nc          <- path_label

All readers use the same rule: relative paths are joined to the folder of the
metadata file, absolute paths are kept unchanged.
"""

import os

PATH_COLS = ("path_input", "path_label")


def resolve_paths(meta_df, data_dir):
    """Return a copy of `meta_df` with relative paths joined to `data_dir`.

    Absolute paths are left untouched, so applying the function to an already
    resolved table is a no-op.
    """
    meta_df = meta_df.copy()
    for col in PATH_COLS:
        if col in meta_df.columns:
            meta_df[col] = [os.path.join(data_dir, p) for p in meta_df[col]]
    return meta_df


def data_dir_of(meta_file):
    """Absolute folder of a metadata file path."""
    return os.path.dirname(os.path.abspath(meta_file))
