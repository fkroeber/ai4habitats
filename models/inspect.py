"""Parse, aggregate & compare training runs of the patchwise / pixelwise Trainers.

Both Trainers write the same core artefacts into their ``save_dir``::

    <save_dir>/                     # basename == wandb_runname == run_id
    |-- config.json                 # constructor args      (Trainer.save_config)
    |-- model_loss.csv              # one row per epoch, columns train/val[/test]
    |-- model_metrics.csv           # one row per epoch, dict-valued cells
    |-- training_completed.txt      # written last -> marks a finished run
    |-- train/ val/ test/           # confusion matrices & classification reports
    `-- model_params.pt | checkpoints/

Division of labour between the two sources:

* **local files** own the *configuration* and the *per-epoch curves*
  (train/val[/test] loss plus the metrics tracked while training).
* **W&B** owns the *final results*. The ``best_*`` summary keys are computed
  after training with ``metric_collection.set_stage("eval")``, which yields a
  different (larger) metric set than the one written to ``model_metrics.csv``.
  Those numbers exist nowhere on disk, so W&B access is **required**, not
  optional.

Only runs present in *both* sources are analysed. Runs found on one side only are
reported in :attr:`ResultsParser.local_only` / :attr:`ResultsParser.wandb_only`
so they can be cleaned up, and are excluded from every aggregate.

Typical use::
    res = ResultsParser(
        "/lustre/.../results/test",
        project="playground",
        tags="test_metrics"
    )
    res.describe()                 # which hyper-parameters actually vary
    res.summary                    # config + final (eval-stage) metrics per run
    res.curves                     # run_name | epoch | split | loss | <tracked metrics>
    res.best_long                  # run_name | split | <eval-stage metrics>
    res.check_consistency()        # local vs. W&B, train/val loss only

    res.save("results.pkl")                 # freeze everything
    res = ResultsParser.load("results.pkl") # reload without disk or W&B access
"""

import ast
import json
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

CONFIG_FILE = "config.json"
LOSS_FILE = "model_loss.csv"
METRICS_FILE = "model_metrics.csv"
DONE_FILE = "training_completed.txt"
SPLITS = ("train", "val", "test")

#: config keys unique to each Trainer, used to tell the two families apart
_PIXELWISE_KEYS = {"n_samples", "pred_batch_size"}
_PATCHWISE_KEYS = {"encoder_name", "train_mode", "train_augmentations"}

#: splits for which per-epoch curves exist locally. The pixelwise Trainer logs
#: only train/val while fitting (keras), the patchwise one loops over all three.
_CURVE_SPLITS = {"patchwise": SPLITS, "pixelwise": ("train", "val")}

#: W&B summary keys holding a final-epoch loss, mapped to a split. Patchwise logs
#: `loss_<split>` explicitly; the keras callback of the pixelwise Trainer logs
#: the usual `loss` / `val_loss` pair.
_LOSS_ALIASES = {
    "loss_train": "train",
    "loss_val": "val",
    "loss_test": "test",
    "loss": "train",
    "val_loss": "val",
    "test_loss": "test",
}

#: config entries that describe the run rather than its configuration; excluded
#: from `describe()` so that only genuine hyper-parameters are compared
_NON_CONFIG_KEYS = [
    "save_dir",
    "best_epoch",
    "n_epochs",
    "wandb_project",
    "wandb_runname",
    "wandb_notes",
    "meta_file",
    "total_params",
    "trainable_params",
    "size_input",
    "size_params",
    "size_forward_backward_pass",
    "size_total",
    "size_train_set",
    "size_val_set",
    "size_test_set",
    "training_epochs",
    "training_time_total",
    "training_time_epoch",
    "run_dir",
    "completed",
]


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #


def infer_family(config):
    """Return 'patchwise', 'pixelwise' or 'unknown' for a Trainer config."""
    keys = set(config)
    if keys & _PIXELWISE_KEYS:
        return "pixelwise"
    if keys & _PATCHWISE_KEYS:
        return "patchwise"
    return "unknown"


def _col(split, metric):
    """('val', 'macro_f1') -> 'val_macro_f1'."""
    return f"{split}_{metric}"


def _parse_col(col):
    """'val_macro_f1' -> ('val', 'macro_f1'); (None, col) if no split prefix."""
    m = re.match(rf"^({'|'.join(SPLITS)})_(.+)$", col)
    return m.groups() if m else (None, col)


def _parse_wandb_key(key):
    """'best_macro_f1_val' -> ('val', 'macro_f1'); (None, ...) if no split suffix."""
    stem = key[len("best_") :] if key.startswith("best_") else key
    m = re.match(rf"^(.+?)_({'|'.join(SPLITS)})$", stem)
    if not m:
        return None, stem
    metric, split = m.groups()
    return split, metric


def _is_number(v):
    """True for real numeric scalars. NaN is rejected: missing summary keys reach
    this as float('nan') once the rows are assembled into a DataFrame."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return False
    return bool(pd.notna(v))


def _literal_eval_safe(x):
    """Parse a stringified dict as written by ``DataFrame.to_csv``.

    ``metric_history`` holds dicts whose values may be numpy scalars, so the csv
    contains reprs like ``{'macro_f1': np.float64(0.31), 'cm': array([...])}``
    which ``ast.literal_eval`` rejects. Those wrappers are stripped first.
    Returns {} for anything unparseable, so one bad row cannot kill a run.
    """
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    s = str(x)
    s = re.sub(r"\bnan\b", "None", s)
    s = re.sub(r"\binf\b", "float('inf')", s)
    s = re.sub(r"np\.\w+\(\s*([-+0-9.eE]+)\s*\)", r"\1", s)
    s = re.sub(r"array\([^()]*\[[^\]]*\][^)]*\)", "None", s, flags=re.DOTALL)
    s = re.sub(r"tensor\([^()]*\)", "None", s, flags=re.DOTALL)
    try:
        out = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}
    return out if isinstance(out, dict) else {}


def _hashable(v):
    """Make list/dict config values usable in `nunique` / `drop_duplicates`."""
    return str(v) if isinstance(v, (list, dict, set, tuple)) else v


# ---------------------------------------------------------------------- #
# a single run (local artefacts only)
# ---------------------------------------------------------------------- #


class RunResults:
    """Local artefacts of one training run, read lazily from its ``save_dir``."""

    def __init__(self, run_dir):
        self.run_dir = os.path.abspath(run_dir)
        with open(os.path.join(self.run_dir, CONFIG_FILE)) as f:
            self.config = json.load(f)
        self.run_name = self.config.get("wandb_runname") or os.path.basename(
            self.run_dir
        )
        self.family = infer_family(self.config)
        self._curves = None

    def __repr__(self):
        state = "completed" if self.completed else "incomplete"
        return f"<RunResults {self.run_name} ({self.family}, {state})>"

    @property
    def completed(self):
        return os.path.exists(os.path.join(self.run_dir, DONE_FILE))

    @property
    def tags(self):
        return self.config.get("wandb_tags") or []

    # -- per-epoch curves ------------------------------------------------ #

    @property
    def curves(self):
        """Long format: epoch | split | loss | <metrics tracked while training>."""
        if self._curves is None:
            loss, metrics = self._read_loss(), self._read_metrics()
            if loss.empty or metrics.empty:
                self._curves = loss if not loss.empty else metrics
            else:
                self._curves = pd.merge(
                    loss, metrics, on=["epoch", "split"], how="outer"
                )
        return self._curves

    def _read_loss(self):
        path = os.path.join(self.run_dir, LOSS_FILE)
        if not os.path.isfile(path):
            return pd.DataFrame(columns=["epoch", "split", "loss"])
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c in SPLITS]
        if not cols:
            warnings.warn(f"⚠️ No train/val/test columns in {path}")
            return pd.DataFrame(columns=["epoch", "split", "loss"])
        df = df[cols].reset_index(names="epoch")
        return df.melt(id_vars="epoch", var_name="split", value_name="loss")

    def _read_metrics(self):
        path = os.path.join(self.run_dir, METRICS_FILE)
        if os.path.isdir(path):
            # pixelwise/torch logs via CSVLogger into a directory instead
            warnings.warn(
                f"⚠️ {path} is a directory (torch CSVLogger); per-epoch metrics "
                "are not parsed for this run."
            )
            return pd.DataFrame(columns=["epoch", "split"])
        if not os.path.isfile(path):
            return pd.DataFrame(columns=["epoch", "split"])
        raw = pd.read_csv(path)
        cols = [c for c in raw.columns if c in SPLITS]
        if not cols:
            warnings.warn(f"⚠️ Unexpected metric layout in {path}; skipped.")
            return pd.DataFrame(columns=["epoch", "split"])
        frames = []
        for c in cols:
            exploded = raw[c].apply(_literal_eval_safe).apply(pd.Series)
            exploded.insert(0, "split", c)
            exploded.insert(0, "epoch", np.arange(len(exploded)))
            frames.append(exploded)
        out = pd.concat(frames, ignore_index=True)
        # drop non-scalar leftovers (confusion matrices etc.)
        keep = ["epoch", "split"] + [
            c
            for c in out.columns
            if c not in ("epoch", "split")
            and pd.api.types.is_numeric_dtype(pd.to_numeric(out[c], errors="coerce"))
            and not out[c].isna().all()
        ]
        return out[keep]

    # -- derived scalars ------------------------------------------------- #

    @property
    def n_epochs(self):
        c = self.curves
        return int(c["epoch"].max()) + 1 if len(c) else 0

    @property
    def best_epoch(self):
        """Epoch of minimum validation loss, derived from the local loss curve.

        Derived locally on purpose: the patchwise Trainer does not log
        `stop_epoch` to W&B, so this is the only source covering both families.
        """
        c = self.curves
        if "loss" not in c.columns:
            return None
        val = c[c["split"] == "val"].dropna(subset=["loss"])
        return int(val.loc[val["loss"].idxmin(), "epoch"]) if len(val) else None

    def loss_at(self, epoch):
        """{split: loss} at a given epoch, from the local curves."""
        c = self.curves
        if "loss" not in c.columns or epoch is None:
            return {}
        rows = c[c["epoch"] == epoch]
        return {r.split: r.loss for r in rows.itertuples() if pd.notna(r.loss)}

    @property
    def final_loss(self):
        """{split: loss} at the last logged epoch."""
        return self.loss_at(self.n_epochs - 1 if self.n_epochs else None)

    def config_row(self):
        """Flat config dict plus derived bookkeeping fields."""
        row = {k: _hashable(v) for k, v in self.config.items()}
        row.update(
            run_name=self.run_name,
            run_dir=self.run_dir,
            family=self.family,
            completed=self.completed,
            n_epochs=self.n_epochs,
            best_epoch=self.best_epoch,
        )
        return row


# ---------------------------------------------------------------------- #
# W&B access
# ---------------------------------------------------------------------- #


class WandbFetcher:
    """Thin wrapper around the W&B API, indexed by run name."""

    def __init__(self, project, entity="fkroeber"):
        import wandb

        self.project = project
        self.entity = entity
        self.api = wandb.Api()

    def summary_frame(self, tags=None, verbose=True):
        """One row per W&B run.

        Columns:
            ``<split>_<metric>``   eval-stage results from the ``best_*`` keys
            ``final_<split>_loss`` last logged loss, for cross-checking
            ``stop_epoch``         where the Trainer logged it (pixelwise only)
            ``wandb_id``/``wandb_url``/``remote_tags``  for locating & deleting runs
        """
        rows = []
        runs = self.api.runs(f"{self.entity}/{self.project}")
        for run in tqdm(runs, desc="Fetching W&B runs", disable=not verbose):
            if tags and not set(run.tags) & set(tags):
                continue
            summary = dict(run.summary._json_dict)
            row = {
                "run_name": run.name,
                "wandb_id": run.id,
                "wandb_url": getattr(run, "url", None),
                "remote_tags": str(list(run.tags)),
            }
            if _is_number(summary.get("stop_epoch")):
                row["stop_epoch"] = summary["stop_epoch"]
            for k, v in summary.items():
                if not _is_number(v):
                    continue
                if k.startswith("best_"):
                    split, metric = _parse_wandb_key(k)
                    if split:
                        row[_col(split, metric)] = v
                elif k in _LOSS_ALIASES:
                    row[f"final_{_LOSS_ALIASES[k]}_loss"] = v
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # a re-used run name would silently duplicate rows on join
        dupes = df["run_name"].duplicated(keep=False)
        if dupes.any():
            warnings.warn(
                f"⚠️ Duplicate W&B run names, keeping the most recent: "
                f"{sorted(df.loc[dupes, 'run_name'].unique())}"
            )
        return df[~df["run_name"].duplicated(keep="last")].set_index("run_name")


# ---------------------------------------------------------------------- #
# a collection of runs
# ---------------------------------------------------------------------- #


class ResultsParser:
    """Discover local runs, intersect them with W&B, and aggregate the results."""

    def __init__(
        self,
        results_dir,
        project,
        entity="fkroeber",
        tags=None,
        family=None,
        only_completed=True,
        max_depth=2,
        verbose=True,
    ):
        """
        Args:
            results_dir (str): Root directory holding one sub-directory per run.
            project (str): W&B project holding the `best_*` summary metrics.
                Required: those metrics are not written to disk by either Trainer.
            entity (str): W&B entity. Default "fkroeber".
            tags (str | list, optional): Keep only runs carrying at least one of
                these tags. Applied to both sources. None keeps everything.
            family (str, optional): "patchwise" or "pixelwise" to restrict to one
                model family. None keeps both.
            only_completed (bool): Drop local runs without `training_completed.txt`.
            max_depth (int): How deep below `results_dir` to search for configs.
                Runs sitting directly in `results_dir` are at depth 1, so the
                default of 2 also picks up runs grouped in sweep sub-directories.
            verbose (bool): Print a discovery / intersection summary.
        """
        self.results_dir = os.path.abspath(results_dir)
        self.project = project
        self.entity = entity
        self.tags = [tags] if isinstance(tags, str) else tags
        self.family = family
        self.only_completed = only_completed
        self.max_depth = max_depth
        self.verbose = verbose
        self.from_cache = False
        self.unreadable = {}
        self._cache = {}
        self._discover()
        self._fetch_wandb()
        self._intersect()

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, i):
        return self.runs[i]

    def __repr__(self):
        src = "cache" if self.from_cache else f"{self.results_dir} + W&B/{self.project}"
        return (
            f"<ResultsParser {len(self.runs)} run(s) from {src}; "
            f"{len(self.local_only)} local-only, {len(self.wandb_only)} W&B-only>"
        )

    # -- construction ---------------------------------------------------- #

    def _discover(self):
        found, skipped = [], {"incomplete": 0, "tags": 0, "family": 0}
        for cur, subdirs, files in os.walk(self.results_dir):
            if cur[len(self.results_dir) :].count(os.sep) > self.max_depth:
                subdirs[:] = []
                continue
            if CONFIG_FILE not in files:
                continue
            try:
                run = RunResults(cur)
            except Exception as e:  # unreadable / truncated config
                self.unreadable[cur] = f"{type(e).__name__}: {e}"
                continue
            if self.only_completed and not run.completed:
                skipped["incomplete"] += 1
            elif self.tags and not set(run.tags) & set(self.tags):
                skipped["tags"] += 1
            elif self.family and run.family != self.family:
                skipped["family"] += 1
            else:
                found.append(run)
                continue
        found.sort(key=lambda r: r.run_name)
        self._local = found
        if self.verbose:
            print(f"Local: {len(found)} run(s) under {self.results_dir}")
            for k, v in skipped.items():
                if v:
                    print(f"  skipped {v} ({k})")
            for d, err in self.unreadable.items():
                print(f"  ⚠️ unreadable config in {d}: {err}")

    def _fetch_wandb(self):
        fetcher = WandbFetcher(self.project, entity=self.entity)
        self._wandb = fetcher.summary_frame(tags=self.tags, verbose=self.verbose)
        if self.verbose:
            print(f"W&B:   {len(self._wandb)} run(s) in {self.entity}/{self.project}")

    def _intersect(self):
        """Keep only runs present in both sources; record the orphans."""
        local_names = {r.run_name for r in self._local}
        remote_names = set(self._wandb.index) if len(self._wandb) else set()
        shared = local_names & remote_names

        self.runs = [r for r in self._local if r.run_name in shared]
        self.local_only = pd.DataFrame(
            [
                {
                    "run_name": r.run_name,
                    "run_dir": r.run_dir,
                    "family": r.family,
                    "completed": r.completed,
                }
                for r in self._local
                if r.run_name not in shared
            ]
        )
        cols = ["wandb_id", "wandb_url", "remote_tags"]
        self.wandb_only = (
            self._wandb.loc[
                sorted(remote_names - shared),
                [c for c in cols if c in self._wandb.columns],
            ].reset_index()
            if remote_names - shared
            else pd.DataFrame(columns=["run_name", *cols])
        )
        self._wandb = self._wandb.loc[sorted(shared)] if shared else pd.DataFrame()
        if self.verbose:
            print(f"Shared: {len(self.runs)} run(s) retained for analysis")
            if len(self.local_only):
                print(f"  {len(self.local_only)} local-only -> .local_only")
            if len(self.wandb_only):
                print(f"  {len(self.wandb_only)} W&B-only   -> .wandb_only")

    # -- aggregated frames ----------------------------------------------- #

    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def configs(self):
        """One row per run: local config + bookkeeping, indexed by run_name."""
        return self._cached(
            "configs",
            lambda: (
                pd.DataFrame([r.config_row() for r in self.runs]).set_index("run_name")
                if self.runs
                else pd.DataFrame()
            ),
        )

    @property
    def curves(self):
        """Per-epoch curves from disk: run_name | epoch | split | loss | <metrics>.

        These are the metrics tracked *while* training. They are a smaller set
        than `best_wide` and are not directly comparable to it.
        """

        def build():
            frames = []
            for r in tqdm(self.runs, desc="Reading curves", disable=not self.verbose):
                if not r.curves.empty:
                    frames.append(r.curves.assign(run_name=r.run_name))
            if not frames:
                return pd.DataFrame()
            out = pd.concat(frames, ignore_index=True)
            lead = ["run_name", "epoch", "split"]
            return out[lead + [c for c in out.columns if c not in lead]]

        return self._cached("curves", build)

    @property
    def best_wide(self):
        """Final (eval-stage) metrics from W&B: one row per run, `<split>_<metric>`.

        This is the authoritative results table. The numbers come from the
        post-training evaluation pass and do not exist in the local csv files.
        """

        def build():
            if self._wandb.empty:
                return pd.DataFrame()
            metric_cols = [
                c for c in self._wandb.columns if _parse_col(c)[0] is not None
            ]
            keep = [c for c in ("stop_epoch",) if c in self._wandb.columns]
            return self._wandb[keep + sorted(metric_cols)]

        return self._cached("best_wide", build)

    @property
    def best_long(self):
        """Final metrics in long format: run_name | split | <metrics>."""

        def build():
            wide = self.best_wide
            if wide.empty:
                return pd.DataFrame()
            recs = {}
            for run, row in wide.iterrows():
                for col, val in row.items():
                    split, metric = _parse_col(col)
                    if split:
                        recs.setdefault((run, split), {})[metric] = val
            out = pd.DataFrame(
                [{"run_name": r, "split": s, **d} for (r, s), d in recs.items()]
            )
            return out.sort_values(["run_name", "split"]).reset_index(drop=True)

        return self._cached("best_long", build)

    @property
    def summary(self):
        """Config joined with the final W&B metrics, one row per run."""

        def build():
            if self.configs.empty:
                return self.configs
            if self.best_wide.empty:
                return self.configs
            overlap = self.configs.columns.intersection(self.best_wide.columns)
            return self.configs.join(self.best_wide.drop(columns=overlap), how="left")

        return self._cached("summary", build)

    # -- inspection ------------------------------------------------------ #

    def describe(self, max_values=10):
        """Print which hyper-parameters actually vary across the runs."""
        df = self.configs.drop(
            columns=[c for c in _NON_CONFIG_KEYS if c in self.configs.columns],
            errors="ignore",
        )
        if df.empty:
            print("No runs to describe.")
            return df
        order = df.nunique().sort_values(ascending=False).index
        print(f"\n--- {len(df)} run(s), unique configuration values ---")
        for col in order:
            n = df[col].nunique(dropna=False)
            marker = "*" if n > 1 else " "
            vals = pd.unique(df[col].astype(object))[:max_values].tolist()
            print(f"{marker} {col}: ({n} value(s)) {vals}")
        print("\n(* = varies across runs)")
        return df

    def check_consistency(self, rtol=1e-4, atol=1e-8):
        """Sanity-check the parsed results; returns a DataFrame of problems.

        Only **train/val loss** is compared between local files and W&B. The
        `best_*` metrics deliberately are not: they are computed after training
        with `set_stage("eval")` and cover a different metric set than the one
        tracked per epoch, so any comparison would be apples-to-oranges. Test
        loss is skipped too, since the pixelwise Trainer does not track a test
        split while fitting.
        """
        issues = []

        def add(run, msg):
            issues.append({"run_name": run, "issue": msg})

        for r in self.runs:
            expected = _CURVE_SPLITS.get(r.family, SPLITS)
            # structural checks on the local curves
            if r.best_epoch is None:
                add(r.run_name, "no validation loss found locally")
            elif r.best_epoch >= r.n_epochs:
                add(r.run_name, f"best_epoch {r.best_epoch} >= n_epochs {r.n_epochs}")
            logged = r.config.get("training_epochs")
            if logged is not None and int(logged) != r.n_epochs:
                add(
                    r.run_name,
                    f"training_epochs={logged} but {r.n_epochs} epochs in {LOSS_FILE}",
                )
            for split in expected:
                if not (r.curves.get("split") == split).any():
                    add(r.run_name, f"no '{split}' rows in local curves")

            wb = (
                self._wandb.loc[r.run_name] if r.run_name in self._wandb.index else None
            )
            if wb is None:
                continue

            # the only like-for-like comparison: final train/val loss
            local_final = r.final_loss
            for split in ("train", "val"):
                key = f"final_{split}_loss"
                if key not in wb.index or not _is_number(wb[key]):
                    continue
                if split not in local_final:
                    add(r.run_name, f"no local {split} loss to compare")
                    continue
                a, b = float(local_final[split]), float(wb[key])
                if not np.isclose(a, b, rtol=rtol, atol=atol):
                    add(
                        r.run_name,
                        f"final {split} loss: local {a:.6g} vs W&B {b:.6g}",
                    )

            # stop_epoch, where the Trainer logged it, must match the local argmin
            if _is_number(wb.get("stop_epoch")) and r.best_epoch is not None:
                if int(wb["stop_epoch"]) != r.best_epoch:
                    add(
                        r.run_name,
                        f"stop_epoch: W&B {int(wb['stop_epoch'])} vs "
                        f"local argmin {r.best_epoch}",
                    )

            # the eval-stage metrics are the point of fetching W&B at all
            if not any(_parse_col(c)[0] for c in wb.index if _is_number(wb[c])):
                add(r.run_name, "no best_* metrics in W&B summary")

        out = pd.DataFrame(issues, columns=["run_name", "issue"])
        print(f"{len(out)} issue(s) found." if len(out) else "No issues found.")
        return out

    # -- persistence ------------------------------------------------------ #

    def save(self, path):
        """Materialise every frame and pickle the parser.

        The reloaded object needs neither the results directory nor W&B access.
        """
        for attr in ("configs", "curves", "best_wide", "best_long", "summary"):
            getattr(self, attr)
        for r in self.runs:
            _ = r.curves
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if self.verbose:
            size = os.path.getsize(path) / 1024**2
            print(f"Saved {len(self.runs)} run(s) to {path} ({size:.1f} MB)")
        return path

    @classmethod
    def load(cls, path):
        """Reload a parser pickled by :meth:`save`."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not hold a {cls.__name__}")
        obj.from_cache = True
        return obj

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_fetcher", None)
        return state
