import cuml
import cupy
import glob
import inspect
import joblib
import json
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import wandb
import warnings

from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    PredefinedSplit,
    RandomizedSearchCV,
)

from models.evaluate import MetricCollection, CovariateEvaluator
from models.patchwise.data import S2_Dataset
from models.paths import data_dir_of, resolve_paths
from models.pixelwise.data import (
    TimeSeriesDatasetCreator,
    TimeSeriesDataset,
    TimeSeriesInferenceSet,
)
from models.pixelwise.machine_learning.model import MLModelConfigurator


class Trainer:
    """Pixelwise machine learning model training & evaluation on a data set of choice"""

    def __init__(
        self,
        # -- data loading parameters (see data.py & crossval.py for details) --
        meta_file: str = None,
        sel_time: list = S2_Dataset.TIME_ORDER,
        sel_band: list = S2_Dataset.BAND_ORDER,
        sel_task: str = "lvl1_bic",
        split_strategy: str = "random",
        split_subset: float = 1.0,
        split_trainsize: float = 0.7,
        split_fold: int = 0,
        split_kwargs: dict = None,
        use_cache: str = "disk",
        n_samples: int = 1000000,
        shard_size_mb: float = 1024,
        # -- model training parameters (see model.py for details) --
        model_name: str = "rf",
        model_params: dict = {},
        scale_features: str = "auto",
        pred_batch_size: int = 250000,
        # -- hyperparameter tuning --
        tuning: str = None,
        tuning_space: dict = None,
        tuning_metric: str = "jaccard_macro",
        tuning_iter: int = 25,
        tuning_subset: int = 100000,
        seed: int = 42,
        # -- logging / output --
        save_dir: str = None,
        wandb_project: str = None,
        wandb_runname: str = None,
        wandb_tags: list = [],
        wandb_notes: str = None,
    ):
        """
        Define model training & logging parameters

        -- data loading parameters (see data.py & crossval.py for details) --
        meta_file (str): Path to CSV file containing metadata for all input tiles
            and their associated labels. The CSV is expected to contain:
            - "path_input" (str): path to the input raster tile (Sentinel-2)
            - "path_label" (str): path to the corresponding label tile (habitat map)
            Relative paths are interpreted relative to the CSV's folder.
        sel_time (list, optional): Time steps to select from the dataset.
        sel_band (list, optional): Bands to select from the dataset.
        sel_task (str, optional): Task type for the dataset. Default "lvl1_bic".
        use_cache (str, optional): Caching mode for the dataset. Default "disk".
        split_strategy (str, optional): Train/val/test split strategy. Default "random".
        split_trainsize (float, optional): Fraction of the dataset used for training. Default 0.7.
        split_fold (int, optional): Ignored, kept for backward compatibility. Splits
            differ by `seed` only. Default 0.
        split_subset (float, optional): Fraction of the total dataset to use. Default 1.0.
        split_kwargs (dict, optional): Additional keyword arguments for the split strategy. Default None.
        n_samples (int, optional): Number of pixel stacks to create from the patch-based dataset.
            Default 1000000. If None, the entire patch-based dataset will be converted to pixel
            stacks. Note that "knn" and especially "svm" do not scale to millions of samples,
            see MLModelConfigurator.MAX_SAMPLES for guidance.
        shard_size_mb (float, optional): Size of the shards to create when converting the patch-based dataset
            to pixel stacks. None means no sharding will be applied.

        -- model training parameters --
        model_name (str, optional): model to be trained, see model.py for available models.
            Defaults to "rf". Choose from "rf", "knn", "svm", "log".
        model_params (dict, optional): hyperparameters overriding the standard configuration
            of model.py, e.g. {"n_estimators": 500}. When tuning is enabled these act as the
            fixed base configuration on top of which the search space is applied.
            Defaults to {} (i.e. the standard configuration is used as is)
        scale_features (str|bool, optional): currently without effect. The inputs are
            already standardised channel-wise by S2_Dataset, so the additional scaling of
            the flattened features is disabled (see _fit_scaler). Kept for backward
            compatibility of configs and checkpoints. Defaults to "auto"
        pred_batch_size (int, optional): number of pixels pushed through the model per
            inference call, limits GPU memory during prediction. Defaults to 250000

        -- hyperparameter tuning --
        tuning (str, optional): search strategy, one of None, "grid", "random".
            If None, the model is fitted once with the standard configuration defined in
            model.py (optionally updated by model_params). Defaults to None
        tuning_space (dict, optional): custom search space, e.g. {"C": [0.1, 1, 10]}.
            Entries replace the corresponding entries of the space defined in model.py,
            additional entries are added to it. Defaults to None
        tuning_metric (str, optional): sklearn scorer used to rank the candidates.
            Defaults to "jaccard_macro"
        tuning_iter (int, optional): number of sampled candidates for "random".
            Defaults to 25
        tuning_subset (int, optional): maximum number of (stratified) training samples used
            during the search; the best configuration is refitted on the full training split
            afterwards. If None, all training samples are used. Defaults to 100000
        seed (int, optional): seed to reproduce results
            Defaults to 42

        -- logging / output --
        save_dir (str): directory to save results for model run
        wandb_project (str): project name for logging results with wandb
        wandb_runname (str): name current run logged with wandb
        wandb_tags (list, optional): tags for current run
        wandb_notes (str, optional): additional notes/description for current run
        """
        self.meta_file = meta_file
        self.sel_time = sel_time
        self.sel_band = sel_band
        self.sel_task = sel_task
        self.use_cache = use_cache
        self.split_strategy = split_strategy
        self.split_trainsize = split_trainsize
        self.split_fold = split_fold
        self.split_subset = split_subset
        self.split_kwargs = split_kwargs or {}
        self.n_samples = n_samples
        self.shard_size_mb = shard_size_mb
        self.model_name = model_name
        self.model_params = dict(model_params)
        self.scale_features = scale_features
        self.pred_batch_size = pred_batch_size
        self.tuning = tuning
        self.tuning_space = tuning_space
        self.tuning_metric = tuning_metric
        self.tuning_iter = tuning_iter
        self.tuning_subset = tuning_subset
        self.seed = seed
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.wandb_runname = wandb_runname
        self.wandb_tags = list(wandb_tags or [])
        self.wandb_notes = wandb_notes
        # validate the tuning strategy early, a search may run for hours
        if self.tuning not in [None, "grid", "random"]:
            raise ValueError(
                f"Tuning strategy '{self.tuning}' is not available."
                " Choose from 'grid', 'random' or None."
            )
        # create save folder if necessary
        os.makedirs(self.save_dir, exist_ok=True)
        # set seed if specified
        if self.seed is not None:
            os.environ["PYTHONHASHSEED"] = str(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cupy.random.seed(self.seed)
        # check if GPU is available
        assert torch.cuda.is_available()
        # let cuML hand back numpy arrays
        cuml.set_global_output_type("numpy")
        # feature scaling statistics, populated in prep_model()
        self._mu, self._sigma = None, None
        # init logger
        if self.wandb_project is None:
            self.logger = None
        else:
            wandb.init(
                project=self.wandb_project,
                dir=self.save_dir,
                notes=self.wandb_notes,
                settings=wandb.Settings(
                    mode="online",
                    console="off",
                    x_stats_sampling_interval=60.0,
                    x_file_stream_transmit_interval=60.0,
                ),
            )
            wandb.run.name = self.wandb_runname
            wandb.run.tags = tuple(self.wandb_tags)
            self.logger = wandb.run
        self._update_logs()
        self.save_config()

    def save_config(self, path=None):
        """Persist the constructor arguments needed to rebuild this Trainer."""
        init_params = inspect.signature(type(self).__init__).parameters
        cfg = {
            name: getattr(self, name)
            for name in init_params
            if name != "self" and hasattr(self, name)
        }
        path = path or os.path.join(self.save_dir, "config.json")
        if os.path.exists(path):
            warn_msg = f"Config file already exists at {path}. Not overwriting."
            warnings.warn(warn_msg)
        else:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2, default=str)
        return path

    @classmethod
    def from_config(cls, path, **overrides):
        """Rebuild a Trainer from a saved config, with optional overrides."""
        with open(path) as f:
            cfg = json.load(f)
        cfg.update(overrides)
        return cls(**cfg)

    def prep_model(self):
        """
        Initialize model, dataset and feature preprocessing
        """
        # load time series data & labels
        tsdc = TimeSeriesDatasetCreator(
            meta_file=self.meta_file,
            sel_time=self.sel_time,
            sel_band=self.sel_band,
            sel_task=self.sel_task,
            use_cache=self.use_cache,
            cache_dir=os.path.dirname(os.path.abspath(self.meta_file)),
            split_strategy=self.split_strategy,
            split_subset=self.split_subset,
            split_trainsize=self.split_trainsize,
            split_fold=self.split_fold,
            split_kwargs=self.split_kwargs,
            sample_size=self.n_samples,
            seed=self.seed,
            read_covars=True,
            shard_size_mb=self.shard_size_mb,
        )
        self.ds = TimeSeriesDataset(tsdc.run())
        # decide if binary or not
        self.binary = len(self.ds.labels_dict) == 2
        if self.binary:
            mode, num_classes = "binary", 1
        else:
            mode, num_classes = "multiclass", len(self.ds.labels_dict)
        self.metric_collection = MetricCollection(mode=mode, num_classes=num_classes)
        # label encoding: cuML expects consecutive class indices 0 ... n_classes-1
        self.labels_dict = dict(self.ds.labels_dict)
        self.labels_inv = {v: k for k, v in self.labels_dict.items()}
        # feature preprocessing (flattening happens on the fly in _prep_X)
        _, n_features, n_timesteps = self.ds.X_train.shape
        self.n_features, self.n_timesteps = n_features, n_timesteps
        self._fit_scaler()
        # initialise model with the standard configuration + user overrides
        self.mmc = MLModelConfigurator(self.model_name, random_state=self.seed)
        self.model_args = self.mmc.resolve_params(**dict(self.model_params))
        self._model = self.mmc.init_model(**self.model_args)
        self.mmc.check_sample_size(len(self.ds.y_train))
        # get model summary
        self._update_logs()

    def tune_model(self):
        """
        Search the hyperparameter space of the configured model.

        The search runs on a stratified subsample of the training split
        (`tuning_subset`) and is scored on the validation split of the
        dataset. The winning configuration is not refitted here -
        train_model() refits it on the full training split.

        Returns:
            best_params (dict): standard configuration updated with the best
            values found during the search
        """
        if not hasattr(self, "ds"):
            self.prep_model()
        space = self.mmc.search_space(self.tuning, extend=self.tuning_space)

        # a) assemble the search data
        X_tr, y_tr = self._subsample(
            self.ds.X_train, self.ds.y_train, self.tuning_subset
        )
        X_va, y_va = self._subsample(self.ds.X_val, self.ds.y_val, self.tuning_subset)
        X_tr, y_tr = self._prep_X(X_tr), self._encode(y_tr)
        X_va, y_va = self._prep_X(X_va), self._encode(y_va)
        X_search = np.concatenate([X_tr, X_va], axis=0)
        y_search = np.concatenate([y_tr, y_va], axis=0)

        # define split
        # note: could be optimized by using cross validation to reliably measure the effect
        # of a given hyperparameter configuration, currently omitted due to spatial correlation
        # of samples: train/val must be kept separate to be drawn from different tiles
        test_fold = np.concatenate([np.full(len(y_tr), -1), np.zeros(len(y_va))])
        test_fold = test_fold.astype(int)
        cv = PredefinedSplit(test_fold)

        # b) run the search
        base = self.mmc.init_model(**self.model_args)
        search_args = dict(
            scoring=self.tuning_metric,
            cv=cv,
            refit=False,
            n_jobs=1,
            verbose=2,
            error_score=np.nan,
            return_train_score=False,
        )
        if self.tuning == "grid":
            search = GridSearchCV(base, space, **search_args)
        elif self.tuning == "random":
            search = RandomizedSearchCV(
                base,
                space,
                n_iter=self.tuning_iter,
                random_state=42,
                **search_args,
            )
        print(f"🔍 {self.tuning} search over {len(y_search):,} samples ...")
        t_start = time.time()
        search.fit(X_search, y_search)
        self.tuning_time = round(time.time() - t_start, 2)

        # c) benchmark the winner against the standard configuration
        best_params = dict(self.model_args)
        best_params.update(search.best_params_)
        self.best_score = float(search.best_score_)
        self.best_score_std = float(
            search.cv_results_["std_test_score"][search.best_index_]
        )
        default_params = {k: self.model_args[k] for k in space if k in self.model_args}
        hit = next(
            (
                i
                for i, p in enumerate(search.cv_results_["params"])
                if p == default_params
            ),
            None,
        )
        # the default was part of the search -> reuse the score
        if hit is not None:
            self.default_score = float(search.cv_results_["mean_test_score"][hit])
            self.default_score_std = float(search.cv_results_["std_test_score"][hit])
        # otherwise score it once on exactly the same split(s)
        else:
            self.default_score, self.default_score_std = self._score_params(
                self.model_args, X_search, y_search, cv
            )
        self.tuning_gain = self.best_score - self.default_score
        self.tuning_gain_pct = (
            100 * self.tuning_gain / abs(self.default_score)
            if self.default_score
            else float("nan")
        )
        changed = {
            k: v
            for k, v in search.best_params_.items()
            if k not in default_params or default_params[k] != v
        }

        # d) report default vs. tuned configuration
        pad = max(len(self.tuning_metric), 8)
        arrow = "📈" if self.tuning_gain > 0 else "📉"
        print(f"\n{'─' * 72}")
        print(
            f"📊 default {self.tuning_metric:<{pad}} : {self.default_score:.4f}"
            f" ± {self.default_score_std:.4f}   {dict(sorted(default_params.items()))}"
        )
        print(
            f"✅ tuned   {self.tuning_metric:<{pad}} : {self.best_score:.4f}"
            f" ± {self.best_score_std:.4f}   {dict(sorted(search.best_params_.items()))}"
        )
        print(
            f"{arrow} gain    {'':<{pad}} : {self.tuning_gain:+.4f}"
            f" ({self.tuning_gain_pct:+.2f} %) in {self.tuning_time:.1f} s"
        )
        print(f"{'─' * 72}\n")

        if self.tuning_gain <= 0:
            warnings.warn(
                f"Tuning did not improve on the standard configuration"
                f" ({self.best_score:.4f} <= {self.default_score:.4f})"
                " - keeping the defaults."
            )
            best_params, changed = dict(self.model_args), {}

        # e) persist & log the search results
        results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        results["is_default"] = [
            p == default_params for p in results["params"].tolist()
        ]
        results.to_csv(os.path.join(self.save_dir, "tuning_results.csv"), index=False)
        summary = {
            "model_name": self.model_name,
            "tuning": self.tuning,
            "metric": self.tuning_metric,
            "n_candidates": int(len(search.cv_results_["params"])),
            "tuning_time": self.tuning_time,
            "default_score": self.default_score,
            "default_score_std": self.default_score_std,
            "best_score": self.best_score,
            "best_score_std": self.best_score_std,
            "gain": self.tuning_gain,
            "gain_pct": self.tuning_gain_pct,
            "changed_params": changed,
            "default_params": self.model_args,
            "best_params": best_params,
        }
        with open(os.path.join(self.save_dir, "tuning_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(os.path.join(self.save_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2, default=str)
        if self.wandb_project is not None:
            wandb.run.summary["default_cv_score"] = self.default_score
            wandb.run.summary["best_cv_score"] = self.best_score
            wandb.run.summary["cv_score_gain"] = self.tuning_gain
            wandb.run.summary["cv_score_gain_pct"] = self.tuning_gain_pct
            wandb.run.summary["tuning_time"] = self.tuning_time
        return best_params

    def train_model(self):
        """Optional hyperparameter tuning, training & evaluation of a model"""
        completed = False
        try:
            if not hasattr(self, "ds"):
                self.prep_model()

            # a) find the hyperparameters to train with
            self.tuning_time = 0.0
            if self.tuning:
                self.best_params = self.tune_model()
            else:
                # standard configuration from model.py (+ model_params overrides)
                self.best_params = dict(self.model_args)
            self._model = self.mmc.init_model(**self.best_params)

            # b) fit on the full training split
            print(
                f"⏳ Fitting {self.model_name} on {len(self.ds.y_train):,} samples ..."
            )
            X_train = self._prep_X(self.ds.X_train)
            y_train = self._encode(self.ds.y_train)
            t_start = time.time()
            self._model.fit(X_train, y_train)
            t_end = time.time()

            # evaluate training time
            self.training_samples = int(len(y_train))
            self.training_time = round(t_end - t_start, 2)

            # persist the fitted model
            self.ckpt_path = self._save_clf()

            # evaluate trained model on train, val & test set
            self.eval_model()
            self._update_logs()

            # finalize
            completed = True
            with open(os.path.join(self.save_dir, "training_completed.txt"), "w") as f:
                f.write("True")

        finally:
            if self.wandb_project is not None:
                wandb.finish(exit_code=0 if completed else 1)

    def eval_model(self, phases=("train", "val", "test"), model_params=None):
        """
        Evaluate a (trained) model on the requested dataset splits.

        Args:
            phases (tuple, optional): splits to evaluate on.
                Defaults to ("train", "val", "test")
            model_params (str, optional): path to a saved model. If given, this
                model is loaded first, which allows evaluating a finished run
                without retraining it.

        Returns:
            metrics (DataFrame): one row of scalar metrics per split
        """
        if model_params is not None:
            self._load_clf(model_params)
        elif not hasattr(self, "_model"):
            self.prep_model()

        rows = []
        self.metric_collection.set_stage("eval")
        for phase in phases:
            print(f"Evaluating on {phase} split ...")

            # get data for current phase
            if phase == "train":
                X = self.ds.X_train
                y = self.ds.y_train
            elif phase == "val":
                X = self.ds.X_val
                y = self.ds.y_val
            elif phase == "test":
                X = self.ds.X_test
                y = self.ds.y_test

            # run model (predictions are returned in the raw label space)
            X = self._prep_X(X)
            y_pred = self._predict(X, prepped=True)
            y_probs = self._predict_proba(X, prepped=True)

            # check that argmax of probabilities matches the predicted class indices
            if not self.binary:
                y_pred_from_probs = np.argmax(y_probs, axis=1)
                if not np.all(y_pred == y_pred_from_probs):
                    warnings.warn(
                        "Predicted class indices do not match argmax of probabilities."
                    )

            # convert to pytorch tensors
            y_pred = torch.tensor(np.asarray(y_pred), dtype=torch.long)
            y_pred = y_pred.reshape(-1, 1)
            y_probs = torch.tensor(np.asarray(y_probs), dtype=torch.float32).unsqueeze(
                -1
            )
            y_true = torch.tensor(np.asarray(y), dtype=torch.long).reshape(-1, 1)

            if self.binary:
                y_probs = y_probs.select(1, 1).unsqueeze(1)
                y_pred = y_pred.unsqueeze(1)
                y_true = y_true.unsqueeze(1)

            # map labels to encoded indices
            y_true_mapped = torch.zeros_like(y_true)
            for k, v in self.ds.labels_dict.items():
                y_true_mapped[y_true == k] = v
            y_pred_mapped = torch.zeros_like(y_pred)
            for k, v in self.ds.labels_dict.items():
                y_pred_mapped[y_pred == k] = v
            y_pred = y_pred_mapped
            y_true = y_true_mapped

            # compute & log metrics
            self.metric_collection.reset()
            self.metric_collection.update(y_pred, y_probs, y_true)
            self.metric_collection.compute(return_dict=True)
            row = {"phase": phase}
            for mk, mv in self.metric_collection.results.items():
                if isinstance(mv, float) or isinstance(mv, int):
                    row[mk] = mv
                    if self.wandb_project is not None:
                        wandb.run.summary[f"best_{mk}_{phase}"] = mv
            rows.append(row)

            # plot confusion matrix & report
            self.metric_collection.plot_confusion_matrix(
                out_dir=os.path.join(self.save_dir, phase),
                class_names=self.ds.labels_str,
                log_scaled=True,
            )
            self.metric_collection.plot_report(
                out_dir=os.path.join(self.save_dir, phase),
                class_names=self.ds.labels_str,
            )

            # evaluate covariate performance
            if phase in ("val", "test"):
                _ = CovariateEvaluator.from_pixelwise(
                    config=os.path.join(self.save_dir, "config.json"),
                    split=phase,
                    out_dir=os.path.join(self.save_dir, phase, "covariates"),
                )

        metrics = pd.DataFrame(rows)
        metrics.to_csv(os.path.join(self.save_dir, "model_metrics.csv"), index=False)
        return metrics

    def apply_model(
        self,
        meta_df: pd.DataFrame = None,
        model_params: str = None,
        return_res: bool = False,
    ):
        """
        Run inference on a set of samples using the trained model.

        Args:
            meta_df (DataFrame): Metadata rows of the patches to predict, e.g. rows of the
                meta CSV; relative paths are resolved against the folder of `self.meta_file`.
                Required.
            model_params (str, optional): Path to model checkpoint for loading weights.
                If None, uses the model already loaded during training.
            return_res (bool): Whether to return the inference results as a dict of
                indices, inputs, targets, predictions & probabilities. Default is False.

        Returns:
            (res, ds) (tuple, optional): Tuple containing inference results and the dataset
            if return_res is True. Inference keys are:
                * indices (Tensor): Indices of the samples in the dataset (N,).
                * inputs (Tensor): Input data (N, C, H, W).
                * targets (Tensor): Ground truth labels (N, H, W).
                * preds (Tensor): Predicted class indices (N, H, W).
                * probs (Tensor): Probabilities (N, C, H, W).
        """
        # create inference dataset from meta_df
        if meta_df is None:
            raise ValueError(
                "meta_df is required: pass the metadata rows of the patches to predict."
            )
        meta_df = resolve_paths(meta_df, data_dir_of(self.meta_file))
        tsi = TimeSeriesInferenceSet(
            meta_file=meta_df,
            sel_time=self.sel_time,
            sel_band=self.sel_band,
            sel_task=self.sel_task,
            seed=self.seed,
        )

        # load model & run inference
        clf = self._load_clf(model_params)
        y_pred = self._predict(tsi.X_test, clf=clf)
        y_prob = self._predict_proba(tsi.X_test, clf=clf)

        # convert predictions and probabilities back to per-tile maps
        input_maps = tsi.inputs_to_maps()
        pred_maps = tsi.predictions_to_maps(tsi.decode(y_pred))
        prob_maps = tsi.probs_to_maps(y_prob)
        gt_maps = tsi.targets_to_maps()

        # assemble a res-like dictionary as output for plotting / evaluation
        res = {
            "indices": torch.arange(len(tsi.shapes)),
            "inputs": torch.tensor(np.stack(input_maps)),
            "targets": torch.tensor(np.stack(gt_maps)),
            "preds": torch.tensor(np.stack(pred_maps)),
            "probs": torch.tensor(np.stack(prob_maps)),
        }
        if return_res:
            return res, tsi.ds
        else:
            return None

    # ------------------------------------------------------------------ #
    # data & label handling
    # ------------------------------------------------------------------ #
    def _fit_scaler(self, skip=True):
        """
        Compute the feature standardisation statistics on the training split.
        Tree based models are scale invariant, so scaling is skipped for them
        when `scale_features="auto"`.

        Arg: skip (bool): if True, skip scaling. Set to true since features are
        already scaled in the dataset creation step.
        """
        if skip:
            return
        if self.scale_features == "auto":
            scale = self.model_name in MLModelConfigurator.NEEDS_SCALING
        else:
            scale = bool(self.scale_features)
        if not scale:
            self._mu, self._sigma = None, None
            return
        X = np.asarray(self.ds.X_train).reshape(self.ds.X_train.shape[0], -1)
        self._mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        self._sigma = X.std(axis=0, dtype=np.float64).astype(np.float32)
        self._sigma[self._sigma < 1e-6] = 1.0

    def _prep_X(self, X):
        """
        Flatten (N, n_bands, n_timesteps) -> (N, n_bands * n_timesteps), cast to
        the float32 C-contiguous layout cuML expects and apply standardisation.
        """
        X = np.asarray(X)
        X = X.reshape(X.shape[0], -1).astype(np.float32, copy=True)
        if not np.isfinite(X).all():
            warnings.warn("Non-finite feature values encountered - set to 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self._mu is not None:
            X -= self._mu
            X /= self._sigma
        return np.ascontiguousarray(X)

    def _encode(self, y):
        """Map raw label values to consecutive class indices (cuML requirement)."""
        y = np.asarray(y).ravel()
        out = np.full(y.shape, -1, dtype=np.int32)
        for k, v in self.labels_dict.items():
            out[y == k] = v
        if (out < 0).any():
            raise ValueError("❌ Label values encountered that are not in labels_dict.")
        return out

    def _decode(self, y):
        """Map consecutive class indices back to the raw label values."""
        y = np.asarray(y).ravel()
        out = np.zeros(y.shape, dtype=np.int64)
        for v, k in self.labels_inv.items():
            out[y == v] = k
        return out

    def _subsample(self, X, y, n_samples):
        """Draw a class-stratified subsample, keeping at least one sample per class."""
        if not n_samples or n_samples >= len(y):
            return X, y
        rng = np.random.default_rng(self.seed)
        y_flat = np.asarray(y).ravel()
        idx = np.arange(len(y_flat))
        frac = n_samples / len(y_flat)
        sel = []
        for c in np.unique(y_flat):
            c_idx = idx[y_flat == c]
            k = min(len(c_idx), max(1, int(round(len(c_idx) * frac))))
            sel.append(rng.choice(c_idx, size=k, replace=False))
        sel = np.sort(np.concatenate(sel))
        return X[sel], y_flat[sel]

    def _score_params(self, params, X, y, cv):
        """
        Cross-validated score of a single configuration on the search split(s).

        Used to put the tuning result in relation to the standard configuration
        of model.py. Scoring reuses the very same `cv` object as the search, so
        the two numbers are directly comparable.

        Args:
            params (dict): full parameter set to evaluate
            X (ndarray): prepared feature matrix used during the search
            y (ndarray): encoded labels used during the search
            cv: the splitter the search was scored with

        Returns:
            (mean, std) (tuple): mean and standard deviation across the folds
        """
        scores = cross_val_score(
            self.mmc.init_model(**params),
            X,
            y,
            scoring=self.tuning_metric,
            cv=cv,
            n_jobs=1,
            error_score=np.nan,
        )
        return float(np.nanmean(scores)), float(np.nanstd(scores))

    # ------------------------------------------------------------------ #
    # inference & persistence
    # ------------------------------------------------------------------ #
    def _predict(self, X, clf=None, prepped=False):
        """Batched prediction, returned in the raw label space."""
        clf = clf if clf is not None else self._model
        X = X if prepped else self._prep_X(X)
        out = [
            np.asarray(clf.predict(X[i : i + self.pred_batch_size]))
            for i in range(0, len(X), self.pred_batch_size)
        ]
        return self._decode(np.concatenate(out).ravel().astype(np.int64))

    def _predict_proba(self, X, clf=None, prepped=False):
        """
        Batched class probabilities, columns ordered by encoded class index.

        Estimators only know the classes seen during fitting, so the raw output
        is (N, n_classes_in_train). It is expanded to (N, n_classes) here, with
        an all-zero column for every class absent from the training split.
        """
        clf = clf if clf is not None else self._model
        X = X if prepped else self._prep_X(X)
        n_classes = len(self.labels_dict)
        # guard against estimators returning a different class ordering
        out = [
            np.asarray(clf.predict_proba(X[i : i + self.pred_batch_size]))
            for i in range(0, len(X), self.pred_batch_size)
        ]
        probs = np.concatenate(out, axis=0).astype(np.float32)
        classes = getattr(clf, "classes_", None)
        if classes is None:
            if probs.shape[1] != n_classes:
                raise ValueError(
                    f"❌ {type(clf).__name__} returned {probs.shape[1]} probability"
                    f" columns for {n_classes} classes and exposes no `classes_`"
                    " to align them with."
                )
            return probs
        # re-order and pad according to the full label space, filling absent classes with zeros
        classes = np.asarray(classes).ravel().astype(int)
        full = np.zeros((len(probs), n_classes), dtype=np.float32)
        full[:, classes] = probs
        return full

    def _save_clf(self, path=None):
        """Store the fitted estimator together with its preprocessing state."""
        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = path or os.path.join(ckpt_dir, "model.joblib")
        joblib.dump(
            {
                "model": self._model,
                "model_name": self.model_name,
                "params": self.best_params,
                "labels_dict": self.labels_dict,
                "input_shape": (self.n_features, self.n_timesteps),
                "scaler": (
                    None if self._mu is None else {"mean": self._mu, "std": self._sigma}
                ),
            },
            path,
        )
        return path

    def _load_clf(self, weights_path=None):
        """Restore a fitted estimator incl. its label encoding & feature scaling.

        Unlike the deep learning branch no zero-epoch fit is needed here, since
        the pickled estimator already carries all fitted state. The dataset is
        rebuilt though, as the evaluation utilities rely on it.
        """
        if not hasattr(self, "ds"):
            self.prep_model()
        weights_path = weights_path or self._resolve_checkpoint()
        artefact = joblib.load(weights_path)
        if isinstance(artefact, dict) and "model" in artefact:
            self._model = artefact["model"]
            self.best_params = artefact.get("params", getattr(self, "model_args", {}))
            if artefact.get("labels_dict"):
                self.labels_dict = artefact["labels_dict"]
                self.labels_inv = {v: k for k, v in self.labels_dict.items()}
            scaler = artefact.get("scaler")
            if scaler is not None:
                self._mu, self._sigma = scaler["mean"], scaler["std"]
        else:  # a bare estimator was pickled
            self._model = artefact
        return self._model

    def _resolve_checkpoint(self):
        """Locate the saved estimator, mirroring the ordering logic used elsewhere."""
        # single-file convention takes precedence
        single = os.path.join(self.save_dir, "checkpoints", "model.joblib")
        if os.path.exists(single):
            return single
        ckpts = glob.glob(os.path.join(self.save_dir, "**", "*.joblib"), recursive=True)
        if not ckpts:
            raise FileNotFoundError(f"No .joblib checkpoint found in {self.save_dir}")
        return max(ckpts, key=os.path.getmtime)

    def _update_logs(self):
        """
        Write config logs to disk & update WandB cloud logs
        """
        if self.wandb_project is not None:
            configs = {
                k: v
                for k, v in self.__dict__.items()
                if type(v) in [str, int, float, bool, dict, list]
                and k
                not in [
                    "dataloaders",
                    "fit_args",
                    "loss_history",
                    "metric_history",
                    "model_args",
                    "labels_dict",
                    "labels_inv",
                ]
            }
            wandb.config.update(configs, allow_val_change=True)
