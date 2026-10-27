import inspect
import glob
import json
import keras
import math
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import random
import tensorflow as tf
import time
import torch
import wandb
import warnings

from lightning.pytorch.loggers import WandbLogger, CSVLogger
from sklearn.utils import class_weight
from sktime.classification.deep_learning._callbacks import TfWandbMetricsLogger

from models.evaluate import CovariateEvaluator, MetricCollection
from models.patchwise.data import S2_Dataset
from models.paths import data_dir_of, resolve_paths
from models.pixelwise.data import (
    TimeSeriesDatasetCreator,
    TimeSeriesDataset,
    TimeSeriesInferenceSet,
)
from models.pixelwise.deep_learning.loss import SegmentationLoss
from models.pixelwise.deep_learning.model import TempModelConfigurator
from models.pixelwise.deep_learning.utils import (
    tf_make_cosine_callback,
    TorchCosineWarmupLR,
    TFLossLogger,
    TFMetricsLogger,
    TorchLogsPostprocessor,
)


class Trainer:
    """Pixelwise timeseries model training & evaluation on a data set of choice"""

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
        # -- model training parameters (see model.py & loss.py for details) --
        model_name: str = "fcn",
        batch_size: int = 64,
        pred_batch_size: int = 2048,
        optimizer: str = "Adam",
        loss: str = "DiceLoss",
        loss_params: dict = {},
        lr: float = 1e-5,
        max_epochs: int = 100,
        patience: int = 50,
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
            Default 1000000. If None, the entire patch-based dataset will be converted to pixel stacks.
        shard_size_mb (float, optional): Size of the shards to create when converting the patch-based dataset
            to pixel stacks. None means no sharding will be applied.

        -- model training parameters --
        model_name (str, optional): model to be trained, see model.py for available models
            Defaults to "fcn"
        batch_size (int, optional) : batch size used for training, choose a multiple of 2
            Defaults to 64
        pred_batch_size (int, optional): prediction batch size used for val/test, choose a multiple of 2
            Defaults to 2048
        optimizer (str, optional): optimizer used for training, available optimizers are "SGD", "Adam"
            Default to "Adam"
        loss (str, optional): loss function used for training, see loss.py for available losses
            Defaults to "DiceLoss"
        loss_params (dict, optional): additional optional loss params - e.g. smooth=1
            Defaults to {}
        lr (float, optional): learning rate used for training
            Defaults to 1e-5
        max_epochs (int, optional): maximum number of epochs used for training
            Defaults to 100
        patience (int, optional): patience for early stopping on validation data
            Defaults to 50
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
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.loss_params = dict(loss_params or {})
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.wandb_runname = wandb_runname
        self.wandb_tags = list(wandb_tags or [])
        self.wandb_notes = wandb_notes
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
            keras.utils.set_random_seed(self.seed)
            tf.random.set_seed(self.seed)
            tf.config.experimental.enable_op_determinism()
        # check if GPU is available
        assert tf.test.is_built_with_cuda()
        assert len(tf.config.list_physical_devices("GPU"))
        assert torch.cuda.is_available()
        # init logger
        library = TempModelConfigurator(self.model_name).library
        if self.wandb_project is None:
            self.logger = None
        elif library == "tensorflow":
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
            self.logger = TfWandbMetricsLogger()
        elif library == "pytorch":
            self.logger = WandbLogger(
                project=self.wandb_project,
                save_dir=self.save_dir,
                notes=self.wandb_notes,
                name=self.wandb_runname,
            )
            self.logger.experiment.tags = self.wandb_tags
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
            warn_msg = f"⚠️ Config file already exists at {path}. Not overwriting."
            warnings.warn(warn_msg)
        else:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
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
        Initialize model and dataloaders
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
        # retrieve model init & fit args
        library = TempModelConfigurator(self.model_name).library
        if library == "tensorflow":
            self.model_args, self.fit_args = self._config_tf()
        elif library == "pytorch":
            self.model_args, self.fit_args = self._config_torch()
        # initialise model
        _, n_features, n_timesteps = self.ds.X_train.shape
        tmd = TempModelConfigurator(
            self.model_name,
            n_features=n_features,
            n_timesteps=n_timesteps,
            num_classes=num_classes if num_classes > 1 else 2,
        )
        self._model = tmd.init_model(**self.model_args)
        # get model summary
        tmd.summarize(verbose=False)
        model_summary = tmd.summary_stats
        self.total_params = model_summary["total_params"]
        self.trainable_params = model_summary["trainable_params"]
        self._update_logs()

    def train_model(self):
        """Train model"""
        try:
            # fit model
            t_start = time.time()
            self._model.fit(
                self.ds.X_train,
                self.ds.y_train,
                self.ds.X_val,
                self.ds.y_val,
                **self.fit_args,
            )
            t_end = time.time()

            # get model summary & save training logs
            model_summary = pd.read_csv(os.path.join(self.save_dir, "model_loss.csv"))

            # evaluate training time
            self.training_epochs = len(model_summary)
            self.training_time_total = round(t_end - t_start, 2)
            self.training_time_epoch = round(
                self.training_time_total / self.training_epochs, 2
            )
            self._update_logs()

            # log stopping epoch
            best_idx = np.argmin(np.array(model_summary["val"]))
            if wandb.run is not None:
                wandb.run.summary["stop_epoch"] = best_idx

            # evaluate trained model on train, val & test set
            self.metric_collection.set_stage("eval")
            for phase in ["train", "val", "test"]:
                self._evaluate(phase)

            # save log file training completed
            with open(os.path.join(self.save_dir, "training_completed.txt"), "w") as f:
                f.write("True")

        finally:
            wandb.finish()

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
        chunks = list(self._predict_chunks(tsi.X_test, clf))
        y_pred = np.concatenate([p for _, p, _ in chunks])
        y_prob = np.concatenate([pr for _, _, pr in chunks])

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

    def _config_tf(self):
        """
        Configure TensorFlow model training parameters
        """
        # a) init callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            verbose=1,
            restore_best_weights=False,
        )
        cosine_lr = tf_make_cosine_callback(
            max_update=self.max_epochs,
            base_lr=self.lr,
            final_lr=0.1 * self.lr,
            warmup_steps=5,
            verbose=1,
        )
        save_best_only = True
        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if save_best_only:
            ckpt_path = os.path.join(ckpt_dir, "model.keras")
        else:
            ckpt_path = os.path.join(
                ckpt_dir, "checkpoint-epoch-{epoch:d}-val_loss-{val_loss:.6f}.keras"
            )
        checkpoint = keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=save_best_only,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )
        logger1 = TFLossLogger(os.path.join(self.save_dir, "model_loss.csv"))
        logger2 = TFMetricsLogger(os.path.join(self.save_dir, "model_metrics.csv"))
        callbacks = [
            early_stopping,
            cosine_lr,
            checkpoint,
            self.logger,
            logger1,
            logger2,
        ]
        callbacks = [c for c in callbacks if c is not None]

        # b) init optimizer
        if self.optimizer == "Adam":
            optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        elif self.optimizer == "SGD":
            optimizer = keras.optimizers.SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not available.")

        # c) init loss (& apply class weights)
        loss = SegmentationLoss(
            backend="tf",
            loss_name=self.loss,
            mode="binary" if self.binary else "multiclass",
            **self.loss_params,
        ).fn

        if "weighted" in self.loss_params.keys():
            if self.loss_params["weighted"]:
                # compute class weights
                class_weights = class_weight.compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(self.ds.y_train),
                    y=self.ds.y_train,
                )
                class_weights = dict(zip(np.unique(self.ds.y_train), class_weights))
                # add zeros for labels not in train, only val/test
                for k in self.ds.labels_dict.keys():
                    if k not in class_weights.keys():
                        class_weights.update({k: 0.0})
                class_weights = {k: class_weights[k] for k in self.ds.labels_dict}
                # reformat to class indices
                class_weights = {
                    self.ds.labels_dict[k]: v for k, v in class_weights.items()
                }
            else:
                class_weights = None
        else:
            class_weights = None

        # d) init some metrics to track (these are not the final evaluation metrics)
        # note only few are selected here cause they do not match the final evaluation metrics
        # ep. in binary case the macro F1 is inflated due to 2 outputs
        metrics = [
            keras.metrics.CategoricalAccuracy(name="micro_acc"),
            keras.metrics.F1Score(average="macro", name="macro_f1"),
        ]

        # e) define evaluation interval at least after every 10K train samples
        train_steps = max(
            min(10000 // self.batch_size, len(self.ds.X_train) // self.batch_size), 1
        )
        val_steps = max(
            min(
                10_000_000 // self.pred_batch_size,
                math.ceil(len(self.ds.X_val) / self.pred_batch_size),
            ),
            1,
        )

        # compile all arguments for model init & fit
        init_args = {
            "n_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "pred_batch_size": self.pred_batch_size,
            "metrics": metrics,
            "loss": loss,
            "callbacks": callbacks,
            "optimizer": optimizer,
            "random_state": self.seed,
            "verbose": True,
        }
        fit_args = {
            "target_dict": self.ds.labels_dict,
            "steps_per_epoch": train_steps,
            "validation_steps": val_steps,
        }
        if class_weights is not None:
            fit_args["class_weight"] = class_weights

        return init_args, fit_args

    def _config_torch(self):
        """
        Configure Torch model training parameters
        """
        # a) init callbacks
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience
        )
        checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=self.save_dir, save_top_k=1, monitor="val_loss", mode="min"
        )
        logger = CSVLogger(self.save_dir, name="model_metrics")
        logger_postprocessor = TorchLogsPostprocessor(
            os.path.join(self.save_dir, "model_metrics")
        )
        callbacks = [early_stopping, checkpoint, logger_postprocessor]

        # b) other kwargs
        model_kwargs = dict(
            lr_scheduler=TorchCosineWarmupLR,
            lr_scheduler_kwargs=dict(
                max_update=self.max_epochs,
                base_lr=self.lr,
                final_lr=0.1 * self.lr,
                warmup_steps=5,
                interval="epoch",
            ),
        )

        # c) init optimizer
        if self.optimizer == "Adam":
            optimizer = "Adam"
        elif self.optimizer == "SGD":
            optimizer = "SGD"
        else:
            raise ValueError(f"Optimizer {self.optimizer} not available.")

        # d) init loss (& omit class weights)
        loss = SegmentationLoss(
            backend="torch",
            loss_name=self.loss,
            mode="binary" if self.binary else "multiclass",
            **self.loss_params,
        ).fn

        if "weighted" in self.loss_params.keys():
            self.loss_params.pop("weighted", None)

        # e) define evaluation interval at least after every 10K train samples
        train_steps = max(
            min(10000 // self.batch_size, len(self.ds.X_train) // self.batch_size), 1
        )
        val_steps = max(
            min(
                10_000_000 // self.pred_batch_size,
                math.ceil(len(self.ds.X_val) / self.pred_batch_size),
            ),
            1,
        )

        # compile all arguments for model init & fit
        init_args = {
            "num_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "pred_batch_size": self.pred_batch_size,
            "lr": self.lr,
            "optimizer": optimizer,
            "criterion": loss,
            "callbacks": callbacks,
            "random_state": self.seed,
            "verbose": True,
        }
        init_args.update(model_kwargs)
        fit_args = {
            "target_dict": self.ds.labels_dict,
            "limit_train_batches": train_steps,
            "limit_val_batches": val_steps,
            "logger": [lg for lg in (self.logger, logger) if lg is not None],
        }

        return init_args, fit_args

    def _encode(self, y):
        """Raw labels -> encoded class indices (labels missing in labels_dict -> 0)."""
        y = np.asarray(y)
        out = np.zeros(y.shape, dtype=np.int64)
        for k, v in self.ds.labels_dict.items():
            out[y == k] = v
        return out

    def _evaluate(self, phase, clf=None):
        """Evaluate one split in chunks: metrics, confusion matrix, report & covariates."""
        X = getattr(self.ds, f"X_{phase}")
        y = np.asarray(getattr(self.ds, f"y_{phase}"))
        y_enc = self._encode(y)

        # accumulate metrics chunk by chunk, as in the patchwise apply_model()
        mc = self.metric_collection
        mc.set_stage("eval")
        mc.reset()
        preds_raw = []
        for lo, preds, probs in self._predict_chunks(X, clf):
            hi = lo + len(preds)
            preds_raw.append(preds)
            y_pred = torch.from_numpy(self._encode(preds)).reshape(-1, 1)
            y_true = torch.from_numpy(y_enc[lo:hi]).reshape(-1, 1)
            y_probs = torch.from_numpy(probs).unsqueeze(-1)
            if self.binary:
                y_probs = y_probs.select(1, 1).unsqueeze(1)
                y_pred = y_pred.unsqueeze(1)
                y_true = y_true.unsqueeze(1)
            mc.update(y_pred, y_probs, y_true)
        mc.compute(return_dict=True)
        preds_raw = np.concatenate(preds_raw)

        # log & plot
        if self.wandb_project is not None:
            for mk, mv in mc.results.items():
                if isinstance(mv, (float, int)):
                    wandb.run.summary[f"best_{mk}_{phase}"] = mv
        mc.plot_confusion_matrix(
            out_dir=os.path.join(self.save_dir, phase),
            class_names=self.ds.labels_str,
            log_scaled=True,
        )
        mc.plot_report(
            out_dir=os.path.join(self.save_dir, phase),
            class_names=self.ds.labels_str,
        )

        # evaluate covariate performance
        if phase in ("val", "test"):
            CovariateEvaluator.from_arrays(
                preds_raw,
                self.ds,
                phase,
                out_dir=os.path.join(self.save_dir, phase, "covariates"),
                metrics=self.metric_collection,
            )

        return mc.results

    def _load_clf(self, weights_path=None):
        """Rebuild the estimator, run a 0-epoch fit to populate label-conversion
        state, then load trained weights. Returns the fitted clf.
        """
        if not hasattr(self, "_model"):
            self.prep_model()
        # zero fit to set classes_, _class_dictionary, onehot_encoder & other attrs
        library = TempModelConfigurator(self.model_name).library
        if library == "tensorflow":
            self._model.n_epochs = 0
        else:
            self._model.num_epochs = 0
        fit_args = self.fit_args.copy()
        fit_args.pop("logger", None)
        self._model.fit(self.ds.X_train, self.ds.y_train, **fit_args)
        # load the weights into the model
        if library == "tensorflow":
            if weights_path is None:
                weights_path = self._resolve_tf_checkpoint()
            self._model.model_ = keras.models.load_model(weights_path)
        else:
            if weights_path is None:
                ckpts = glob.glob(os.path.join(self.save_dir, "*.ckpt"))
                if not ckpts:
                    raise FileNotFoundError(f"No .ckpt found in {self.save_dir}")
                weights_path = max(ckpts, key=os.path.getmtime)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(weights_path, weights_only=False, map_location=device)[
                "state_dict"
            ]
            self._model.model.load_state_dict(state)
        return self._model

    def _predict_chunks(self, X, clf=None):
        """Predict X chunk by chunk, like iterating over a dataloader.

        Yields (lo, preds, probs) per chunk: `preds` in the raw label space (identical
        to clf.predict), `probs` (n, n_classes) as returned by clf.predict_proba.
        Only one chunk of inputs/probabilities is on the GPU at a time.

        Note: deliberately not named `_predict`, CovariateEvaluator.from_pixelwise uses
        hasattr(trainer, "_predict") to detect the machine learning trainer.
        """
        clf = clf if clf is not None else self._model
        classes = np.asarray(clf.classes_)
        # multiple of pred_batch_size (no ragged keras batches) and of 100
        # (MetricCollection subsamples heavy metrics with [::100], chunk-invariant only then)
        chunk = 100 * self.pred_batch_size
        for lo in range(0, len(X), chunk):
            probs = np.asarray(clf.predict_proba(X[lo : lo + chunk]), dtype=np.float32)
            # sktime's predict is exactly classes_[argmax(predict_proba)]
            yield lo, classes[probs.argmax(axis=1)], probs

    def _resolve_tf_checkpoint(self):
        """Locate the best Keras checkpoint, mirroring the
        ordering logic used elsewhere in the project."""
        # single-file convention takes precedence
        single = os.path.join(self.save_dir, "checkpoint.model.keras")
        if os.path.exists(single):
            return single

        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".keras")]
        if not ckpts:
            raise FileNotFoundError(f"No .keras checkpoints in {ckpt_dir}")
        # if only one checkpoint, return it
        if len(ckpts) == 1:
            return os.path.join(ckpt_dir, ckpts[0])
        # order by epoch index parsed from "checkpoint-epoch-{N}-val_loss-{L}.keras"
        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[2]))
        val_loss = [
            float(f.split("val_loss-")[1].rsplit(".keras", 1)[0]) for f in ckpts
        ]
        # truncate at the best epoch, as the reference snippet does
        best_pos = val_loss.index(min(val_loss))
        ckpt_paths = [os.path.join(ckpt_dir, f) for f in ckpts[: best_pos + 1]]
        return ckpt_paths[-1]

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
                ]
            }
            wandb.config.update(configs, allow_val_change=True)
