import csv
import keras
import os
import pandas as pd
import pytorch_lightning as pl
import shutil
import torch

from models.patchwise.utils import CosineScheduler


def tf_make_cosine_callback(
    max_update,
    base_lr,
    final_lr=0.0,
    warmup_steps=0,
    verbose=0,
):
    """Expose CosineScheduler through the Keras callback interface."""
    schedule = CosineScheduler(
        max_update=max_update,
        base_lr=base_lr,
        final_lr=final_lr,
        warmup_steps=warmup_steps,
    )

    def _schedule(epoch, lr):
        return float(schedule(min(epoch, max_update)))

    return keras.callbacks.LearningRateScheduler(_schedule, verbose=verbose)


class _CosineFactor:
    """Absolute lr -> multiplicative factor on each param group's initial lr."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, epoch):
        return self.fn(epoch) / self.fn.base_lr_orig


class TorchCosineWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    """CosineScheduler exposed with the torch LR-scheduler signature."""

    def __init__(
        self,
        optimizer,
        max_update,
        base_lr=0.01,
        final_lr=0.0,
        warmup_steps=0,
        last_epoch=-1,
    ):
        fn = CosineScheduler(
            max_update=max_update,
            base_lr=base_lr,
            final_lr=final_lr,
            warmup_steps=warmup_steps,
        )
        super().__init__(optimizer, lr_lambda=_CosineFactor(fn), last_epoch=last_epoch)


class TFLossLogger(keras.callbacks.Callback):
    """Keras callback to log training and validation loss to a CSV file."""

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.append_header = True

    def on_train_begin(self, logs=None):
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["train", "val"])

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([logs.get("loss"), logs.get("val_loss")])


class TFMetricsLogger(keras.callbacks.Callback):
    """
    Keras callback to log per-split evaluation metrics (train/val)
    to a CSV file. Each entry is a string representing the metric dict.
    """

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def on_train_begin(self, logs=None):
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["train", "val"])

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        def extract_metrics(prefix=""):
            """Extract a dict of metrics for one split (e.g. '', 'val_')."""
            metric_keys = [
                "macro_f1",
                "micro_acc",
            ]
            return {
                k: logs.get(f"{prefix}{k}")
                for k in metric_keys
                if f"{prefix}{k}" in logs
            }

        train_metrics = extract_metrics("")
        val_metrics = extract_metrics("val_")

        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    str(train_metrics),
                    str(val_metrics),
                ]
            )


class TorchLogsPostprocessor(pl.callbacks.Callback):
    """
    Callback to postprocess PyTorch Lightning CSVLogger output into:
      - model_loss.csv
      - model_metrics.csv
    and remove the default model_metrics/version_0/ directory.
    """

    def __init__(self, logger_dir: str):
        """
        Parameters:
            logger_dir (str): Path to the CSVLogger version_0 directory.
        """
        super().__init__()
        self.logger_dir = logger_dir

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics_path = os.path.join(self.logger_dir, "version_0", "metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"[CSVPostprocessor] metrics.csv not found at {metrics_path}")
            return

        metrics_df = pd.read_csv(metrics_path)
        metrics_df = metrics_df.groupby("epoch").first().reset_index()

        # write model_loss.csv
        loss_df = metrics_df[["train_loss", "val_loss"]].rename(
            columns={"train_loss": "train", "val_loss": "val"}
        )
        loss_df.to_csv(
            os.path.join(os.path.dirname(self.logger_dir), "model_loss.csv"),
            index=False,
        )

        # write model_metrics.csv
        def collect_metrics(row, split):
            return str(
                {
                    "macro_f1": row.get(f"{split}_macro_f1"),
                    "micro_acc": row.get(f"{split}_micro_acc"),
                }
            )

        metrics_df["train"] = metrics_df.apply(
            lambda row: collect_metrics(row, "train"), axis=1
        )
        metrics_df["val"] = metrics_df.apply(
            lambda row: collect_metrics(row, "val"), axis=1
        )
        metrics_str_df = metrics_df[["train", "val"]]
        metrics_str_df.to_csv(
            os.path.join(os.path.dirname(self.logger_dir), "model_metrics.csv"),
            index=False,
        )

        # cleanup
        if os.path.isdir(self.logger_dir):
            shutil.rmtree(self.logger_dir)
