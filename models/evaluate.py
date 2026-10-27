import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import seaborn as sns
import sys
import torch
import torchmetrics
from scipy import stats
from torchmetrics.classification.classification_report import ClassificationReport


class MetricCollection:
    def __init__(self, mode="multiclass", stage="train", num_classes=2, **kwargs):
        """
        Collection of evaluation metrics for semantic segmentation tasks.

        Args:
            mode (str): One of "binary", "multiclass", "multilabel".
            stage (str): Stage of training. One of "train", "eval".
                During training, only computationally effficient metrics are computed.
                During evaluation, all metrics are computed (incl. e.g. confusion matrices).
            num_classes (int): Number of target classes (ignored if binary).
            **kwargs: Additional keyword arguments passed to TorchMetrics initializers.
        """
        self.mode = mode
        self.stage = stage
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_groups = {
            # RAM-intense metrics
            "heavy": [
                "ece",
                "mce",
                "prc_curve",
                "roc_curve",
            ],
            # take probs as inputs not preds
            "probs": [
                "auprc",
                "auroc",
                "prc_curve",
                "roc_curve",
                "ece",
                "mce",
                "report",
            ],
            # only to be computed at the end of training
            "final": [
                "auprc",
                "auroc",
                "prc_curve",
                "roc_curve",
                "ece",
                "mce",
                "conf_matrix",
                "report",
            ],
        }
        self._init_metrics(**kwargs)

    def _init_metrics(self, **kwargs):
        """
        Initialize TorchMetrics objects depending on mode.

        Note: In case of rare classes, averages for metrics such as F1, classification report, etc
        are usually computed using the set of predicted and true labels. Therefore we can initialise
        the metrics with the higher number of classes (num_classes), and metrics' averages will
        still only be computed for the relevant classes.
        """
        self.num_samples = 0

        # extend kwargs
        if "task" not in kwargs:
            kwargs["task"] = self.mode
        if "num_classes" not in kwargs and self.mode != "binary":
            kwargs["num_classes"] = self.num_classes

        self.metrics = {
            "micro_f1": torchmetrics.F1Score(average="micro", **kwargs),
            "macro_f1": torchmetrics.F1Score(average="macro", **kwargs),
            "micro_precision": torchmetrics.Precision(average="micro", **kwargs),
            "macro_precision": torchmetrics.Precision(average="macro", **kwargs),
            "micro_recall": torchmetrics.Recall(average="micro", **kwargs),
            "macro_recall": torchmetrics.Recall(average="macro", **kwargs),
            "micro_iou": torchmetrics.JaccardIndex(average="micro", **kwargs),
            "macro_iou": torchmetrics.JaccardIndex(average="macro", **kwargs),
            "micro_acc": torchmetrics.Accuracy(average="micro", **kwargs),
            "macro_acc": torchmetrics.Accuracy(average="macro", **kwargs),
            "auprc": torchmetrics.AveragePrecision(thresholds=100, **kwargs),
            "auroc": torchmetrics.AUROC(thresholds=100, **kwargs),
            "prc_curve": torchmetrics.PrecisionRecallCurve(thresholds=100, **kwargs),
            "roc_curve": torchmetrics.ROC(thresholds=100, **kwargs),
            "ece": torchmetrics.CalibrationError(norm="l1", **kwargs),
            "mce": torchmetrics.CalibrationError(norm="max", **kwargs),
            "conf_matrix": torchmetrics.ConfusionMatrix(**kwargs),
            "report": ClassificationReport(output_dict=True, digits=4, **kwargs),
        }

        for metric in self.metrics.values():
            metric.to(self.device)

    def clone(self):
        """
        Return a deep-copied MetricCollection with the same configuration
        and metric definitions, but with all metric states reset.
        """
        cloned = copy.deepcopy(self)
        cloned.reset()
        return cloned

    def set_stage(self, stage):
        """
        Set the stage for the metrics collection.

        Args:
            stage (str): Stage of training. One of "train", "eval".
        """
        if stage not in {"train", "eval"}:
            raise ValueError(f"Unsupported stage: {stage}")
        self.stage = stage

    def reset(self):
        """Reset all metric states."""
        for metric in self.metrics.values():
            metric.reset()
        self.results = {}
        self.num_samples = 0

    def update(self, preds, prob_preds, targets, sample_ratio=0.01):
        """Update metrics for a batch.

        `heavy` and `probs` are independent axes:
        - probs: metric consumes probabilities (N, C, H, W) vs class indices (N, H, W)
        - heavy: metric is subsampled (and, for prob metrics, flattened to (N_pix, C))
        The four combinations:
        heavy & probs      -> flattened, subsampled probs       (ece, mce, curves)
        heavy & not probs  -> flattened, subsampled preds
        not heavy & probs  -> full probs, no sampling           (report, auprc, auroc)
        not heavy & not    -> full preds, no sampling           (f1, iou, acc, conf_matrix)

        Args:
            preds (torch.Tensor): Predicted class indices (N, H, W)
            prob_preds (torch.Tensor): Predicted probabilities (N, C, H, W)
            targets (torch.Tensor): Ground truth class indices (N, H, W)
            sample_ratio (float): Fraction of pixels kept for heavy (subsampled) metrics.
        """
        self.num_samples += targets.shape[0]
        preds = preds.to(self.device)
        prob_preds = prob_preds.to(self.device)
        targets = targets.to(self.device)

        def _active(name):
            return not (self.stage == "train" and name in self.metrics_groups["final"])

        # build the flattened, subsampled views if some active heavy metric needs them
        prob_flat = pred_flat = target_flat = None
        if any(
            name in self.metrics_groups["heavy"] and _active(name)
            for name in self.metrics
        ):
            C = prob_preds.shape[1]
            prob_flat = (
                prob_preds.reshape(prob_preds.shape[0], C, -1)
                .permute(0, 2, 1)
                .reshape(-1, C)
                .squeeze(-1)
            )
            pred_flat = preds.reshape(-1)
            target_flat = targets.reshape(-1)
            if sample_ratio < 1:
                step = max(int(1 / sample_ratio), 1)
                prob_flat = prob_flat[::step]
                pred_flat = pred_flat[::step]
                target_flat = target_flat[::step]

        for name, metric in self.metrics.items():
            if not _active(name):
                continue
            is_probs = name in self.metrics_groups["probs"]
            is_heavy = name in self.metrics_groups["heavy"]

            if is_heavy and is_probs:
                metric.update(prob_flat, target_flat)
            elif is_heavy:
                metric.update(pred_flat, target_flat)
            elif is_probs:
                metric.update(prob_preds, targets)
            else:
                metric.update(preds, targets)

    def compute(self, return_dict=True):
        """
        Compute all metrics at the end of an epoch.

        Args:
            return_dict (bool): If True, return results as a dictionary.
        """
        self.results = {}
        self.results["num_samples"] = int(self.num_samples)
        for name, metric in self.metrics.items():
            if self.stage == "train" and name in self.metrics_groups["final"]:
                continue
            if name in ["prc_curve", "roc_curve"]:
                continue  # will be done at plot time
            result = metric.compute()
            if isinstance(result, dict):
                self.results[name] = result
            elif isinstance(result, torch.Tensor) and result.ndim > 0:
                self.results[name] = result.detach().cpu().numpy()
            else:
                self.results[name] = float(result.detach().cpu())
        if return_dict:
            return self.results

    def plot_report(self, out_dir=None, class_names=None):
        """
        Display and optionally save classification report.

        Args:
            out_dir (str or None, optional): If provided, directory where files will be saved.
            class_names (list, optional): Class labels for axes.
        """
        report = self.results["report"]
        report = pd.DataFrame(report).transpose().round(4)
        report["support"] = report["support"].astype(int)
        if class_names is not None:
            idx = report.index.values
            idx[: len(class_names)] = class_names
            report.set_index(pd.Index(idx), inplace=True)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            report.to_csv(os.path.join(out_dir, "report.csv"))
        else:
            print("Classification report:")
            print(report)

    def plot_confusion_matrix(self, out_dir=None, class_names=None, log_scaled=True):
        """
        Display and optionally save confusion matrix (absolute + relative log-scaled).

        Args:
            out_dir (str or None, optional): If provided, directory where files will be saved.
            class_names (list, optional): Class labels for axes.
            log_scaled (bool, optional): Whether to apply log scaling to the relative figure heatmap.
        """

        # convert cm to pandas df & save to CSV if requested
        cm = self.results["conf_matrix"]
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        else:
            class_names = [name[:20] for name in class_names]
        cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # format numbers as thousands with 'K'
        def format_k(x):
            return f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"

        sns.heatmap(
            cm,
            annot=np.vectorize(format_k)(cm),
            fmt="",
            cmap="viridis",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            annot_kws={"size": 8},
        )
        axes[0].set_title("absolute figures", fontsize=12)
        axes[0].set_xlabel("predicted labels", fontsize=12)
        axes[0].set_ylabel("true labels", fontsize=12)

        cm_prop = (cm + sys.float_info.epsilon) / cm.sum()
        norm = mcolors.LogNorm(vmin=1e-4, vmax=1) if log_scaled else None

        sns.heatmap(
            cm_prop,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            annot_kws={"size": 8},
            norm=norm,
        )
        axes[1].set_title("relative figures", fontsize=12)
        axes[1].set_xlabel("predicted labels", fontsize=12)
        axes[1].set_ylabel("true labels", fontsize=12)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
            plt.close(fig)
        else:
            print("Confusion Matrix DataFrame:")
            plt.show()


class CovariateEvaluator:
    """
    Covariate-stratified evaluation of habitat predictions.

    Besides the habitat code, each label tile carries per-pixel covariates,
    all of them only defined inside mapped habitats, which are returned by
    the ``S2_Dataset`` when ``read_covars=True``:

        * ``polygon_id``       id of the habitat polygon the pixel belongs to
        * ``lrt_status``       conservation status (A / B / C)
        * ``acquisition_date`` offset of the field survey from the imagery, in years


    Results land in ``self.results`` after :meth:`compute`:

        ``by_status``        metrics per status (A / B / C / nan) plus an "all" row
        ``by_status_class``  the same strata, broken down per class
        ``by_shift_bin``     metrics per acquisition-shift bin
        ``by_shift_corr``    correctness vs. shift, at pixel and at polygon level
        ``by_polygon``       per polygon: raw TP/FP counts and metrics

    Usage online (during model training):

        a) batch-wise: call :meth:`update` per batch, then :meth:`compute`
           and :meth:`save`.

        b) from predictions that are already available

        CovariateEvaluator.from_arrays(
            preds_raw, trainer.ds, "val",
            out_dir=os.path.join(trainer.save_dir, "val", "covariates"),
            metrics=trainer.metric_collection,
        )

    Usage post-hoc (after model training) via class methods. These rebuild the
    trainer from the run's config.json, restore the best checkpoint and run
    inference on the requested split:

        [01 - deep patchwise learning]
        import os
        from models.evaluate import CovariateEvaluator
        RUN = "run_path"
        ev = CovariateEvaluator.from_patchwise(
            config=os.path.join(RUN, "config.json"),
            split="test",
            model_params=os.path.join(RUN, "model_params.pt"),  # optional, default
            out_dir=os.path.join(RUN, "test", "covariates"),     # optional, default
        )

        [02 - pixelwise learning, deep and machine learning models]
        import os
        from models.evaluate import CovariateEvaluator
        RUN = "run_path"
        ev = CovariateEvaluator.from_pixelwise(
            config=os.path.join(RUN, "config.json"),
            split="test",
            out_dir=os.path.join(RUN, "test", "covariates"),     # optional, default
        )
        # the trainer type (deep vs. machine learning) is inferred from the config;
        # model_params defaults to the checkpoint the trainer resolves itself

        [03 - pixelwise, with predictions computed separately]
        import os
        from models.pixelwise.deep_learning.train import Trainer
        from models.evaluate import CovariateEvaluator
        RUN = "run_path"
        trainer = Trainer.from_config(os.path.join(RUN, "config.json"), wandb_project=None)
        trainer.prep_model()
        clf = trainer._load_clf()
        preds = clf.predict(trainer.ds.X_test)
        ev = CovariateEvaluator.from_arrays(
            preds, trainer.ds, "test",
            out_dir=os.path.join(RUN, "test", "covariates"),
            metrics=trainer.metric_collection,
        )
    """

    MISSING = "nan"

    def __init__(
        self,
        metrics=None,
        status_labels={1: "A", 2: "B", 3: "C"},
        shift_bin=1.0,
        ignore_index=-99,
    ):
        """
        Args:
            metrics (MetricCollection): Collection to take the metric definitions (mode,
                number of classes, ignore index) from; it is cloned, not modified. Required.
            status_labels (dict, optional): Status code -> label.
            shift_bin (float, optional): Bin width in years for the shift. Default 1.0.
            ignore_index (int, optional): Target/prediction value to drop. Default -99.
        """
        if metrics is None:
            raise TypeError(
                "CovariateEvaluator requires `metrics` (a MetricCollection)"
            )
        self._mc = metrics.clone()
        self._mc.set_stage("train")
        self.status_labels = status_labels
        self.shift_bin = float(shift_bin)
        self.ignore_index = int(ignore_index)
        self.reset()

    # ------------------------------------------------------------------ #
    # computation
    # ------------------------------------------------------------------ #

    def reset(self):
        """Drop all accumulated pixels."""
        self._buf = []
        self.results = {}

    def update(self, preds, targets, status=None, shift=None, polygon=None):
        """
        Accumulate one batch. Shapes are irrelevant, everything is flattened.

        Args:
            preds (array-like): Predicted class indices.
            targets (array-like): Ground truth class indices.
            status (array-like, optional): Status codes, NaN where undefined.
            shift (array-like, optional): Shift in years, NaN where undefined.
            polygon (array-like, optional): Polygon id per pixel, NaN outside mapped habitats.
        """
        targets, preds = self._flat(targets), self._flat(preds)
        n = targets.size
        keep = (targets != self.ignore_index) & (preds != self.ignore_index)
        if not keep.any():
            return
        cols = {
            "target": targets[keep].astype(np.int16),
            "pred": preds[keep].astype(np.int16),
        }
        for name, arr in (("status", status), ("shift", shift), ("polygon", polygon)):
            arr = self._flat(arr, n)
            if arr is not None:
                cols[name] = arr[keep]
        self._buf.append(pd.DataFrame(cols))

    def compute(self):
        """Aggregate the accumulated pixels into the result tables."""
        if not self._buf:
            raise RuntimeError("no valid pixels accumulated")
        df = pd.concat(self._buf, ignore_index=True)
        self._buf = [df]  # keep it concatenated for repeated compute() calls
        if "status" not in df:
            df["status"] = np.nan
        df["correct"] = (df["target"] == df["pred"]).astype(np.int8)
        if "shift" in df:
            df["shift"] = df["shift"].abs()

        # ---- per conservation status, overall and per class ----------- #
        groups = [
            (self._status_label(code), sub)
            for code, sub in df.groupby("status", dropna=False)
        ] + [("all", df)]
        rows, per_class = zip(*(self._metrics(sub, lab) for lab, sub in groups))

        by_status = pd.DataFrame(rows).set_index("stratum")
        order = list(self.status_labels.values()) + [self.MISSING, "all"]
        by_status = by_status.loc[[s for s in order if s in by_status.index]]
        by_status["share_of_pixels"] = by_status["n_pixels"] / len(df)

        by_status_class = pd.concat(per_class, ignore_index=True)
        by_status_class["stratum"] = pd.Categorical(
            by_status_class["stratum"], by_status.index, ordered=True
        )
        by_status_class = by_status_class.sort_values(["stratum", "class"]).reset_index(
            drop=True
        )

        # ---- shift bins & correlations ----------------------- #
        by_shift = pd.DataFrame()
        corr = []
        sh = df.dropna(subset=["shift"]) if "shift" in df else df.iloc[:0]
        if len(sh):
            sh = sh.assign(bin=(sh["shift"] / self.shift_bin).round() * self.shift_bin)
            by_shift = [self._metrics(g, b)[0] for b, g in sh.groupby("bin")]
            by_shift = pd.DataFrame(by_shift).rename(columns={"stratum": "shift_years"})
            corr.append(
                self._corr(sh["shift"].to_numpy(), sh["correct"].to_numpy(), "pixel")
            )

        # ---- polygons ----------------------- #
        by_polygon = pd.DataFrame()
        poly = df.dropna(subset=["polygon"]) if "polygon" in df else df.iloc[:0]
        if len(poly):
            by_polygon = self._polygon_tables(poly)
            # one observation per polygon: the survey date is a polygon property, so
            # this is the level at which the shift actually varies
            pol = (
                by_polygon.dropna(subset=["mean_shift_years"])
                if ("mean_shift_years" in by_polygon)
                else by_polygon.iloc[:0]
            )
            if len(pol):
                px, py = pol["mean_shift_years"], pol["accuracy"]
                corr.append(self._corr(px, py, "polygon"))
                corr.append(self._corr(px, py, "polygon_weighted", w=pol["n_pixels"]))

        self.results = {
            "by_status": by_status,
            "by_status_class": by_status_class,
            "by_shift_bin": by_shift,
            "by_shift_corr": pd.DataFrame(corr),
            "by_polygon": by_polygon,
        }
        return self.results

    def save(self, out_dir):
        """Write the tables as CSV and the figure as PNG."""
        os.makedirs(out_dir, exist_ok=True)
        for name, df in self.results.items():
            if len(df):
                df.to_csv(
                    os.path.join(out_dir, f"{name}.csv"), index=(name == "by_status")
                )
        self.plot(out_dir=out_dir)
        return out_dir

    def _metrics(self, df, stratum):
        """
        Metrics calculation: aggregated & per-class breakdown.
        The aggregates come from MetricCollection at stage "train".
        The per-class (one-vs-rest) states are derived from the
        collection's confusion matrix.

        Args:
            df (DataFrame): pixel-level data for one stratum.
            stratum (str): label of the stratum.

        Returns:
            (dict, DataFrame): aggregate row, per-class rows.
        """
        dev = self._mc.device
        preds = torch.as_tensor(df["pred"].to_numpy().astype(np.int64)).to(dev)
        targets = torch.as_tensor(df["target"].to_numpy().astype(np.int64)).to(dev)

        # calculate main metrics
        self._mc.reset()
        self._mc.update(preds, preds, targets)
        res = self._mc.compute()
        row = {"stratum": stratum, "n_pixels": len(df)}
        row.update({k: v for k, v in res.items() if isinstance(v, float)})
        if "shift" in df:
            row["mean_shift_years"] = float(df["shift"].mean())

        # calculate conf_matrix (part of "final" group -> not calculated at train)
        cm_metric = self._mc.metrics["conf_matrix"]
        cm_metric.update(preds, targets)
        cm = cm_metric.compute().detach().cpu().numpy().astype(np.int64)
        cm_metric.reset()

        # fetch per-class metrics from the confusion matrix, one-vs-rest
        tp = np.diag(cm)
        support, predicted, total = cm.sum(1), cm.sum(0), cm.sum()
        fp, fn = predicted - tp, support - tp
        precision, recall = self._div(tp, predicted), self._div(tp, support)
        per_class = pd.DataFrame(
            {
                "stratum": stratum,
                "class": np.arange(len(tp)),
                "support": support,
                "n_predicted": predicted,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": total - tp - fp - fn,
                "precision": precision,
                "recall": recall,
                "f1": self._div(2 * precision * recall, precision + recall),
                "iou": self._div(tp, tp + fp + fn),
            }
        )

        return row, per_class[(support > 0) | (predicted > 0)]

    def _polygon_tables(self, poly):
        """
        Per-polygon raw counts
        """
        # group by polygon, target, pred to get counts of each class in each polygon
        cls_counts = (
            poly.groupby(["polygon", "target", "pred"], sort=False)
            .size()
            .reset_index(name="n")
        )
        by_target = cls_counts.groupby(["polygon", "target"], as_index=False)["n"].sum()
        main = by_target.sort_values("n").drop_duplicates("polygon", keep="last")
        d = cls_counts.merge(
            main[["polygon", "target"]].rename(columns={"target": "cls"}), on="polygon"
        )
        is_t, is_p = d["target"] == d["cls"], d["pred"] == d["cls"]
        for name, mask in (("TP", is_t & is_p), ("FN", is_t & ~is_p)):
            d[name] = d["n"].where(mask, 0)
        d["n_correct"] = d["n"].where(d["target"] == d["pred"], 0)
        by_polygon = d.groupby("polygon", as_index=False).agg(
            target=("cls", "first"),
            n_pixels=("n", "sum"),
            n_correct=("n_correct", "sum"),
            TP=("TP", "sum"),
            FN=("FN", "sum"),
        )
        by_polygon["n_target_classes"] = by_polygon["polygon"].map(
            by_target.groupby("polygon").size()
        )
        by_polygon["accuracy"] = by_polygon["n_correct"] / by_polygon["n_pixels"]
        extra = {c: (c, "first") for c in ("status",) if c in poly}
        if "shift" in poly:
            extra["mean_shift_years"] = ("shift", "mean")
        if extra:
            by_polygon = by_polygon.merge(
                poly.groupby("polygon", as_index=False).agg(**extra), on="polygon"
            )
        return by_polygon

    def _status_label(self, code):
        """Semantic label of a status code, or "nan" where the covariate is missing."""
        if pd.isna(code):
            return self.MISSING
        return self.status_labels.get(int(round(code)), f"code_{code:g}")

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #

    def plot(self, out_dir=None):
        """Status bars, and accuracy vs. shift with one point per shift bin."""
        st, sh = self.results["by_status"], self.results["by_shift_bin"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        metrics = ["micro_acc", "macro_f1"]
        st[metrics].plot.bar(ax=axes[0], width=0.8, rot=0, legend=False)
        for xi, share in enumerate(st["share_of_pixels"]):
            axes[0].annotate(f"{share:.1%}", (xi, 1.0), ha="center", fontsize=8)
        axes[0].set(xlabel="conservation status", ylabel="metric value", ylim=(0, 1.15))
        axes[0].set_title("stratified by conservation status", fontsize=12)
        axes[0].legend(frameon=False, ncol=len(metrics), loc="upper left")

        fit, covered = "no fit", np.nan
        if len(sh):
            axes[1].scatter(
                sh["mean_shift_years"],
                sh["micro_acc"],
                alpha=0.6,
                color="#8c510a",
                s=sh["n_pixels"] / sh["n_pixels"].max() * 100,
            )
            covered = sh["n_pixels"].sum() / st.loc["all", "n_pixels"]
            if np.ptp(sh["mean_shift_years"]) > 0:
                b, a = np.polyfit(
                    sh["mean_shift_years"], sh["micro_acc"], 1, w=sh["n_pixels"]
                )
                xs = np.array(
                    [sh["mean_shift_years"].min(), sh["mean_shift_years"].max()]
                )
                axes[1].plot(xs, a + b * xs, color="#762a83")
                fit = f"intercept = {a:.3f}, slope = {b:.3f} / year"
            axes[1].set_ylim(0, 1.05)
        axes[1].set(
            xlabel="shift to acquisition date [Δ years]",
            ylabel="accuracy",
        )
        # spell out the subset, so this panel is not read against the "all" bar
        axes[1].set_title(
            f"{self._corr_title()}\n{fit}\n"
            f"marker = shift bin; mapped habitat only ({covered:.1%} of pixels)",
            fontsize=9,
        )

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, "eval_covariates.png"), dpi=150)
            plt.close(fig)
        else:
            plt.show()

        # plot status by class
        df = self.results["by_status_class"]
        x_order = (
            df[df["stratum"] == "all"][["class", "support"]]
            .sort_values("support", ascending=False)["class"]
            .to_list()
        )
        g = sns.catplot(
            data=df[df["stratum"] != "all"],
            x="class",
            order=x_order,
            y="f1",
            hue="stratum",
            hue_order=list(df[df["stratum"] != "all"]["stratum"].unique()),
            kind="bar",
            height=5,
            aspect=2,
            palette="magma_r",
        )

        if out_dir is not None:
            g.savefig(os.path.join(out_dir, "eval_covariates_by_class.png"), dpi=150)
            plt.close(g.fig)
        else:
            plt.show()

    def _corr_title(self):
        corr = self.results["by_shift_corr"]
        if not len(corr):
            return "no shift covariate"
        return "\n".join(
            f"{row['level']}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.2g}, "
            f"n={row['n']:,}"
            for _, row in corr.iterrows()
            if row["level"] in ("pixel", "polygon")
        )

    # ------------------------------------------------------------------ #
    # inference drivers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_arrays(cls, preds, ds, split, out_dir, **kwargs):
        ev = cls(**kwargs)
        ev.update(
            preds=preds,
            targets=getattr(ds, f"y_{split}"),
            status=ds.get_covar("lrt_status", split),
            shift=ds.get_covar("acquisition_date", split),
            polygon=ds.get_covar("polygon_id", split),
        )
        ev.compute()
        ev.save(out_dir)
        return ev

    @classmethod
    def from_patchwise(
        cls, config, split="test", model_params=None, out_dir=None, **kwargs
    ):
        """
        Build and run an evaluator for a patchwise run, from its config file.

        Rebuilds the Trainer, restores the checkpoint and runs inference over the
        split with the covariates switched on.

        Args:
            config (str): Path to the run's config.json.
            split (str, optional): Split to evaluate. Default "test".
            model_params (str, optional): Checkpoint to restore. Defaults to
                "model_params.pt" in the run directory.
            out_dir (str, optional): Where tables & figures are written. Defaults to
                <save_dir>/<split>/covariates.
            **kwargs: Passed to the constructor (`shift_bin`, `status_labels`, ...).

        Returns:
            CovariateEvaluator: with `results` computed and written to `out_dir`.
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from models.patchwise.train import Trainer

        trainer = Trainer.from_config(config, wandb_project=None)
        trainer.prep_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = trainer.model.model.to(device)
        model.load_state_dict(
            torch.load(
                model_params or os.path.join(trainer.save_dir, "model_params.pt"),
                weights_only=True,
                map_location=device,
            )
        )
        model.eval()

        # reuse the split's dataset (and its disk cache), only ask it for the covariates
        ds = trainer.dataloaders[split].dataset
        ds.read_covars, ds.transform = True, None
        loader = DataLoader(
            ds,
            batch_size=trainer.batch_size,
            num_workers=(
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1
            ),
            shuffle=False,
        )
        idx_of = {c: i for i, c in enumerate(ds.COVAR_ORDER)}

        kwargs.setdefault("metrics", trainer.metric_collection)
        ev = cls(**kwargs)
        with torch.no_grad():
            for inputs, labels, covars, _ in tqdm(
                loader, desc=f"Covariate eval on {len(ds)} patches"
            ):
                logits = model(inputs.to(device, dtype=torch.float))
                preds = (
                    (logits.squeeze(1).sigmoid() > 0.5).long()
                    if trainer.num_classes == 2
                    else logits.softmax(dim=1).argmax(dim=1)
                )
                ev.update(
                    preds=preds.cpu(),
                    targets=labels,
                    **{
                        key: covars[:, idx_of[covar]]
                        for key, covar in (
                            ("status", "lrt_status"),
                            ("shift", "acquisition_date"),
                            ("polygon", "polygon_id"),
                        )
                        if covar in idx_of
                    },
                )
        ev.compute()
        ev.save(out_dir or os.path.join(trainer.save_dir, split, "covariates"))
        return ev

    @classmethod
    def from_pixelwise(
        cls, config, split="test", model_params=None, out_dir=None, **kwargs
    ):
        """
        Build and run an evaluator for a pixelwise run, from its config file.

        Serves both pixelwise trainers: the one to rebuild is taken from the config
        itself, and predictions are routed through the trainer's own `_predict` where
        it exists (the machine learning branch, whose estimator needs `_prep_X` and
        `_decode` around it) and through the estimator directly otherwise.

        Args:
            config (str): Path to the run's config.json.
            split (str, optional): Split to evaluate. Default "test".
            model_params (str, optional): Checkpoint to restore. Defaults to the one
                the trainer resolves by itself.
            out_dir (str, optional): Where tables & figures are written. Defaults to
                <save_dir>/<split>/covariates.
            **kwargs: Passed to the constructor (`shift_bin`, `status_labels`, ...).

        Returns:
            CovariateEvaluator: with `results` computed and written to `out_dir`.
        """
        import importlib
        import json
        from models.pixelwise.machine_learning.model import MLModelConfigurator

        # init trainer & fetch predictions, targets and covariates
        with open(config) as f:
            cfg = json.load(f)
        kind = (
            "machine_learning"
            if cfg.get("model_name") in MLModelConfigurator.MODELS.keys()
            else "deep_learning"
        )
        Trainer = importlib.import_module(f"models.pixelwise.{kind}.train").Trainer
        trainer = Trainer.from_config(config, wandb_project=None)
        trainer.prep_model()
        clf = trainer._load_clf(model_params)
        X = getattr(trainer.ds, f"X_{split}")
        preds = trainer._predict(X) if hasattr(trainer, "_predict") else clf.predict(X)

        # evaluate against the covariates
        kwargs.setdefault("metrics", trainer.metric_collection)
        return cls.from_arrays(
            preds,
            trainer.ds,
            split,
            out_dir or os.path.join(trainer.save_dir, split, "covariates"),
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # helper functions for weighted correlation and ranks, and p-value of a correlation
    # ------------------------------------------------------------------ #

    def _corr(self, x, y, level, w=None):
        """
        Pearson & Spearman of two 1-D arrays, NaN-safe for constant input.

        Args:
            x, y (array-like): The two variables.
            level (str): Label copied into the result row.
            w (array-like, optional): Weight per observation, for rows that already
                summarise several pixels - a polygon row stands for its pixel count, so
                weighting stops a 20-pixel polygon counting as much as a 20,000-pixel
                one. Pixel rows are one sample each and need no weights.

        Returns:
            dict: level, n, n_eff (Kish effective sample size, == n when unweighted)
            and the two coefficients with their p-values.
        """
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            ok = np.isfinite(w) & (w > 0)
            x, y, w = x[ok], y[ok], w[ok]
        n = len(x)
        n_eff = float(w.sum() ** 2 / (w**2).sum()) if w is not None and n else float(n)
        row = {"level": level, "n": n, "n_eff": n_eff}
        if n > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            if w is None:
                r, rp = stats.pearsonr(x, y)
                rho, sp = stats.spearmanr(x, y)
            else:
                r = self._wcorr(x, y, w)
                rho = self._wcorr(self._wranks(x, w), self._wranks(y, w), w)
                # p-values use n_eff, so a few huge polygons cannot fake significance
                rp, sp = (self._pvalue(v, n_eff) for v in (r, rho))
        else:
            r = rp = rho = sp = np.nan
        row.update(pearson_r=r, pearson_p=rp, spearman_rho=rho, spearman_p=sp)
        return row

    def _div(self, a, b):
        """Division with Zeros rather than NaN for undefined ratios, matching torchmetrics,
        so that column means equal the macro average."""
        return np.divide(a, b, out=np.zeros(len(a), dtype=float), where=b > 0)

    def _flat(self, x, n=None):
        """Flatten to 1-D numpy, repeating per-sample values to per-pixel if needed."""
        if x is None:
            return None
        if hasattr(x, "detach"):  # torch tensor, without importing torch
            x = x.detach().cpu().numpy()
        elif hasattr(x, "numpy"):  # tf / keras tensor
            x = x.numpy()
        x = np.asarray(x).reshape(-1)
        if n is not None and x.size != n:
            if not x.size or n % x.size:
                raise ValueError(f"cannot broadcast {x.size} values to {n} pixels")
            x = np.repeat(x, n // x.size)
        return x

    def _pvalue(self, r, n):
        """Two-sided p-value of a correlation coefficient, via the t transform."""
        if not np.isfinite(r) or n < 3 or abs(r) >= 1:
            return np.nan
        t = r * np.sqrt((n - 2) / (1 - r**2))
        return float(2 * stats.t.sf(abs(t), n - 2))

    def _wranks(self, v, w):
        """Tie-corrected mid-ranks of a weighted multiset; == scipy ranks when w == 1."""
        uniq, inv = np.unique(v, return_inverse=True)
        totals = np.zeros(uniq.size)
        np.add.at(totals, inv, w)
        return (np.concatenate([[0.0], np.cumsum(totals)])[:-1] + (totals + 1.0) / 2.0)[
            inv
        ]

    def _wcorr(self, x, y, w):
        """Pearson correlation over weighted observations."""
        c = np.cov(np.vstack([x, y]), aweights=w, ddof=0)
        denom = np.sqrt(c[0, 0] * c[1, 1])
        return float(c[0, 1] / denom) if denom > 0 else np.nan
