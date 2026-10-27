import albumentations as A
import inspect
import json
import numpy as np
import pandas as pd
import os
import random
import time
import torch
import torch.optim as optim
import wandb
import warnings

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.patchwise.crossval import SplitGenerator
from models.patchwise.data import S2_Dataset
from models.patchwise.model import SegmentationModel
from models.patchwise.loss import SegmentationLoss
from models.patchwise.utils import EarlyStopping, CosineScheduler
from models.evaluate import MetricCollection, CovariateEvaluator
from models.paths import data_dir_of, resolve_paths


class Trainer:
    """Semantic segmentation model training & evaluation on a data set of choice"""

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
        # -- model training parameters (see model.py & loss.py for details) --
        model_name: str = "Unet",
        encoder_name: str = "resnet18",
        encoder_weights: str = None,
        train_mode: str = "finetune",
        train_augmentations: bool = True,
        batch_size: int = 8,
        optimizer: str = "Adam",
        loss: str = "CrossEntropyLoss",
        loss_params: dict = None,
        lr: float = 1e-4,
        max_epochs: int = 20,
        patience: int = 7,
        seed: int = 42,
        # -- logging / output --
        save_dir: str = None,
        wandb_project: str = None,
        wandb_runname: str = None,
        wandb_tags: list = None,
        wandb_notes: str = None,
        verbose: bool = True,
    ):
        """
        Define model training & logging parameters.

        -- data loading parameters (see data.py & crossval.py for details) --
        meta_file (str): Path to CSV file containing metadata for all input tiles
            and their associated labels. The CSV is expected to contain:
            - "path_input" (str): path to the input raster tile (Sentinel-2)
            - "path_label" (str): path to the corresponding label tile (habitat map)
            Relative paths are interpreted relative to the CSV's folder.
        sel_time (list, optional): Time steps to select from the dataset.
        sel_band (list, optional): Bands to select from the dataset.
        sel_task (str, optional): Task type for the dataset. Default "lvl1_bic".
        split_strategy (str, optional): Train/val/test split strategy. Default "random".
        split_subset (float, optional): Fraction of the total dataset to use. Default 1.0.
        split_trainsize (float, optional): Fraction of the dataset used for training. Default 0.7.
        split_fold (int, optional): Ignored, kept for backward compatibility. Splits
            differ by `seed` only. Default 0.
        use_cache (str, optional): Caching mode for the dataset. Default "disk".
        split_kwargs (dict, optional): Additional keyword arguments for the split strategy. Default None.

        -- model training parameters --
        model_name (str, optional): Model architecture (see model.py). Default "Unet".
        encoder_name (str, optional): Encoder architecture (see model.py). Not all encoders
            are compatible with all architectures. Default "resnet18".
        encoder_weights (str, optional): "imagenet" for pretrained weights or None for no pretraining.
        train_mode (str, optional): "scratch", "finetune" (default), or "feat_extract".
        train_augmentations (bool, optional): Whether to apply data augmentations during training. Default True.
        batch_size (int, optional): Batch size. Default 8.
        optimizer (str, optional): "SGD" or "Adam" (default).
        loss (str, optional): Loss function (see loss.py). Default "CrossEntropyLoss".
        loss_params (dict, optional): Extra loss params, e.g. gamma=2 for FocalLoss. Default {}.
        lr (float, optional): Learning rate. Default 1e-4.
        max_epochs (int, optional): Max training epochs. Default 20.
        patience (int, optional): Early-stopping patience on validation. Default 7.
        seed (int, optional): Seed for reproducibility. Default 42.

        -- logging / output --
        save_dir (str): Directory to save results for the run.
        wandb_project (str): wandb project name.
        wandb_runname (str): wandb run name.
        wandb_tags (list, optional): wandb tags. Default [].
        wandb_notes (str, optional): Additional notes/description for the run.
        verbose (bool, optional): Whether to print progress messages. Default True.
        """
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
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.train_mode = train_mode
        self.train_augmentations = train_augmentations
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.loss_params = loss_params if loss_params is not None else {}
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.wandb_runname = wandb_runname
        self.wandb_tags = wandb_tags if wandb_tags is not None else []
        self.wandb_notes = wandb_notes
        self.verbose = verbose
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
        # define wandb session & log model configuration
        if self.wandb_project is not None:
            wandb.init(
                project=self.wandb_project,
                dir=self.save_dir,
                tags=self.wandb_tags,
                notes=self.wandb_notes,
                settings=wandb.Settings(
                    mode="online",
                    console="off",
                    x_stats_sampling_interval=60.0,
                    x_file_stream_transmit_interval=60.0,
                ),
            )
            wandb.run.name = self.wandb_runname
        # save & update config
        self.save_config()
        self._update_logs()

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
        Initialize model and dataloaders using CSV file
        """
        # prepare metafile by creating splits
        split_generator = SplitGenerator(
            meta_file=self.meta_file,
            strategy=self.split_strategy,
            trainsize=self.split_trainsize,
            fold=self.split_fold,
            subset=self.split_subset,
            seed=self.seed,
            **self.split_kwargs,
        )
        split_meta_df = split_generator.generate()

        if self.verbose:
            print("Generating train/val/test splits")
            print(f"- split strategy '{self.split_strategy}'")
            print(f"- train size {self.split_trainsize}")
            print(f"- fold {self.split_fold}")
            print(split_generator.summary_)

        # init dataset
        dataset = S2_Dataset(
            split_meta_df,
            sel_time=self.sel_time,
            sel_band=self.sel_band,
            sel_task=self.sel_task,
        )

        # get dataset characteristics
        self.num_classes = len(dataset.labels_unique)
        num_channels = dataset[0][0].shape[0]

        # initialize the segmentation model
        self.model = SegmentationModel(
            model_name=self.model_name,
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=num_channels,
            num_classes=1 if self.num_classes == 2 else self.num_classes,
            train_mode=self.train_mode,
        )
        self.model.init_model()
        self.model.summarize(verbose=self.verbose)
        for k, v in self.model.summary_stats.items():
            setattr(self, k, v)

        # define transforms
        if self.train_augmentations:
            data_transforms = {
                "train": A.Compose(
                    [
                        A.RandomRotate90(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    seed=self.seed,
                ),
                "val": None,
                "test": None,
            }
        else:
            data_transforms = {"train": None, "val": None, "test": None}

        # re-init datasets for different splits & applying transforms
        datasets = {
            split: S2_Dataset(
                meta_file=split_meta_df[split_meta_df["split"] == split],
                sel_time=self.sel_time,
                sel_band=self.sel_band,
                sel_task=self.sel_task,
                normalize=True,
                transform=data_transforms[split],
                use_cache=self.use_cache,
                cache_dir=os.path.dirname(os.path.abspath(self.meta_file)),
            )
            for split in ["train", "val", "test"]
        }

        # init metric collection
        mode = "binary" if self.num_classes == 2 else "multiclass"
        self.metric_collection = MetricCollection(
            mode=mode,
            num_classes=1 if self.num_classes == 2 else self.num_classes,
            ignore_index=-99,
        )

        # dataloader parameters
        loader_args = {
            "batch_size": self.batch_size,
            "num_workers": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1
            ),
            "pin_memory": True,
            "generator": torch.Generator().manual_seed(self.seed),
            "persistent_workers": True,
        }
        self.dataloaders = {
            split: DataLoader(
                datasets[split],
                **loader_args,
                shuffle=(split == "train"),
            )
            for split in ["train", "val", "test"]
        }
        self.size_train_set = len(datasets["train"])
        self.size_val_set = len(datasets["val"])
        self.size_test_set = len(datasets["test"])

        # update logs
        self._update_logs()

    def train_model(self):
        """Train model"""
        # check GPU availability & send the model to GPU
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        model_ft = self.model.model.to(device)
        # gather params to be updated
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        # set up optimizer
        if self.optimizer == "SGD":
            optimizer_ft = optim.SGD(params_to_update, lr=self.lr, momentum=0.9)
        elif self.optimizer == "Adam":
            optimizer_ft = optim.Adam(params_to_update, lr=self.lr)
        # set up learning rate scheduler
        lr_scheduler = CosineScheduler(
            max_update=self.max_epochs,
            base_lr=self.lr,
            final_lr=0.1 * self.lr,
            warmup_steps=5,
        )
        # set up loss
        mode = "binary" if self.num_classes == 2 else "multiclass"
        criterion = SegmentationLoss(
            loss_name=self.loss, mode=mode, **self.loss_params, ignore_index=-99
        ).fn
        # initialise early stopping
        early_stopping = EarlyStopping(
            patience=self.patience, path=os.path.join(self.save_dir, "model_params.pt")
        )
        # create logging objects
        self.loss_history = {"train": [], "val": [], "test": []}
        self.metric_history = {"train": [], "val": [], "test": []}
        # start training
        since = time.time()
        completed_epochs = 0
        for epoch in range(self.max_epochs):
            # update learning rate
            lr = lr_scheduler(epoch)
            for param_group in optimizer_ft.param_groups:
                param_group["lr"] = lr
            # evaluate early stopping condition
            if early_stopping.early_stop:
                break
            print("Epoch {}/{}".format(epoch, self.max_epochs - 1))
            print("-" * 10)
            for phase in ["train", "val", "test"]:
                if phase == "train":
                    model_ft.train()
                else:
                    model_ft.eval()
                self.metric_collection.reset()
                self.metric_collection.set_stage("train")
                running_loss = torch.zeros((), device=device)
                sample_count = 0
                # iterate over data
                for inputs, labels, _ in tqdm(
                    self.dataloaders[phase], desc="Iterate over batches"
                ):
                    # forward pass
                    optimizer_ft.zero_grad()
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device).long()

                    with torch.set_grad_enabled(phase == "train"):
                        logits = model_ft(inputs)
                        if mode == "binary":
                            labels = labels.unsqueeze(1)
                        loss = criterion(logits, labels)
                        if mode == "multiclass":
                            probs = logits.softmax(dim=1)
                            preds = probs.argmax(dim=1)
                        elif mode == "binary":
                            probs = logits.squeeze(1).sigmoid()
                            preds = (probs > 0.5).int()
                    # backward pass only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer_ft.step()
                    # add iteration statistics to epoch summary
                    if mode == "binary":
                        preds = preds.unsqueeze(1)
                        probs = probs.unsqueeze(1)
                    self.metric_collection.update(preds, probs, labels)
                    batch_size = inputs.size(0)
                    running_loss += loss.detach() * batch_size
                    sample_count += batch_size
                # calculate/log epoch loss & accs
                epoch_loss = (running_loss / sample_count).item()
                epoch_metrics = self.metric_collection.compute()
                self.loss_history[phase].append(epoch_loss)
                self.metric_history[phase].append(
                    {
                        k: v
                        for k, v in epoch_metrics.items()
                        if isinstance(v, float) or isinstance(v, int)
                    }
                )
                # log & print information
                if self.wandb_project is not None:
                    wandb.log(
                        {
                            f"loss_{phase}": epoch_loss,
                            **{f"{k}_{phase}": v for k, v in epoch_metrics.items()},
                        },
                        step=epoch,
                    )
                print(
                    f"{phase}\t loss: {epoch_loss:.4f}, "
                    + ", ".join(
                        [
                            f"{k}: {v:.4f}"
                            for k, v in epoch_metrics.items()
                            if isinstance(v, float) or isinstance(v, int)
                        ]
                    )
                )
                if phase == "test":
                    completed_epochs += 1
                    print()
                # evaluate early stopping
                if phase == "val":
                    early_stopping(epoch_loss, model_ft)

        # load best model
        model_ft.load_state_dict(
            torch.load(
                os.path.join(self.save_dir, "model_params.pt"), weights_only=True
            )
        )
        self.model_ft = model_ft

        # update logs
        self.training_epochs = completed_epochs
        self.training_time_total = round(time.time() - since, 2)
        self.training_time_epoch = round(
            self.training_time_total / self.training_epochs, 2
        )
        pd.DataFrame(self.loss_history).to_csv(
            os.path.join(self.save_dir, "model_loss.csv"), index=False
        )
        pd.DataFrame(self.metric_history).to_csv(
            os.path.join(self.save_dir, "model_metrics.csv"), index=False
        )
        self._update_logs()

        # plot final metrics
        for phase in ["train", "val", "test"]:
            # load label information
            ds = self.dataloaders[phase].dataset
            label_map = ds.label_map_df[["target_encoding", "target_semantics"]].copy()
            label_map.sort_values(by="target_encoding", inplace=True)
            labels_str = label_map["target_semantics"].drop_duplicates().to_numpy()
            labels_str = [str(x) for x in labels_str]
            # apply model
            _ = self.apply_model(phase)
            # compute metrics
            self.metric_collection.compute()
            if self.wandb_project is not None:
                for mk, mv in self.metric_collection.results.items():
                    if isinstance(mv, float) or isinstance(mv, int):
                        wandb.run.summary[f"best_{mk}_{phase}"] = mv
            # plot confusion matrix & report
            self.metric_collection.plot_confusion_matrix(
                out_dir=os.path.join(self.save_dir, phase),
                class_names=labels_str,
                log_scaled=True,
            )
            self.metric_collection.plot_report(
                out_dir=os.path.join(self.save_dir, phase),
                class_names=labels_str,
            )
            # evaluate covariate performance
            if phase in ("val", "test"):
                _ = CovariateEvaluator.from_patchwise(
                    config=os.path.join(self.save_dir, "config.json"),
                    split=phase,
                    out_dir=os.path.join(self.save_dir, phase, "covariates"),
                )

        # finalize wandb session
        if self.wandb_project is not None:
            wandb.finish()

        # save log file training completed
        with open(os.path.join(self.save_dir, "training_completed.txt"), "w") as f:
            f.write("True")

    @torch.no_grad()
    def apply_model(
        self,
        split="test",
        meta_df=None,
        model_params=None,
        eval_metrics=True,
        return_res=False,
    ):
        """
        Run inference on a set of samples using the trained model.

        Args:
            split (str): Which split of the data loader to use for inference. Default is "test".
            meta_df (DataFrame, optional): Custom metadata dataframe with sample paths,
                e.g. rows of the meta CSV; relative paths are resolved against the folder of
                `self.meta_file`. If None, uses self.dataloaders[split]. Default is None.
            model_params (str, optional): Path to model checkpoint for loading weights.
                If None, uses the model already loaded in self.model_ft.
            eval_metrics (bool): Whether to compute metrics on the predictions. Default is True.
            return_res (bool): Whether to return the inference results as a dict of
                indices, inputs, targets, predictions & probabilities. Default is False.

        Returns:
            (res, ds) (tuple, optional): Tuple containing inference results
            and the dataset if return_res is True. Inference keys are:
                * indices (Tensor): Indices of the samples in the dataset (N,).
                * inputs (Tensor): Input data (N, C, H, W).
                * targets (Tensor): Ground truth labels (N, H, W).
                * preds (Tensor): Predicted class indices (N, H, W).
                * probs (Tensor): Probabilities (N, C, H, W).
        """
        # check that model is loaded or already trained
        if model_params is not None:
            if not hasattr(self, "model"):
                self.prep_model()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_ft = self.model.model.to(device)
            model_ft.load_state_dict(
                torch.load(model_params, weights_only=True, map_location=device)
            )
            self.model_ft = model_ft
            self.model_ft.eval()
        else:
            if not hasattr(self, "model_ft"):
                raise ValueError("Model not trained yet and no model_params provided.")

        # setup
        device = next(self.model_ft.parameters()).device
        mode = "binary" if self.num_classes == 2 else "multiclass"
        if eval_metrics:
            self.metric_collection.set_stage("eval")
            self.metric_collection.reset()

        # get dataloader
        if meta_df is not None:
            meta_df = resolve_paths(meta_df, data_dir_of(self.meta_file))
            # re-init datasets for different splits & applying transforms
            ds = S2_Dataset(
                meta_file=meta_df,
                sel_time=self.sel_time,
                sel_band=self.sel_band,
                sel_task=self.sel_task,
                transform=None,
                use_cache=self.use_cache,
                cache_dir=os.path.dirname(os.path.abspath(self.meta_file)),
            )
            # dataloader parameters
            loader_args = {
                "batch_size": self.batch_size,
                "num_workers": (
                    len(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity")
                    else os.cpu_count() or 1
                ),
                "pin_memory": True,
                "generator": torch.Generator().manual_seed(self.seed),
                "persistent_workers": True,
            }
            dl = DataLoader(ds, **loader_args, shuffle=False)
        else:
            dl = self.dataloaders[split]

        # disable augmentations during inference
        orig_transform = dl.dataset.transform
        dl.dataset.transform = None

        # iterate over dataloader
        res = {"indices": [], "inputs": [], "targets": [], "preds": [], "probs": []}
        for inputs, labels, indices in tqdm(
            dl,
            desc=f"Inference on {len(dl.dataset)} samples",
            total=(len(dl.dataset) + self.batch_size - 1) // self.batch_size,
        ):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device).long()
            logits = self.model_ft(inputs)
            if mode == "multiclass":
                probs = logits.softmax(dim=1)
                preds = probs.argmax(dim=1)
            else:
                probs = logits.squeeze(1).sigmoid().unsqueeze(1)
                preds = (probs > 0.5).long()
                labels = labels.unsqueeze(1)
            if eval_metrics:
                self.metric_collection.update(preds, probs, labels)
            if return_res:
                res["indices"].append(indices.cpu())
                res["inputs"].append(inputs.cpu())
                res["targets"].append(labels.cpu())
                res["preds"].append(preds.cpu())
                res["probs"].append(probs.cpu())

        # aggregate results if requested
        if return_res:
            for key in res.keys():
                res[key] = torch.cat(res[key], dim=0)

        # restore original transform
        dl.dataset.transform = orig_transform

        if return_res:
            return res, dl.dataset
        else:
            return None

    def _update_logs(self):
        """
        Write config logs to WandB
        """
        configs = {
            k: v
            for k, v in self.__dict__.items()
            if (type(v) in [str, int, float, bool, dict, list] or v is None)
            and k not in ["dataloaders", "loss_history", "metric_history"]
        }
        if self.wandb_project is not None:
            wandb.config.update(configs)
