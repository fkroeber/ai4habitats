import math
import numpy as np
import os
import time
import torch
import warnings


class CosineScheduler:
    """Cosine learning-rate schedule with linear warmup (epoch-based).

    Args:
        max_update (int): Total number of training epochs.
        base_lr (float): Peak learning rate reached at the end of warmup.
        final_lr (float): Learning rate at max_update after cosine decay.
        warmup_steps (int): Number of warmup epochs.
    """

    def __init__(self, max_update, base_lr=0.01, final_lr=0.0, warmup_steps=5):
        if warmup_steps >= max_update:
            warnings.warn(
                f"warmup_steps ({warmup_steps}) >= max_update ({max_update}); "
                "clamping to max_update - 1."
            )
            warmup_steps = max_update - 1

        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = max(warmup_steps, 1)
        self.max_steps = max(max_update - self.warmup_steps, 1)

    def get_warmup_lr(self, epoch):
        return self.base_lr_orig * float(epoch + 1) / float(self.warmup_steps)

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch > self.max_update:
            return self.final_lr
        return (
            self.final_lr
            + (self.base_lr_orig - self.final_lr)
            * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps))
            / 2
        )


# credits: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Initialise early stopping to retrieve best model after patience epochs without increase in validation loss"""

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # save using atomic swap & retry mechanism
        tmp = f"{self.path}.tmp"
        for attempt in range(3):
            try:
                torch.save(model.state_dict(), tmp)
                os.replace(tmp, self.path)
                break
            except (RuntimeError, OSError) as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                if attempt == 2:
                    raise RuntimeError(
                        f"Checkpoint save to {self.path} failed after 3 attempts"
                    ) from e
                self.trace_func(f"Checkpoint save failed ({e}), retrying...")
                time.sleep(10 * (attempt + 1))
        self.val_loss_min = val_loss
