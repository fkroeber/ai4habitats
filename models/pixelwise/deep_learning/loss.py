"""
Loss functions for training models in PyTorch and TensorFlow.
"""

import segmentation_models_pytorch as smp
import tensorflow as tf
import torch
from tensorflow.keras import backend as K


class SegmentationLoss:
    """
    Top-level loss factory used to instantiate a specific tensorflow or pytorch loss.

    Loss function inputs are adapted to the definition of the network inputs/outputs in sktime.
    -> tensorflow models provide probabilistic outputs & losses expect one-hot encoded labels
    -> pytorch models provide logits & losses expect label class indices

    Loss functions modes differentiate between binary and multiclass. For binary mode, losses
    are designed to work with two-output models, since this is the default for binary models
    in sktime (instead of single outputs).

    Class weights for losses are only supported for TensorFlow models and need to be provided
    upon model.fit() call, not during loss instantiation.
    """

    def __init__(self, backend="torch", loss_name=None, **kwargs):
        """
        Args:
            backend (str): 'torch' or 'tf'
            loss_name (str): loss name without backend prefix, e.g. "CrossentropyLoss",
                "FocalLoss", "DiceLoss", "JaccardLoss", "TverskyLoss"
            kwargs: all other keyword arguments to forward to the loss constructor
        """
        self.backend = backend.lower()
        self.loss_name = loss_name

        # remove weighting as parameter if given
        # will be applied later via model.fit() for TF models
        kwargs.pop("weighted", None)

        # resolve full class name like "TF_DiceLoss" or "PT_FocalLoss"
        prefix = "TF_" if self.backend == "tf" else "PT_"
        full_loss_name = f"{prefix}{self.loss_name}"

        # instantiate the loss
        try:
            self.loss_class = globals()[full_loss_name]
        except KeyError:
            raise ValueError(f"Loss class {full_loss_name} not found.")
        self.fn = self.loss_class(**kwargs)


# --- tensorflow losses --- #
class TF_CrossentropyLoss(tf.keras.losses.CategoricalCrossentropy):
    def __init__(self, **kwargs):
        kwargs.pop("mode", None)  # ignored since same for binary and multiclass
        super().__init__(**kwargs)


class TF_FocalLoss(tf.keras.losses.CategoricalFocalCrossentropy):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_focal_crossentropy",
        **kwargs,
    ):
        kwargs.pop("mode", None)  # ignored since same for binary and multiclass
        super().__init__(
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
            **kwargs,
        )


class TF_TverskyLoss(tf.keras.losses.Loss):
    """
    Custom Keras multi-class Tversky loss function.
    modified src: https://www.tensorflow.org/api_docs/python/tf/keras/losses/tversky
    """

    def __init__(
        self, name="tversky_loss", mode="binary", alpha=0.5, beta=0.5, **kwargs
    ):
        super().__init__(name=name)
        self.mode = mode
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        """
        Computes loss. Labels are expected to be provided in a one_hot representation
        (according to the default Keras loss function interface). Predictions are expected
        to be probabilities not logits (e.g., output of a softmax layer or
        sigmoid in binary case).

        Args:
            y_true: Tensor of shape (N, C), with one-hot encoded class labels.
            y_pred: Tensor of shape (N, C) where N=batch, C=classes.
        """
        if self.mode == "binary":
            y_true = tf.cast(y_true[:, 1], tf.float32)
            y_pred = tf.cast(y_pred[:, 1], tf.float32)
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])

            intersection = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1 - y_pred))
            tversky = (intersection + K.epsilon()) / (
                intersection + self.alpha * fp + self.beta * fn + K.epsilon()
            )
            return 1 - tversky
        else:
            return self.tversky_multiclass(y_true, y_pred)

    def tversky_multiclass(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        def tversky_per_class(c):
            y_true_c = y_true[..., c]
            y_pred_c = y_pred[..., c]
            y_true_c = tf.reshape(y_true_c, [-1])
            y_pred_c = tf.reshape(y_pred_c, [-1])
            intersection = tf.reduce_sum(y_true_c * y_pred_c)
            fp = tf.reduce_sum((1 - y_true_c) * y_pred_c)
            fn = tf.reduce_sum(y_true_c * (1 - y_pred_c))
            return (intersection + K.epsilon()) / (
                intersection + self.alpha * fp + self.beta * fn + K.epsilon()
            )

        num_classes = tf.shape(y_pred)[-1]
        class_indices = tf.range(num_classes)
        tversky_scores = tf.map_fn(
            tversky_per_class,
            class_indices,
            fn_output_signature=tf.TensorSpec(shape=(), dtype=y_pred.dtype),
        )
        return 1 - tf.reduce_mean(tversky_scores)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "beta": self.beta})
        return config


class TF_DiceLoss(tf.keras.losses.Loss):
    def __init__(self, name="dice_loss", mode="binary", epsilon=1e-8, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mode = mode
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        if self.mode == "binary":
            y_true = tf.cast(y_true[:, 1], tf.float32)
            y_pred = tf.cast(y_pred[:, 1], tf.float32)
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])

            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            return 1 - dice
        else:
            return self._dice_multiclass(y_true, y_pred)

    def _dice_multiclass(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        def dice_per_class(c):
            y_true_c = tf.reshape(y_true[..., c], [-1])
            y_pred_c = tf.reshape(y_pred[..., c], [-1])
            intersection = tf.reduce_sum(y_true_c * y_pred_c)
            union = tf.reduce_sum(y_true_c) + tf.reduce_sum(y_pred_c)
            score = (2 * intersection + self.epsilon) / (union + self.epsilon)
            return score

        num_classes = tf.shape(y_pred)[-1]
        class_indices = tf.range(num_classes)
        dice_scores = tf.map_fn(
            dice_per_class,
            class_indices,
            fn_output_signature=tf.TensorSpec(shape=(), dtype=y_pred.dtype),
        )
        return 1 - tf.reduce_mean(dice_scores)

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode, "epsilon": self.epsilon})
        return config


class TF_JaccardLoss(tf.keras.losses.Loss):
    def __init__(self, name="jaccard_loss", mode="binary", epsilon=1e-8, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mode = mode
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        if self.mode == "binary":
            y_true = tf.cast(y_true[:, 1], tf.float32)
            y_pred = tf.cast(y_pred[:, 1], tf.float32)
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])

            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            jaccard = (intersection + self.epsilon) / (union + self.epsilon)
            return 1 - jaccard
        else:
            return self._jaccard_multiclass(y_true, y_pred)

    def _jaccard_multiclass(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        def jaccard_per_class(c):
            y_true_c = tf.reshape(y_true[..., c], [-1])
            y_pred_c = tf.reshape(y_pred[..., c], [-1])
            intersection = tf.reduce_sum(y_true_c * y_pred_c)
            union = tf.reduce_sum(y_true_c) + tf.reduce_sum(y_pred_c) - intersection
            score = (intersection + self.epsilon) / (union + self.epsilon)
            return score

        num_classes = tf.shape(y_pred)[-1]
        class_indices = tf.range(num_classes)
        jaccard_scores = tf.map_fn(
            jaccard_per_class,
            class_indices,
            fn_output_signature=tf.TensorSpec(shape=(), dtype=y_pred.dtype),
        )
        return 1 - tf.reduce_mean(jaccard_scores)

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode, "epsilon": self.epsilon})
        return config


# --- pytorch losses --- #
class PT_BaseLoss(torch.nn.Module):
    """
    Pytorch base class for segmentation losses.
    """

    def __init__(self, **kwargs):
        """
        Args:
            loss_name (str): name of the loss function, e.g., 'DiceLoss', 'FocalLoss', etc.
                see https://smp.readthedocs.io/en/latest/losses.html for available options;
                'CrossEntropyLoss' maps to smp's SoftBCEWithLogitsLoss / SoftCrossEntropyLoss
            mode (str): mode of the loss function, e.g., 'binary', 'multiclass'
            All other kwargs are forwarded to the respective loss constructor
        """
        super().__init__()
        self.loss_name = kwargs.pop("loss_name", "DiceLoss")
        self.mode = kwargs.pop("mode", "multiclass")
        self.loss_kwargs = kwargs
        self.fn = self._get_loss_function()

    def _get_loss_function(self):
        if self.loss_name == "CrossEntropyLoss":
            epsilon_smooth = 1e-8
            if self.mode == "binary":
                LossClass = getattr(smp.losses, "SoftBCEWithLogitsLoss")(
                    smooth_factor=epsilon_smooth, **self.loss_kwargs
                )
            elif self.mode == "multiclass":
                LossClass = getattr(smp.losses, "SoftCrossEntropyLoss")(
                    smooth_factor=epsilon_smooth, **self.loss_kwargs
                )
            else:
                raise ValueError(f"Invalid mode '{self.mode}' for CrossEntropyLoss.")
        else:
            LossClass = getattr(smp.losses, self.loss_name)(
                mode=self.mode, **self.loss_kwargs
            )
        return LossClass

    def forward(self, y_pred, y_true):
        """
        Forward pass for the loss function.

        Args:
            y_pred: Tensor of shape (N, C, H, W) where N=batch, C=classes.
            y_true: Tensor of shape (N, H, W) with class indices (not one-hot encoded labels).

        Returns:
            Loss value as a scalar tensor.
        """
        if self.mode == "binary":
            # assuming two outputs channels, take the second channel
            y_pred = y_pred[:, 1].reshape(-1, 1)
            y_true = y_true.unsqueeze(1)
        return self.fn(y_pred, y_true)


class PT_JaccardLoss(PT_BaseLoss):
    """
    Jaccard loss as standalone nn.Module using PT_BaseLoss.
    """

    def __init__(self, mode="binary", **kwargs):
        super().__init__(loss_name="JaccardLoss", mode=mode, **kwargs)


class PT_CrossentropyLoss(PT_BaseLoss):
    """
    CrossEntropyLoss loss as standalone nn.Module using PT_BaseLoss.
    """

    def __init__(self, mode="binary", **kwargs):
        super().__init__(loss_name="CrossEntropyLoss", mode=mode, **kwargs)


class PT_DiceLoss(PT_BaseLoss):
    """
    Dice loss as standalone nn.Module using PT_BaseLoss.
    """

    def __init__(self, mode="binary", **kwargs):
        super().__init__(loss_name="DiceLoss", mode=mode, **kwargs)


class PT_TverskyLoss(PT_BaseLoss):
    """
    Tversky loss as standalone nn.Module using PT_BaseLoss.
    """

    def __init__(self, mode="binary", **kwargs):
        super().__init__(loss_name="TverskyLoss", mode=mode, **kwargs)


class PT_FocalLoss(PT_BaseLoss):
    """
    Focal loss as standalone nn.Module using PT_BaseLoss.
    """

    def __init__(self, mode="binary", **kwargs):
        super().__init__(loss_name="FocalLoss", mode=mode, **kwargs)


# register tensorflow loss functions
tf.keras.utils.get_custom_objects().update(
    {
        "TF_TverskyLoss": TF_TverskyLoss,
        "TF_CrossentropyLoss": TF_CrossentropyLoss,
        "TF_FocalLoss": TF_FocalLoss,
        "TF_DiceLoss": TF_DiceLoss,
        "TF_JaccardLoss": TF_JaccardLoss,
    }
)
