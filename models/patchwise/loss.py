import segmentation_models_pytorch as smp


class SegmentationLoss:
    """
    Initialising loss functions for image segmentation.
    """

    def __init__(self, **kwargs):
        """
        Args:
            loss_name (str): name of the loss function, e.g., 'DiceLoss', 'FocalLoss', etc.
                see https://smp.readthedocs.io/en/latest/losses.html for available options;
                'CrossEntropyLoss' maps to smp's SoftBCEWithLogitsLoss (binary) or
                SoftCrossEntropyLoss (multiclass). Note that smp's JaccardLoss does not accept
                the `ignore_index` the Trainer passes.
            mode (str): mode of the loss function, e.g., 'binary', 'multiclass'
            All other kwargs are forwarded to the respective loss constructor
        """
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
