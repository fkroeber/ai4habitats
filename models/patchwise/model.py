import segmentation_models_pytorch as smp
import warnings
from torchinfo import summary


class SegmentationModel:
    """
    Initialising SOTA deep learning models for image segmentation.
    """

    def __init__(self, **kwargs):
        """
        Models can be retrieved with/without pre-trained weights and
        with the last layer being adjusted to the number of required classes.

        Args:
            model_name (str): smp class name of the architecture (e.g., 'Unet', 'DeepLabV3Plus', 'Segformer', 'DPT')
                See "https://smp.readthedocs.io/en/latest/models.html" for all available models.
            encoder_name (str): encoder backbone (e.g., 'resnet34', 'mit_b0', 'efficientnet-b0')
                See "https://smp.readthedocs.io/en/latest/encoders.html" for all available encoders.
            encoder_weights (str): pretrained weights to use ('imagenet' or None)
            in_channels (int): number of input channels
            num_classes (int): number of output segmentation classes
            train_mode (str): degree of use of pretrained models ('scratch', 'finetune', 'feat_extract')
        """
        self.model_name = kwargs.get("model_name", "unet")
        self.encoder_name = kwargs.get("encoder_name", "resnet18")
        self.encoder_weights = kwargs.get("encoder_weights", "imagenet")
        self.in_channels = kwargs.get("in_channels", 3)
        self.num_classes = kwargs.get("num_classes", 10)
        self.train_mode = kwargs.get("train_mode", "finetune")
        # intialise further attributes
        self.model = None
        self.summary = None
        self.summary_stats = None
        # check validity of mode
        allowed_modes = ["scratch", "finetune", "feat_extract"]
        if self.train_mode not in allowed_modes:
            raise ValueError(
                f"Invalid argument for mode: {self.train_mode}. Allowed args are: {allowed_modes}"
            )
        # translate mode into information about usage of pretrained weights
        if self.train_mode == "scratch" and self.encoder_weights is not None:
            warnings.warn(
                f"Using pretrained weights '{self.encoder_weights}' with 'scratch' mode is possible. "
                "Encoder weights will be set to None."
            )
            self.encoder_weights = None

    def init_model(self):
        """
        For each model the model object along with necessary transforms
        to make the use of pretrained weights applicable is returned.
        """
        # fetch model
        ModelClass = getattr(smp, self.model_name)
        if self.model_name == "DPT":
            init_kwargs = {"dynamic_img_size": True}
        else:
            init_kwargs = {}
        model = ModelClass(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.num_classes,
            **init_kwargs,
        )
        # freeze weigths
        if self.train_mode == "feat_extract":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.segmentation_head.parameters():
                param.requires_grad = True
        # write as instance attributes
        self.model = model

    def summarize(self, verbose=True, **kwargs):
        """
        Print a summary of the model architecture.

        Args:
            verbose (bool): whether to print the summary or not
            **kwargs: any keyword arg that can be passed to `torchinfo.summary`
        """

        default_kwargs = {
            "input_size": (1, self.in_channels, 256, 256),
            "col_names": ["input_size", "output_size", "num_params", "trainable"],
            "depth": 4,
        }
        default_kwargs.update(kwargs)

        self.summary = summary(self.model, verbose=0, **default_kwargs)
        self.summary_stats = {
            "total_params": self.summary.total_params,
            "trainable_params": self.summary.trainable_params,
            "size_input": self.summary.total_input,
            "size_params": self.summary.total_param_bytes,
            "size_forward_backward_pass": self.summary.total_output_bytes,
            "size_total": self.summary.total_input
            + self.summary.total_param_bytes
            + self.summary.total_output_bytes,
        }

        if verbose:
            print(self.summary)
