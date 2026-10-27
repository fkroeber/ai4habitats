"""
Timeseries models in TensorFlow and PyTorch.
Model interfaces of the sktime library are configured.
"""

import io
import re
import torch
from sktime.classification.deep_learning import (
    CNNClassifier,
    ConvTranClassifierTorch,
    FCNClassifier,
    InceptionTimeClassifier,
    LSTMFCNClassifier,
    MACNNClassifier,
    MCDCNNClassifier,
    MLPClassifier,
    MVTSTransformerClassifier,
    ResNetClassifier,
    TapNetClassifier,
)
from torchinfo import summary as torch_summary


class TempModelConfigurator:
    """Pre-configured temporal classification model"""

    MODELS = {
        "mlp": "tensorflow",
        "cnn": "tensorflow",
        "convtran": "pytorch",
        "fcn": "tensorflow",
        "resnet": "tensorflow",
        "inception": "tensorflow",
        "lstm_fcn": "tensorflow",
        "macnn": "tensorflow",
        "mcdcnn": "tensorflow",
        "tapnet": "tensorflow",
        "mvts": "pytorch",
    }

    def __init__(self, model_name, n_timesteps=20, n_features=10, num_classes=2):
        """
        Args:
            model_name (str): name of the model to be configured
            n_timesteps (int): number of timesteps in the input data
                Has a default value of 20.
            n_features (int): number of features in the input data
                Has a default value of 10.
            num_classes (int): number of output classes
                Has a default value of 2.
        """
        self.model_name = model_name
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.num_classes = num_classes
        if self.model_name not in self.MODELS:
            raise ValueError(
                f"❌ Model '{model_name}' is not available."
                f" Available models are: {', '.join(self.MODELS.keys())}"
            )
        self.library = self.MODELS.get(model_name, None)

    def init_model(self, **init_kwargs):
        """
        Initialise the model with the given parameters.

        Note: Final activation is softmax regardless of num_classes since
        in sktime binary classification is currently handled as multi-class
        classification with two classes.

        Args:
            init_kwargs (dict): initialisation kwargs for the model
        """
        init_kwargs["activation"] = "softmax"
        if self.model_name == "mlp":
            self.model = MLPClassifier(**init_kwargs)
        elif self.model_name == "cnn":
            self.model = CNNClassifier(**init_kwargs)
        elif self.model_name == "fcn":
            self.model = FCNClassifier(**init_kwargs)
        elif self.model_name == "resnet":
            self.model = ResNetClassifier(**init_kwargs)
        elif self.model_name == "inception":
            self.model = InceptionTimeClassifier(**init_kwargs)
        elif self.model_name == "lstm_fcn":
            self.model = LSTMFCNClassifier(attention=False, **init_kwargs)
        elif self.model_name == "macnn":
            self.model = MACNNClassifier(**init_kwargs)
        elif self.model_name == "mcdcnn":
            self.model = MCDCNNClassifier(**init_kwargs)
        elif self.model_name == "tapnet":
            self.model = TapNetClassifier(**init_kwargs)
        elif self.model_name == "convtran":
            init_kwargs.pop("activation", None)
            self.model = ConvTranClassifierTorch(
                emb_size=64,  # from paper
                abs_pos_encoding_scheme="learn",  # to enable discontinuous monthly gap time series
                rel_pos_encoding_scheme="erpe",
                **init_kwargs,
            )
        elif self.model_name == "mvts":
            # default config with batch size 128 as specified in the paper
            init_kwargs.pop("activation", None)
            self.model = MVTSTransformerClassifier(
                activation="gelu",
                pos_encoding="learnable",
                d_model=128,
                dim_feedforward=256,
                n_heads=16,
                num_layers=3,
                **init_kwargs,
            )
        if self.library == "tensorflow":
            self.network = self.model.build_model(
                input_shape=(self.n_timesteps, self.n_features),
                n_classes=self.num_classes,
            )
        else:
            self.network = self.model._build_network(
                X=None,
                y=None,
                X_shape=(32, self.n_features, self.n_timesteps),
                num_classes=self.num_classes,
            )

        return self.model

    def summarize(self, verbose=True):
        """
        Print a summary of the model architecture.

        Args:
            verbose (bool): whether to print the summary or not
        """
        if self.library == "tensorflow":
            self.summary = self._keras_get_model_summary(self.network)
            self.summary_stats = self._keras_parse_summary_stats(self.summary)
        else:
            # define general torchinfo summary kwargs
            summary_kwargs = {
                "col_names": ["input_size", "output_size", "num_params", "trainable"],
                "depth": 4,
            }
            if "mvts" in self.model_name:
                # specify required input data shapes for model
                summary_kwargs["input_size"] = [
                    (32, self.n_timesteps, self.n_features),
                    (32, self.n_timesteps),
                ]
                summary_kwargs["dtypes"] = [torch.float32, torch.bool]
            elif "convtran" in self.model_name:
                summary_kwargs["input_size"] = (32, self.n_features, self.n_timesteps)
            self.summary = torch_summary(self.network, **summary_kwargs)
            self.summary_stats = {
                "total_params": self.summary.total_params,
                "trainable_params": self.summary.trainable_params,
            }

        if verbose:
            print(self.summary)

    def _keras_get_model_summary(self, model):
        """Parse model summary as string"""
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_str = stream.getvalue()
        stream.close()
        return summary_str

    def _keras_parse_summary_stats(self, summary_str):
        """Extract relevant statistics from the summary string"""
        pattern = r"(Total params|Trainable params): ([\d,]+) \(([\d.]+ [A-Z]+)\)"
        matches = re.findall(pattern, summary_str)
        stats = {}
        for name, count, size in matches:
            key = name.lower().replace(" ", "_")
            stats[key] = int(count.replace(",", ""))
        return stats
