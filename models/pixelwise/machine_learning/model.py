"""
Pixelwise machine learning models for classification.
Model interfaces of the RAPIDS cuML library are configured.
"""

import numpy as np

from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.neighbors import KNeighborsClassifier
from cuml.svm import SVC
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if


class LogUniformInt:
    """Integer values drawn log-uniformly - k acts roughly logarithmically."""

    def __init__(self, a, b):
        self._d = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        return self._d.rvs(*args, **kwargs).astype(int)


class ResampledClassifier(ClassifierMixin, BaseEstimator):
    """Fit an estimator on a class-rebalanced subset of each training fold.

    The wrapped estimator's hyperparameters are deliberately exposed at the
    top level. Consequently, existing ``MLModelConfigurator`` search spaces can
    keep using ``C``, ``gamma``, ``n_neighbors`` etc. instead of changing to
    ``estimator__C``, ``estimator__gamma`` and so forth.
    """

    def __init__(self, estimator=None, class_weights=None, seed=None):
        self.estimator = estimator
        self.class_weights = class_weights
        self.seed = seed

    def get_params(self, deep=True):
        """Expose wrapper and estimator parameters without changing their names."""
        params = {
            "estimator": self.estimator,
            "class_weights": self.class_weights,
            "seed": self.seed,
        }

        if deep and self.estimator is not None:
            estimator_params = self.estimator.get_params(deep=True)
            # class balancing is handled exclusively through `class_weights` by resampling the training data.
            estimator_params.pop("class_weight", None)
            # flat parameters preserve the existing MLModelConfigurator interface
            params.update(
                {
                    name: value
                    for name, value in estimator_params.items()
                    if name not in params
                }
            )
            # nested parameters retain standard sklearn compatibility
            params.update(
                {
                    f"estimator__{name}": value
                    for name, value in estimator_params.items()
                }
            )
        return params

    @staticmethod
    def resample_indices(y, balance=None, n_samples=None, seed=None):
        """Return indices for a class-stratified resampled version of ``y``.

        Depending on the requested quotas, majority classes may be undersampled and
        minority classes may be oversampled.

        Args:
            y (array-like): One-dimensional class labels.
            balance (None|bool|str):
                None/False keeps the natural distribution.
                True/"balanced" assigns every class the same quota.
                "sqrt" assigns quotas proportional to the square root of class size.
            n_samples (int, optional):
                Target number of returned indices. When omitted, the original sample
                count is used.
            seed (int, optional): Random seed used for sampling within each class.

        Returns:
            ndarray: Sorted integer indices into ``y``. Indices may occur repeatedly
                when a class is oversampled.
        """
        y = np.asarray(y).ravel()

        if y.size == 0:
            return np.empty(0, dtype=np.int64)

        if balance is True:
            balance = "balanced"
        elif balance is False:
            balance = None

        if balance not in {None, "sqrt", "balanced"}:
            raise ValueError(
                "class_weights must be one of None, False, True, 'sqrt' or "
                f"'balanced', got {balance!r}."
            )

        classes, counts = np.unique(y, return_counts=True)

        n_total = len(y) if n_samples is None else int(n_samples)

        if n_total < len(classes):
            raise ValueError(
                f"n_samples={n_total} is smaller than the number of classes "
                f"({len(classes)})."
            )

        if balance == "balanced":
            weights = np.ones(len(classes), dtype=float)
        elif balance == "sqrt":
            weights = np.sqrt(counts.astype(float))
        else:
            weights = counts.astype(float)

        # allocate the requested total budget according to the selected weighting
        raw_quota = n_total * weights / weights.sum()
        quota = np.floor(raw_quota).astype(int)
        quota = np.maximum(1, quota)

        # ensure quota.sum() == n_total.
        remainder = n_total - quota.sum()
        if remainder > 0:
            order = np.argsort(-(raw_quota - np.floor(raw_quota)))
            quota[order[:remainder]] += 1
        elif remainder < 0:
            order = np.argsort(raw_quota - np.floor(raw_quota))
            for class_pos in order:
                removable = min(quota[class_pos] - 1, -remainder)
                quota[class_pos] -= removable
                remainder += removable
                if remainder == 0:
                    break

        # sample class indices according to the computed quotas, allowing oversampling if necessary
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y))
        selected = []
        for class_id, class_count, class_quota in zip(classes, counts, quota):
            class_idx = idx[y == class_id]
            selected.append(
                rng.choice(
                    class_idx,
                    size=class_quota,
                    replace=class_quota > class_count,
                )
            )
        return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)

    def set_params(self, **params):
        """Set wrapper parameters or forward flat parameters to the estimator."""
        if not params:
            return self
        # never forward a class_weight parameter to the underlying estimator
        params.pop("class_weight", None)
        params.pop("estimator__class_weight", None)
        # forward the estimator itself if given
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        # set wrapper parameters if given
        for name in ("class_weights", "seed"):
            if name in params:
                setattr(self, name, params.pop(name))
        # forward any remaining parameters to the estimator
        estimator_params = {}
        for name, value in params.items():
            if name.startswith("estimator__"):
                name = name.split("__", 1)[1]
            if name == "class_weight":
                continue
            estimator_params[name] = value
        if estimator_params:
            if self.estimator is None:
                raise ValueError(
                    "Cannot set estimator parameters before estimator is set."
                )

            self.estimator.set_params(**estimator_params)
        return self

    def fit(self, X, y, **fit_params):
        if self.estimator is None:
            raise ValueError("estimator must be provided before fit().")

        y = np.asarray(y).ravel()
        selected = ResampledClassifier.resample_indices(
            y, balance=self.class_weights, seed=self.seed
        )
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[selected], y[selected], **fit_params)
        self.classes_ = np.asarray(self.estimator_.classes_)
        self.n_fit_samples_ = int(len(selected))
        self.sample_indices_ = selected
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def _has_decision_function(self):
        estimator = getattr(self, "estimator_", self.estimator)
        return estimator is not None and hasattr(estimator, "decision_function")

    @available_if(_has_decision_function)
    def decision_function(self, X):
        return self.estimator_.decision_function(X)


class MLModelConfigurator:
    """Pre-configured pixelwise machine learning classifier"""

    MODELS = {
        "rf": "cuml",
        "knn": "cuml",
        "svm": "cuml",
        "log": "cuml",
    }

    # estimators that are sensitive to the scale of the input features and
    # therefore require standardisation of the flattened band/time features
    NEEDS_SCALING = {"knn", "svm", "log"}
    # estimators exposing a `random_state` argument
    SUPPORTS_SEED = {"rf", "svm"}
    # rough upper bound of training samples that is still practical on a single
    # GPU: kNN keeps the full reference set (O(n_train * n_query)), the kernel
    # SVM scales ~O(n^2). `None` means "no practical limit".
    MAX_SAMPLES = {
        "rf": None,
        "log": None,
        "knn": 1000000,
        "svm": 1000000,
    }

    # ------------------------------------------------------------------ #
    # standard configuration - used when Trainer(tuning=None)
    # ------------------------------------------------------------------ #
    DEFAULTS = {
        # Random Forest: strong, low-maintenance baseline for spectral-temporal
        # features - scale invariant and robust against correlated bands.
        "rf": {
            "n_estimators": 100,
            "split_criterion": "gini",
            "max_depth": 16,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "bootstrap": True,
            "max_samples": 1.0,
            "n_bins": 128,
            "n_streams": 4,
            "class_weights": None,
        },
        # k Nearest Neighbours: non-parametric reference, no training cost but
        # expensive at inference time (brute force search over the train set).
        "knn": {
            "n_neighbors": 5,
            "algorithm": "brute",
            "metric": "euclidean",
            "weights": "uniform",
            "class_weights": None,
        },
        # Kernel SVM: strong on moderate sample sizes, needs scaled features.
        # `probability=True` fits Platt scaling on top - required for the
        # probability based metrics (AUC etc.) of the MetricCollection.
        "svm": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "tol": 1e-3,
            "cache_size": 4096.0,
            "max_iter": -1,
            "probability": True,
            "class_weights": None,
        },
        # Multinomial logistic regression: linear reference model, fast and a
        # useful sanity check on how much non-linearity the task actually needs.
        "log": {
            "penalty": "l2",
            "C": 1.0,
            "tol": 1e-4,
            "max_iter": 1000,
            "class_weights": None,
        },
    }

    # ------------------------------------------------------------------ #
    # exhaustive search space - Trainer(tuning="grid")
    # ------------------------------------------------------------------ #
    GRIDS = {
        # 3 * 3 * 2 * 4 * 2 = 144 candidates
        "rf": {
            "n_estimators": [100, 300, 500],
            "max_depth": [8, 16, 24],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 2, 4, 8],
            "class_weights": [None, "sqrt"],
        },
        # 10 * 2 * 2 = 40 candidates
        "knn": {
            "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 50, 100],
            "metric": ["euclidean", "cosine"],
            "class_weights": [None, "sqrt"],
        },
        # 5 * 5 * 2 = 50 candidates
        "svm": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", 0.01, 0.1, 1.0, 10.0],
            "class_weights": [None, "sqrt"],
        },
        # 6 * 3 * 2 = 36 candidates
        "log": {
            "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
            "penalty": [None, "l1", "l2"],
            "class_weights": [None, "sqrt"],
        },
    }

    # ------------------------------------------------------------------ #
    # randomised search space - Trainer(tuning="random")
    # ------------------------------------------------------------------ #
    DISTRIBUTIONS = {
        "rf": {
            "n_estimators": randint(100, 800),
            "max_depth": randint(6, 32),
            "max_features": uniform(0.1, 0.7),
            "min_samples_leaf": randint(1, 32),
            "class_weights": [None, "sqrt"],
        },
        "knn": {
            "n_neighbors": LogUniformInt(1, 100),
            "metric": ["euclidean", "cosine"],
            "class_weights": [None, "sqrt"],
        },
        "svm": {
            "C": loguniform(1e-3, 1e3),
            "gamma": loguniform(1e-4, 1e1),
            "class_weights": [None, "sqrt"],
        },
        "log": {
            "C": loguniform(1e-4, 1e2),
            "penalty": [None, "l1", "l2"],
            "class_weights": [None, "sqrt"],
        },
    }

    def __init__(
        self,
        model_name,
        random_state=None,
    ):
        """
        Args:
            model_name (str): name of the model to be configured
            random_state (int, optional): seed forwarded to the estimators that
                support it (see SUPPORTS_SEED).
        """
        self.model_name = model_name
        self.random_state = random_state
        if self.model_name not in self.MODELS:
            raise ValueError(
                f" Model '{model_name}' is not available."
                f" Available models are: {', '.join(self.MODELS.keys())}"
            )
        self.library = self.MODELS.get(model_name, None)

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #
    def resolve_params(self, **overrides):
        """
        Merge the standard configuration with user supplied overrides.

        Args:
            overrides (dict): hyperparameters replacing the defaults

        Returns:
            params (dict): the full parameter set the estimator is built from
        """
        params = dict(self.DEFAULTS[self.model_name])
        params.update(overrides)
        if self.model_name in self.SUPPORTS_SEED and self.random_state is not None:
            params.setdefault("random_state", self.random_state)
        return params

    def init_model(self, **init_kwargs):
        """
        Initialise the estimator with the standard configuration, updated by the
        given parameters.

        Args:
            init_kwargs (dict): initialisation kwargs overriding the defaults.
            Look at the ``DEFAULTS`` attribute for the available hyperparameters.
        """
        self.params = self.resolve_params(**init_kwargs)

        # init base model
        estimator_params = dict(self.params)
        class_weights = estimator_params.pop("class_weights")
        estimator_params.pop("class_weight", None)
        estimator_params.pop("estimator__class_weight", None)
        if self.model_name == "rf":
            estimator = RandomForestClassifier(**estimator_params)
        elif self.model_name == "knn":
            estimator = KNeighborsClassifier(**estimator_params)
        elif self.model_name == "svm":
            estimator = SVC(**estimator_params)
        elif self.model_name == "log":
            estimator = LogisticRegression(**estimator_params)

        # wrap with resampled classifier
        # note: with class_weights=None it is a transparent pass-through
        self.model = ResampledClassifier(
            estimator=estimator,
            class_weights=class_weights,
            seed=self.random_state,
        )

        return self.model

    def search_space(self, strategy="grid", extend=None):
        """
        Return the hyperparameter space scanned for the configured model.

        Args:
            strategy (str):
                "grid" -> discrete value lists (GridSearchCV),
                "random" -> scipy distributions (RandomizedSearchCV)
            extend (dict, optional): additional / replacing entries

        Returns:
            space (dict): parameter name -> list of values or distribution
        """
        if strategy == "grid":
            space = dict(self.GRIDS[self.model_name])
        elif strategy == "random":
            space = dict(self.DISTRIBUTIONS[self.model_name])
        else:
            raise ValueError(
                f"❌ Tuning strategy '{strategy}' is not available."
                " Choose from 'grid', 'random' or None."
            )
        if extend:
            space.update(extend)
        return space

    def check_sample_size(self, n_samples):
        """Warn if the training set exceeds what the estimator handles well."""
        limit = self.MAX_SAMPLES.get(self.model_name)
        if limit is not None and n_samples > limit:
            print(
                f"⚠️ '{self.model_name}' was given {n_samples:,} training samples,"
                f" which is above the recommended {limit:,}. Consider reducing"
                " `n_samples` of the Trainer to keep runtime/memory in check."
            )
        return limit

    # ------------------------------------------------------------------ #
    # inspection
    # ------------------------------------------------------------------ #
    def summarize(self, verbose=True):
        """
        Print a summary of the model configuration and its complexity.

        Args:
            verbose (bool): whether to print the summary or not
        """
        if not hasattr(self, "model"):
            raise RuntimeError("❌ Call init_model() before summarize().")
        estimator = getattr(self.model, "estimator", self.model)
        lines = [f"Model: {type(estimator).__name__} ({self.library})"]
        lines.append("Parameters:")
        lines += [f"  {k:<24}{v}" for k, v in sorted(self.params.items())]
        if isinstance(self.model, ResampledClassifier):
            lines.append(f"Class resampling: {self.model.class_weights or 'disabled'}")
        self.summary = "\n".join(lines)
        if verbose:
            print(self.summary)
        return None

    @classmethod
    def describe(cls, model_name=None):
        """Print defaults and search spaces of one or all available models."""
        names = [model_name] if model_name else list(cls.MODELS)
        for name in names:
            print(f"\n=== {name} ({cls.MODELS[name]}) ===")
            print("default configuration (tuning=None):")
            for k, v in cls.DEFAULTS[name].items():
                print(f"  {k:<24}{v}")
            print("grid search space:")
            for k, v in cls.GRIDS[name].items():
                print(f"  {k:<24}{v}")
            print("randomised search space:")
            for k, v in cls.DISTRIBUTIONS[name].items():
                label = v if isinstance(v, list) else type(v).__name__
                print(f"  {k:<24}{label}")
