#!/usr/bin/env bash
# ===========================================================================
# env_setup.sh - create the conda environment for the AI4Habitats benchmark
# ===========================================================================
#
# The benchmark trains PyTorch (patchwise, pixelwise) and TensorFlow (pixelwise)
# models plus RAPIDS cuML models in ONE environment, so all three must share
# CUDA-compatible builds.
#
# Requirements
#   Linux x86_64, an NVIDIA GPU with a driver supporting CUDA 12.4, conda (e.g.
#   Miniforge) and git. Run it on a GPU node: the final checks import cuDF/CuPy.
#   Afterwards, `python env_verify.py` tests that both frameworks see the GPU.
#
# Example Usage
#   ENV_NAME=ai4h-test FORCE_RECREATE=1 bash env_setup.sh
#
# Optional environment variables
#   ENV_NAME        name of the conda environment               [ai4habitats]
#   FORCE_RECREATE  1 = delete an existing env of that name;
#                   otherwise the script stops instead           [0]
#   CONDA_SH        conda's profile script     [$(conda info --base)/etc/profile.d/conda.sh]
#   SKTIME_DIR      existing checkout of the sktime fork. It is installed
#                   editable and used as is (no checkout/pull). If unset, the
#                   fork is cloned at SKTIME_REF into a temporary directory
#                   (under $TMPDIR), installed non-editable, and the clone is
#                   deleted when the script exits.
#   SKTIME_REF      commit/tag/branch of the fork when cloning   [see below]
#   LOCK_FILE       where the exact `pip freeze` lock is written [./${ENV_NAME}-lock.txt]
#
# Steps
#   1. (re)create the conda env with Python 3.12, NumPy and pandas
#   2. install PyTorch + cuML via conda and TensorFlow via pip, then freeze the
#      resulting CUDA/NVIDIA package set as pip constraints, so that no later
#      install can replace it
#   3. install the sktime fork (deep learning extras)
#   4. install the remaining project packages and the torchmetrics commit with
#      the classification report (not yet released)
#   5. link the pip-installed NVIDIA libraries where TensorFlow finds them
#   6. make TensorFlow allocate GPU memory on demand (conda activate hook)
#   7. check imports/versions, register a Jupyter kernel, write the lock file
# ===========================================================================
set -Eeo pipefail

ENV_NAME="${ENV_NAME:-ai4habitats}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
SKTIME_URL="https://github.com/fkroeber/sktime.git"
SKTIME_REF="${SKTIME_REF:-5f433f885563e992d3799d53561feb0a759a1ec3}"
LOCK_FILE="$(realpath -m "${LOCK_FILE:-$PWD/$ENV_NAME-lock.txt}")"

if [[ -z "${CONDA_SH:-}" ]]; then
    if ! command -v conda >/dev/null; then
        echo "ERROR: conda not found. Load/install conda or set CONDA_SH=/path/to/etc/profile.d/conda.sh"
        exit 1
    fi
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi

# scratch space for constraint files (and the sktime clone), removed on exit
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/$ENV_NAME-setup.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
BASE_CONSTRAINTS="$WORK_DIR/base-constraints.txt"
GPU_CONSTRAINTS="$WORK_DIR/gpu-constraints.txt"


# ---------------------------------------------------------------------------
# 0. Re-define pip_check to ignore known conda/RAPIDS metadata artefacts
# ---------------------------------------------------------------------------
pip_check() {
    local output real

    if output="$(python -m pip check 2>&1)"; then
        echo "$output"
        return 0
    fi

    # Conda-provided RAPIDS packages carry wheel-style Requires-Dist metadata,
    # so pip reports conda-satisfied dependencies as missing. Expected.
    real="$(
        printf '%s\n' "$output" |
            grep -vE 'requires (cupy-cuda12x|cuvs|libcudf|libcuml|libraft|librmm|libucxx|pynvjitlink|rapids-dask-dependency|nvidia-(cublas|cufft|curand|cusolver|cusparse)), which is not installed' \
            || true
    )"

    if [[ -n "$real" ]]; then
        echo "pip check reported real problems:"
        printf '%s\n' "$real"
        return 1
    fi

    echo "pip check: only known conda/RAPIDS metadata artefacts, continuing."
    return 0
}

# ---------------------------------------------------------------------------
# 1. Initialize Conda and recreate the environment
# ---------------------------------------------------------------------------

source "$CONDA_SH"

conda deactivate 2>/dev/null || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [[ "$FORCE_RECREATE" != "1" ]]; then
        echo "ERROR: conda environment '$ENV_NAME' already exists."
        echo "Re-run with FORCE_RECREATE=1 to replace it, or choose another ENV_NAME."
        exit 1
    fi
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
fi

conda create \
    -n "$ENV_NAME" \
    python=3.12 \
    pip \
    numpy=1.26.4 \
    pandas=2.2.3 \
    -c conda-forge \
    -y

conda activate "$ENV_NAME"

python -m pip install --upgrade pip wheel
python -m pip install "setuptools==81.0.0"
python -m pip install "protobuf==3.20.3"

# ---------------------------------------------------------------------------
# 2. Pin the shared framework stack
# ---------------------------------------------------------------------------

cat > "$BASE_CONSTRAINTS" <<'EOF'
matplotlib==3.10.8
numpy==1.26.4
pandas==2.2.3
setuptools==81.0.0
protobuf==3.20.3

torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1

tensorflow==2.16.2
keras==3.15.0
EOF

# Install PyTorch through Conda so that its CUDA libraries do not conflict
# with the exact pip-level NVIDIA package versions required by TensorFlow.
#
# PyTorch 2.4.1 uses CUDA 12.4 and cuDNN 9, while TensorFlow 2.16.2 uses
# its own pip-managed CUDA 12.3 and cuDNN 8 libraries.
conda install \
    pytorch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    pytorch-cuda=12.4 \
    cuml=25.08 \
    cuda-version=12.4 \
    scikit-learn=1.6 \
    fsspec \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    -c rapidsai \
    --solver=libmamba \
    -y

# TensorFlow 2.16.2 satisfies sktime's requirement:
# tensorflow>=2,<2.17
python -m pip install \
    -c "$BASE_CONSTRAINTS" \
    "tensorflow[and-cuda]==2.16.2" \
    "keras==3.15.0" \
    keras-self-attention

# Stop immediately if the supposedly compatible core is not compatible.
pip_check

# Lock all NVIDIA packages selected by the successful core installation.
# All subsequent installations are prevented from replacing them.
{
    cat "$BASE_CONSTRAINTS"

    python -m pip freeze |
        grep -E '^(cuda-|nvidia-|triton==)' |
        grep -v '@' || true

    python - <<'PY'
import importlib.metadata as md

for dist in (
    "cuml", "cudf", "rmm", "pylibraft", "raft-dask", "treelite",
    "cupy", "numba", "llvmlite", "scikit-learn", "scipy", "pyarrow",
):
    try:
        print(f"{dist}=={md.version(dist)}")
    except md.PackageNotFoundError:
        pass
PY
} |
    awk '!seen[$0]++' > "$GPU_CONSTRAINTS"

# ---------------------------------------------------------------------------
# 3. Install the sktime fork
# ---------------------------------------------------------------------------
# Both extras are installed in one dependency-resolution operation. The
# constraints ensure they cannot replace torch, TensorFlow, NumPy, or the
# working NVIDIA package set.

if [[ -n "${SKTIME_DIR:-}" ]]; then
    SKTIME_DIR="$(realpath "$SKTIME_DIR")"
    if [[ ! -f "$SKTIME_DIR/pyproject.toml" ]]; then
        echo "ERROR: SKTIME_DIR=$SKTIME_DIR is not an sktime checkout (no pyproject.toml)."
        exit 1
    fi
    echo "Installing existing sktime checkout (editable, as is): $SKTIME_DIR" \
        "@ $(git -C "$SKTIME_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown commit')"
    python -m pip install \
        -c "$GPU_CONSTRAINTS" \
        -e "$SKTIME_DIR[all-extras,dl]"
else
    # temporary clone; non-editable install copies the package into the env,
    # so removing the clone (trap on EXIT) does not break the installation
    echo "Installing sktime fork $SKTIME_URL @ $SKTIME_REF (temporary clone)"
    git clone --quiet "$SKTIME_URL" "$WORK_DIR/sktime"
    git -C "$WORK_DIR/sktime" -c advice.detachedHead=false checkout --quiet "$SKTIME_REF"
    python -m pip install \
        -c "$GPU_CONSTRAINTS" \
        "$WORK_DIR/sktime[all-extras,dl]"
fi

# ---------------------------------------------------------------------------
# 4. Remaining project packages
# ---------------------------------------------------------------------------

python -m pip install \
    -c "$GPU_CONSTRAINTS" \
    "albumentations==2.0.8" \
    geocube \
    geopandas \
    jupyterlab \
    ipykernel \
    ipywidgets \
    nbformat \
    rioxarray \
    segmentation_models_pytorch \
    torchinfo \
    wandb

# Install torchmetrics PR with classification report capability, which is not yet released.
python -m pip install \
    --no-build-isolation \
    -c "$GPU_CONSTRAINTS" \
    "git+https://github.com/Lightning-AI/torchmetrics.git@820c56d66a6d1496f22e68f59ee0bd31741ec9fa"

# ---------------------------------------------------------------------------
# 5. Make pip-installed NVIDIA libraries discoverable by TensorFlow
# ---------------------------------------------------------------------------

SITE_PACKAGES="$(
    python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

TF_DIR="$(
    python - <<'PY'
from pathlib import Path
import tensorflow as tf

print(Path(tf.__file__).resolve().parent)
PY
)"

shopt -s nullglob

NVIDIA_LIBS=(
    "$SITE_PACKAGES"/nvidia/*/lib/*.so*
)

if (( ${#NVIDIA_LIBS[@]} == 0 )); then
    echo "ERROR: No pip-installed NVIDIA shared libraries found."
    exit 1
fi

echo "Linking ${#NVIDIA_LIBS[@]} NVIDIA library entries into:"
echo "  $TF_DIR"

for library in "${NVIDIA_LIBS[@]}"; do
    ln -sfn "$library" "$TF_DIR/$(basename "$library")"
done

PTXAS="$(
    find "$SITE_PACKAGES/nvidia/cuda_nvcc" \
        -type f \
        -path '*/bin/ptxas' \
        -print \
        -quit
)"

if [[ -z "$PTXAS" ]]; then
    echo "ERROR: ptxas was not found in nvidia-cuda-nvcc-cu12."
    exit 1
fi

ln -sfn "$PTXAS" "$CONDA_PREFIX/bin/ptxas"

# ---------------------------------------------------------------------------
# 6. Avoid TensorFlow reserving all GPU memory
# ---------------------------------------------------------------------------

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/$ENV_NAME-gpu.sh" <<'EOF'
export TF_FORCE_GPU_ALLOW_GROWTH=true
EOF

export TF_FORCE_GPU_ALLOW_GROWTH=true

# ---------------------------------------------------------------------------
# 7. Static validation
# ---------------------------------------------------------------------------

pip_check

python - <<'PY'
import cudf
import cuml
import cupy
import numpy
import pandas
import sktime
import torch
import torchvision
import tensorflow as tf
import segmentation_models_pytorch
import albumentations
import geopandas
import rioxarray
import wandb
from keras_self_attention import SeqSelfAttention

assert torch.__version__.startswith("2.4.1"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert tf.__version__.startswith("2.16.2"), tf.__version__
assert tf.test.is_built_with_cuda(), "TensorFlow is not a CUDA-enabled build"

print("Static installation checks passed")
print("NumPy:          ", numpy.__version__)
print("pandas:         ", pandas.__version__)
print("sktime:         ", sktime.__version__)
print("cuML:           ", cuml.__version__)
print("cuDF:           ", cudf.__version__)
print("CuPy:           ", cupy.__version__)
print("PyTorch:        ", torch.__version__)
print("PyTorch CUDA:   ", torch.version.cuda)
print("torchvision:    ", torchvision.__version__)
print("TensorFlow:     ", tf.__version__)
print("TF CUDA build:  ", tf.test.is_built_with_cuda())
PY

python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

python -m pip freeze > "$LOCK_FILE"

echo
echo "Installation completed."
echo "Exact package lock written to:"
echo "  $LOCK_FILE"