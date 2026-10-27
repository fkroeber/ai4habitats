# Environment setup

The benchmark trains PyTorch, TensorFlow and RAPIDS cuML models from one environment, so all three need mutually compatible CUDA builds. `env_setup.sh` creates that environment with conda.

**Requirements:** Linux x86_64, an NVIDIA GPU with a driver supporting CUDA 12.4, conda (e.g. [Miniforge](https://github.com/conda-forge/miniforge)) and git.

```bash
bash setup/env_setup.sh                                   # creates the env "ai4habitats"
ENV_NAME=ai4h FORCE_RECREATE=1 bash setup/env_setup.sh    # other name / replace an existing env
SKTIME_DIR=~/src/sktime bash setup/env_setup.sh           # use your own checkout of the sktime fork
```

| Variable | Meaning | Default |
|---|---|---|
| `ENV_NAME` | name of the conda environment | `ai4habitats` |
| `FORCE_RECREATE` | `1` replaces an existing environment of that name | `0` (stop instead) |
| `CONDA_SH` | conda's `etc/profile.d/conda.sh` | derived from `conda info --base` |
| `SKTIME_DIR` | existing checkout of the [sktime fork](https://github.com/fkroeber/sktime), installed editable | unset: temporary clone, removed afterwards |
| `SKTIME_REF` | commit of the fork to install when cloning | pinned in the script |
| `LOCK_FILE` | where the exact `pip freeze` is written | `./<ENV_NAME>-lock.txt` |

Afterwards, check on a GPU node that both frameworks see the GPU:

```bash
conda activate ai4habitats
python setup/env_verify.py
```

The script header of `env_setup.sh` describes the individual installation steps.
